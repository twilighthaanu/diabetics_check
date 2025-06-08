from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
import os
import uuid
import logging
import sys
from werkzeug.utils import secure_filename
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
logger.info(f"Base directory: {BASE_DIR}")

try:
    from src.utils.feature_extraction import FeatureExtractor
    from src.models.predictor import DiabetesPredictor
    from src.utils.openai_client import AzureOpenAIClient
    logger.info("All required modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    raise

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_default_secret_key') # Use environment variable in production
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Initialize AI client
try:
    openai_client = AzureOpenAIClient()
except Exception as e:
    print(f"Warning: Could not initialize Azure OpenAI client: {str(e)}")
    openai_client = None

# In-memory user store (replace with a database in production)
users = {}

# Initialize feature extractor and model
feature_extractor = FeatureExtractor()
model = DiabetesPredictor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Enable CORS
cors = CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ["*"],  # Allow all origins for development
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "x-access-token", "Authorization"],
            "supports_credentials": True
        }
    },
    supports_credentials=True
)

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,x-access-token')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Decorator for token requirement
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = users.get(data['email'])
            if not current_user:
                 return jsonify({'message': 'User not found!'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required!'}), 400
    if email in users:
        return jsonify({'message': 'User already exists!'}), 409

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    users[email] = {'email': email, 'password': hashed_password}
    return jsonify({'message': 'Registered successfully!'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    auth = request.get_json()
    if not auth or not auth.get('email') or not auth.get('password'):
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})

    user = users.get(auth.get('email'))
    if not user or not check_password_hash(user['password'], auth.get('password')):
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Invalid credentials!"'})

    token = jwt.encode({
        'email': user['email'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm="HS256")

    return jsonify({'token': token})

@app.route('/api/predict', methods=['POST'])
@token_required
def predict(current_user):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Ensure unique filenames to prevent overwrites if multiple users upload same filename
        unique_filename = str(uuid.uuid4()) + "_" + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(file_path)
            logger.info(f"File saved to {file_path} by user {current_user['email']}")

            # Extract features from the report
            extracted_features = feature_extractor.extract_features(file_path)
            
            if not extracted_features:
                return jsonify({"error": "No features could be extracted from the report"}), 400
            
            # Make prediction
            prob, feature_importance_dict = model.predict(extracted_features)
            
            # Prepare response
            response = {
                'prediction': prob,
                'feature_importance': feature_importance_dict,
                'extracted_text': getattr(feature_extractor, 'extracted_text', None),
                'diabetes_mentioned': getattr(feature_extractor, 'diabetes_mentioned', None),
                'diabetes_type': getattr(feature_extractor, 'diabetes_type', None),
                'symptoms_present': getattr(feature_extractor, 'symptoms_present', None),
                'glucose': getattr(feature_extractor, 'glucose', None),
                'hba1c': getattr(feature_extractor, 'hba1c', None),
                'insulin': getattr(feature_extractor, 'insulin', None),
                'bmi': getattr(feature_extractor, 'bmi', None),
                'systolic': getattr(feature_extractor, 'systolic', None),
                'diastolic': getattr(feature_extractor, 'diastolic', None),
                'confidence_score': getattr(feature_extractor, 'confidence_score', None)
            }
            
            # Add AI analysis if text was extracted
            if feature_extractor.extracted_text:
                clinical_features = {k: v for k, v in response.items() 
                                  if k in ['glucose', 'hba1c', 'insulin', 'bmi', 
                                         'systolic', 'diastolic'] and v is not None}
                response['ai_analysis'] = openai_client.analyze_report(
                    feature_extractor.extracted_text,
                    clinical_features
                )
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error processing request for user {current_user['email']}: {str(e)}")
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Successfully removed temp file: {file_path}")
                except Exception as e_remove:
                    logger.error(f"Error removing temp file {file_path}: {str(e_remove)}")
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 