import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time

# API endpoint - make sure this matches your backend URL
# API configuration
import os

# Get API URL from environment variable or use default
API_URL = os.environ.get('API_URL', 'http://127.0.0.1:5000/api')

# Enable detailed error logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure requests session
import requests
session = requests.Session()
session.headers.update({
    'Content-Type': 'application/json',
    'Accept': 'application/json'
})

# --- Authentication Functions ---
def register_user(email, password):
    """
    Register a new user with the provided email and password.
    Returns True if registration is successful, False otherwise.
    """
    try:
        logger.info(f"Attempting to register user: {email}")
        register_url = f"{API_URL}/register"
        
        # Prepare the request data
        register_data = {
            'email': email,
            'password': password
        }
        
        logger.info(f"Sending registration request to: {register_url}")
        logger.debug(f"Request data: {register_data}")
        
        # Make the request using the session
        response = session.post(
            register_url,
            json=register_data,
            timeout=10
        )
        
        # Log response details
        logger.info(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        # Parse response
        try:
            response_data = response.json()
            logger.debug(f"Response data: {response_data}")
        except ValueError as ve:
            logger.error(f"Failed to parse JSON response: {ve}")
            logger.error(f"Raw response: {response.text}")
            st.error("Invalid response format from server. Please try again.")
            return False
        
        # Handle successful registration
        if response.status_code == 201:  # 201 Created is standard for successful registration
            success_msg = response_data.get('message', 'Registration successful! Please log in.')
            logger.info(f"Registration successful for user: {email}")
            st.success(success_msg)
            return True
        else:
            # Handle different error status codes
            error_msg = response_data.get('message', 'Registration failed. Please try again.')
            if response.status_code == 400:
                error_msg = response_data.get('error', 'Invalid registration data. Please check your input.')
            elif response.status_code == 409:
                error_msg = "An account with this email already exists. Please log in instead."
            elif response.status_code >= 500:
                error_msg = "Server error during registration. Please try again later."
                
            logger.error(f"Registration failed (HTTP {response.status_code}): {error_msg}")
            st.error(error_msg)
            return False
            
    except requests.exceptions.Timeout:
        error_msg = "Registration request timed out. The server is taking too long to respond."
        logger.error(error_msg)
        st.error(error_msg)
        return False
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to connect to the server: {str(e)}"
        logger.error(f"Network error during registration: {str(e)}", exc_info=True)
        st.error("Unable to connect to the server. Please check your internet connection and try again.")
        return False
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during registration: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error("An unexpected error occurred. Please try again.")
        return False

def login_user(email, password):
    """
    Attempt to log in a user with the provided email and password.
    Returns True if login is successful, False otherwise.
    """
    try:
        logger.info(f"Attempting to login user: {email}")
        login_url = f"{API_URL}/login"
        
        # Prepare the request data
        login_data = {
            'email': email,
            'password': password
        }
        
        logger.info(f"Sending login request to: {login_url}")
        logger.debug(f"Request data: {login_data}")
        
        # Make the request using the session
        response = session.post(
            login_url,
            json=login_data,
            timeout=10
        )
        
        # Log response details
        logger.info(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        # Parse response
        try:
            response_data = response.json()
            logger.debug(f"Response data: {response_data}")
        except ValueError as ve:
            logger.error(f"Failed to parse JSON response: {ve}")
            logger.error(f"Raw response: {response.text}")
            st.error("Invalid response format from server. Please try again.")
            return False
        
        # Handle successful login
        if response.status_code == 200:
            token = response_data.get('token')
            if token:
                # Store token and user info in session state
                st.session_state.token = token
                st.session_state.logged_in = True
                st.session_state.user_email = email
                
                # Update session headers with the new token
                session.headers.update({
                    'Authorization': f'Bearer {token}'
                })
                
                logger.info(f"Login successful for user: {email}")
                st.success("Login successful!")
                st.experimental_rerun()
                return True
            else:
                error_msg = "Login failed: No authentication token received"
                logger.error(error_msg)
                st.error("Login failed. Please try again.")
        else:
            # Handle different error status codes
            error_msg = response_data.get('message', 'Login failed. Please check your credentials.')
            if response.status_code == 401:
                error_msg = "Invalid email or password. Please try again."
            elif response.status_code >= 500:
                error_msg = "Server error. Please try again later."
                
            logger.error(f"Login failed (HTTP {response.status_code}): {error_msg}")
            st.error(error_msg)
        
        return False
        
    except requests.exceptions.Timeout:
        error_msg = "Login request timed out. The server is taking too long to respond."
        logger.error(error_msg)
        st.error(error_msg)
        return False
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to connect to the server: {str(e)}"
        logger.error(f"Network error during login: {str(e)}", exc_info=True)
        st.error("Unable to connect to the server. Please check your internet connection and try again.")
        return False
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during login: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error("An unexpected error occurred. Please try again.")
        return False

def logout_user():
    st.session_state.token = None
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.success("Logged out successfully!")
    time.sleep(1) # Brief pause for user to see message
    st.experimental_rerun()

# --- API Call for Prediction ---
def call_predict_api(uploaded_file, token):
    headers = {'x-access-token': token}
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    print(f"Sending prediction request for file: {uploaded_file.name}")  # Debug log
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            headers=headers,
            timeout=30  # Increased timeout for file uploads
        )
        response.raise_for_status()
        result = response.json()
        print("Prediction API response received")  # Debug log
        return result
        
    except requests.exceptions.Timeout:
        error_msg = "Prediction request timed out. The server took too long to respond."
        print(error_msg)  # Debug log
        st.error(error_msg)
        return None
        
    except requests.exceptions.RequestException as e:
        error_message = "Prediction request failed. "
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_message += error_details.get('error', 
                    error_details.get('message', str(e.response.text)))
            except ValueError:
                error_message += e.response.text or str(e)
        else:
            error_message += str(e)
            
        print(f"Prediction error: {error_message}")  # Debug log
        st.error(error_message)
        return None
        
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {str(e)}"
        print(error_msg)  # Debug log
        st.error(error_msg)
        return None

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Diabetes-Check", page_icon="🏥", layout="wide")

    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None

    st.sidebar.title("Authentication")
    if st.session_state.logged_in:
        st.sidebar.write(f"Welcome, {st.session_state.user_email}!")
        if st.sidebar.button("Logout"):
            logout_user()
    else:
        auth_choice = st.sidebar.radio("Choose action:", ("Login", "Register"))
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type="password")

        if auth_choice == "Login":
            if st.sidebar.button("Login"):
                if email and password:
                    login_user(email, password)
                else:
                    st.sidebar.warning("Please enter both email and password.")
        elif auth_choice == "Register":
            if st.sidebar.button("Register"):
                if email and password:
                    register_user(email, password)
                else:
                    st.sidebar.warning("Please enter both email and password.")

    st.title("Diabetes-Check")
    st.subheader("Medical Report Analysis for Diabetes Risk Assessment")

    if st.session_state.logged_in:
        st.header("Upload Medical Report")
        uploaded_file = st.file_uploader("Choose a medical report (PDF or image)", 
                                       type=['pdf', 'png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            if st.button("Analyze Report"):
                with st.spinner('Analyzing report...'):
                    response = call_predict_api(uploaded_file, st.session_state.token)
                
                if response:
                    st.header("Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Diabetes Risk Assessment")
                        risk_prob = response.get("prediction", 0) * 100
                        
                        if risk_prob < 30:
                            st.success(f"Low Risk: {risk_prob:.1f}%")
                        elif risk_prob < 70:
                            st.warning(f"Moderate Risk: {risk_prob:.1f}%")
                        else:
                            st.error(f"High Risk: {risk_prob:.1f}%")
                        
                        confidence = response.get("confidence_score")
                        if confidence is not None:
                             st.metric("Analysis Confidence", f"{float(confidence):.1f}%")
                        
                        if response.get("diabetes_mentioned"):
                            st.warning("Diabetes mentioned in the report")
                            if response.get("diabetes_type"):
                                st.info(f"Type: {response['diabetes_type']}")
                        
                        symptoms = response.get("symptoms_present")
                        if symptoms and isinstance(symptoms, list):
                            st.subheader("Detected Symptoms")
                            for symptom in symptoms:
                                st.write(f"• {symptom}")
                    
                    with col2:
                        st.subheader("Clinical Features")
                        features_data = {k: v for k, v in response.items() 
                                      if k in ['glucose', 'hba1c', 'insulin', 'bmi', 
                                             'systolic', 'diastolic'] and v is not None}
                        
                        if features_data:
                            df_features = pd.DataFrame(list(features_data.items()), 
                                            columns=['Feature', 'Value'])
                            st.dataframe(df_features)
                            
                        if "feature_importance" in response and response["feature_importance"]:
                            st.subheader("Feature Importance")
                            importance_data = response["feature_importance"]
                            # Convert dict to DataFrame for Plotly
                            df_importance = pd.DataFrame(list(importance_data.items()), 
                                                         columns=['Feature', 'Importance'])
                            df_importance = df_importance.sort_values(by="Importance", ascending=False)
                            fig = px.bar(df_importance, 
                                       x='Feature', 
                                       y='Importance',
                                       title='Feature Importance in Prediction')
                            st.plotly_chart(fig)
                    
                    # AI Analysis Section
                    if "ai_analysis" in response and response["ai_analysis"]:
                        st.subheader("AI-Powered Analysis")
                        st.markdown("---")
                        st.markdown(response["ai_analysis"])
                        st.markdown("---")
                    
                    with st.expander("View Extracted Text"):
                        st.text(response.get("extracted_text", "No text extracted or available."))
    else:
        st.info("Please log in or register using the sidebar to upload and analyze reports.")

if __name__ == "__main__":
    main()