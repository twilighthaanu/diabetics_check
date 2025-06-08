# Diabetes-Check

A web application that analyzes medical reports to predict the likelihood of Type 2 Diabetes using machine learning.

## Project Overview

Diabetes-Check is a minimum-viable web application that helps identify potential Type 2 Diabetes risk from medical reports. The application uses OCR and machine learning to extract clinical features and provide risk assessment with explainable AI insights.

### Key Features

- User authentication (email/password)
- Medical report upload (PDF/Image)
- OCR-based feature extraction
- Diabetes risk prediction using XGBoost
- SHAP-based feature importance visualization
- Interactive web interface

## Technical Architecture

### Frontend
- Streamlit-based web interface
- Responsive design for medical report upload
- Interactive visualization of results
- User authentication and session management

### Backend
- Flask REST API
- JWT-based authentication
- OCR processing using pytesseract and pdfplumber
- Feature extraction and validation
- XGBoost model inference
- SHAP-based model interpretation

### AI Pipeline
1. **Data Extraction**
   - OCR processing of medical reports
   - Feature extraction from text
   - Validation of clinical values

2. **Model Architecture**
   - XGBoost classifier
   - Features: glucose, BMI, age, insulin, blood pressure
   - SHAP-based feature importance
   - Probability-based risk assessment

3. **Model Choice Rationale**
   - XGBoost was chosen for its:
     - High performance on tabular medical data
     - Built-in feature importance
     - Fast inference time
     - Good handling of missing values
     - Excellent integration with SHAP for interpretability

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-check.git
cd diabetes-check
```

2. Build and run using Docker:
```bash
docker-compose up --build
```

3. Access the application:
- Frontend: http://localhost:8501
- Backend API: http://localhost:5000

## Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
# Terminal 1 - Backend
cd src/api
python app.py

# Terminal 2 - Frontend
cd src/frontend
streamlit run app.py
```

## Model Details

The application uses an XGBoost classifier trained on clinical features extracted from medical reports. The model predicts the probability of Type 2 Diabetes based on features such as:

- Glucose levels
- BMI
- Age
- Blood pressure
- Insulin levels
- Other clinical markers

### Model Training
- Synthetic data generation for demonstration
- Real-world implementation would use:
  - Medical records dataset
  - Cross-validation
  - Hyperparameter tuning
  - Regular model retraining

## Future Roadmap

### Short-term Improvements
1. **Data Processing**
   - Support for more medical report formats
   - Enhanced OCR accuracy
   - Additional feature extraction

2. **Model Enhancement**
   - Integration with real medical datasets
   - Model retraining pipeline
   - Additional risk factors

3. **User Experience**
   - Report history
   - Trend analysis
   - PDF report generation

### Long-term Vision
1. **Advanced Features**
   - Multi-modal analysis (images + text)
   - Integration with EMR systems
   - Real-time monitoring

2. **Clinical Integration**
   - HIPAA compliance
   - Doctor dashboard
   - Patient management

3. **Research Integration**
   - Research data collection
   - Clinical trial support
   - Population health insights

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 