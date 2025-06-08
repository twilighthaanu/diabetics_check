import pytesseract
import pdfplumber
# Use try-except for PyMuPDF import to handle different installation methods
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import PyMuPDF as fitz
    except ImportError:
        fitz = None
        print("Warning: PyMuPDF (fitz) not found. PDF processing will be limited.")

from PIL import Image
import re
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import io
import logging
from pdf2image import convert_from_path
import cv2
import os


class FeatureExtractor:
    def __init__(self):
        # Define diabetes-related keywords and their variations
        self.diabetes_keywords = {
            'diabetes': ['diabetes', 'diabetic', 'diabetes mellitus', 'type 2', 'type ii', 't2dm'],
            'glucose': ['glucose', 'blood sugar', 'blood glucose', 'fasting glucose', 'fg', 'fbg'],
            'hba1c': ['hba1c', 'a1c', 'glycated hemoglobin', 'hemoglobin a1c'],
            'insulin': ['insulin', 'insulin level', 'fasting insulin'],
            'metabolic': ['metabolic syndrome', 'insulin resistance', 'prediabetes', 'impaired glucose'],
            'symptoms': ['polyuria', 'polydipsia', 'polyphagia', 'fatigue', 'blurred vision', 'slow healing']
        }
        
        # Define patterns for extracting numerical values
        self.value_patterns = {
            'glucose': r'(?i)(?:glucose|blood sugar|blood glucose|fasting glucose|fg|fbg)[\s:]*(\d+(?:\.\d+)?)\s*(?:mg/dl|mmol/l)?',
            'hba1c': r'(?i)(?:hba1c|a1c|glycated hemoglobin)[\s:]*(\d+(?:\.\d+)?)\s*(?:%|percent)?',
            'insulin': r'(?i)(?:insulin|fasting insulin)[\s:]*(\d+(?:\.\d+)?)\s*(?:µU/ml|mU/l)?',
            'bmi': r'(?i)(?:bmi|body mass index)[\s:]*(\d+(?:\.\d+)?)',
            'age': r'(?i)(?:age|patient age)[\s:]*(\d+)',
            'systolic': r'(?i)(?:systolic|systolic blood pressure)[\s:]*(\d+)',
            'diastolic': r'(?i)(?:diastolic|diastolic blood pressure)[\s:]*(\d+)'
        }
        
        # Define normal ranges for validation
        self.normal_ranges = {
            'glucose': (70, 140),  # mg/dl
            'hba1c': (4.0, 5.6),   # %
            'insulin': (2.6, 24.9), # µU/ml
            'bmi': (18.5, 24.9),
            'age': (0, 120),
            'systolic': (90, 140),
            'diastolic': (60, 90)
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced

    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract features from an image file with enhanced processing."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not read image file")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Save processed image temporarily for OCR
            temp_path = "temp_processed.png"
            cv2.imwrite(temp_path, processed_image)
            
            # Perform OCR with different configurations
            text_results = []
            
            # Try different OCR configurations
            configs = [
                '--oem 3 --psm 6',  # Assume uniform text block
                '--oem 3 --psm 4',  # Assume single column of text
                '--oem 3 --psm 3'   # Fully automatic page segmentation
            ]
            
            for config in configs:
                text = pytesseract.image_to_string(processed_image, config=config)
                if text.strip():
                    text_results.append(text)
            
            # Combine results
            combined_text = "\n".join(text_results)
            
            # Extract features from text
            features = self._parse_features(combined_text)
            
            # Add image analysis results
            image_features = self._analyze_image(image)
            features.update(image_features)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return features
            
        except Exception as e:
            logging.error(f"Error in image processing: {str(e)}")
            return {}

    def _analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image for visual indicators of diabetes-related information."""
        features = {}
        
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Look for highlighted or colored regions (common in medical reports)
            # Define color ranges for highlighting
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255])
            
            # Create masks for highlighted regions
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            
            # Calculate percentage of highlighted regions
            total_pixels = image.shape[0] * image.shape[1]
            yellow_percentage = (np.sum(yellow_mask > 0) / total_pixels) * 100
            red_percentage = (np.sum(red_mask > 0) / total_pixels) * 100
            
            # Add to features if significant highlighting is found
            if yellow_percentage > 1.0:
                features['highlighted_regions'] = True
                features['highlight_color'] = 'yellow'
            elif red_percentage > 1.0:
                features['highlighted_regions'] = True
                features['highlight_color'] = 'red'
            
            # Look for graph-like structures (common in medical reports)
            edges = cv2.Canny(image, 100, 200)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 10:
                features['contains_graphs'] = True
            
            # Look for table structures
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangular_contours = [cnt for cnt in contours if self._is_rectangular(cnt)]
            
            if len(rectangular_contours) > 5:
                features['contains_tables'] = True
            
        except Exception as e:
            logging.error(f"Error in image analysis: {str(e)}")
        
        return features

    def _is_rectangular(self, contour) -> bool:
        """Check if a contour is approximately rectangular."""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        return len(approx) == 4

    def extract_from_pdf(self, pdf_data: bytes) -> Dict[str, float]:
        """Extract features from a PDF file."""
        try:
            # Try pdfplumber first
            try:
                with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except:
                # Fallback to PyMuPDF
                doc = fitz.open(stream=pdf_data, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()

            return self._parse_features(text)
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def _parse_features(self, text: str) -> Dict[str, Any]:
        """Extract and parse features from text."""
        features = {}
        
        # Extract numerical values
        for feature, pattern in self.value_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                try:
                    value = float(matches[0])
                    if self._validate_feature(feature, value):
                        features[feature] = value
                except ValueError:
                    continue

        # Extract diabetes-related information
        diabetes_info = self._extract_diabetes_info(text)
        features.update(diabetes_info)
        
        return features

    def _extract_diabetes_info(self, text: str) -> Dict[str, Any]:
        """Extract diabetes-related information from text."""
        diabetes_info = {
            'diabetes_mentioned': False,
            'diabetes_type': None,
            'symptoms_present': [],
            'risk_factors': []
        }
        
        # Check for diabetes mentions
        for keyword, variations in self.diabetes_keywords.items():
            for variation in variations:
                if re.search(rf'\b{variation}\b', text.lower()):
                    if keyword == 'diabetes':
                        diabetes_info['diabetes_mentioned'] = True
                        # Try to determine diabetes type
                        if re.search(r'\btype\s*2\b|\bt2dm\b', text.lower()):
                            diabetes_info['diabetes_type'] = 'Type 2'
                        elif re.search(r'\btype\s*1\b|\bt1dm\b', text.lower()):
                            diabetes_info['diabetes_type'] = 'Type 1'
                    elif keyword == 'symptoms':
                        diabetes_info['symptoms_present'].append(variation)
                    elif keyword == 'metabolic':
                        diabetes_info['risk_factors'].append(variation)
        
        return diabetes_info

    def _validate_feature(self, feature: str, value: float) -> bool:
        """Validate if a feature value is within normal range."""
        if feature in self.normal_ranges:
            min_val, max_val = self.normal_ranges[feature]
            return min_val <= value <= max_val
        return True

    def validate_features(self, features: Dict[str, float]) -> bool:
        """Validate if all required features are present and within reasonable ranges."""
        required_features = ['glucose', 'bmi', 'age']
        ranges = {
            'glucose': (50, 500),  # mg/dL
            'bmi': (10, 50),       # kg/m²
            'age': (18, 100),      # years
            'systolic': (70, 200),  # mmHg
            'diastolic': (40, 120), # mmHg
        }

        # Check if all required features are present
        if not all(feature in features for feature in required_features):
            return False

        # Check if values are within reasonable ranges
        for feature, value in features.items():
            if feature in ranges:
                min_val, max_val = ranges[feature]
                if not (min_val <= value <= max_val):
                    return False

        return True

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image file using OCR."""
        try:
            return pytesseract.image_to_string(image_path)
        except Exception as e:
            logging.error(f"Error in OCR processing: {str(e)}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            logging.error(f"Error in PDF processing: {str(e)}")
            return ""

    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """Extract features from a medical report file."""
        try:
            # Determine file type and extract features
            if file_path.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
                features = self._parse_features(text)
            else:
                features = self.extract_from_image(file_path)
            
            # Add confidence score
            features['confidence_score'] = self._calculate_confidence_score(features)
            
            # Add image analysis confidence if available
            if 'highlighted_regions' in features or 'contains_graphs' in features:
                features['image_analysis_confidence'] = self._calculate_image_confidence(features)
            
            return features
            
        except Exception as e:
            logging.error(f"Error in feature extraction: {str(e)}")
            return {}

    def _calculate_confidence_score(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score based on available features and information."""
        score = 0.0
        total_possible = 0.0
        
        # Check for presence of key features
        key_features = ['glucose', 'hba1c', 'insulin', 'bmi']
        for feature in key_features:
            if feature in features:
                score += 1.0
            total_possible += 1.0
        
        # Check for diabetes-related information
        if features.get('diabetes_mentioned'):
            score += 1.0
            if features.get('diabetes_type'):
                score += 0.5
        total_possible += 1.5
        
        # Check for symptoms and risk factors
        if features.get('symptoms_present'):
            score += min(len(features['symptoms_present']) * 0.2, 1.0)
        if features.get('risk_factors'):
            score += min(len(features['risk_factors']) * 0.2, 1.0)
        total_possible += 2.0
        
        return (score / total_possible) * 100 if total_possible > 0 else 0.0

    def _calculate_image_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score based on image analysis results."""
        score = 0.0
        total_possible = 0.0
        
        # Check for highlighted regions
        if features.get('highlighted_regions'):
            score += 1.0
            if features.get('highlight_color') == 'red':
                score += 0.5  # Red highlighting often indicates important medical information
        total_possible += 1.5
        
        # Check for structured content
        if features.get('contains_graphs'):
            score += 1.0
        if features.get('contains_tables'):
            score += 1.0
        total_possible += 2.0
        
        return (score / total_possible) * 100 if total_possible > 0 else 0.0 