import sys
import os
import unittest
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.feature_extraction import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()
        self.sample_report_path = Path(__file__).parent.parent / 'data' / 'sample_report.txt'

    def test_feature_extraction(self):
        # Read sample report
        with open(self.sample_report_path, 'r') as f:
            text = f.read()

        # Extract features
        features = self.extractor._parse_features(text)

        # Verify extracted features
        self.assertIn('glucose', features)
        self.assertIn('bmi', features)
        self.assertIn('age', features)
        self.assertIn('systolic', features)
        self.assertIn('diastolic', features)
        self.assertIn('insulin', features)

        # Verify values
        self.assertEqual(features['glucose'], 110.0)
        self.assertEqual(features['bmi'], 26.1)
        self.assertEqual(features['age'], 45.0)
        self.assertEqual(features['systolic'], 130.0)
        self.assertEqual(features['diastolic'], 85.0)
        self.assertEqual(features['insulin'], 15.0)

    def test_feature_validation(self):
        # Test valid features
        valid_features = {
            'glucose': 110.0,
            'bmi': 26.1,
            'age': 45.0,
            'systolic': 130.0,
            'diastolic': 85.0,
            'insulin': 15.0
        }
        self.assertTrue(self.extractor.validate_features(valid_features))

        # Test invalid features (missing required)
        invalid_features = {
            'glucose': 110.0,
            'bmi': 26.1
        }
        self.assertFalse(self.extractor.validate_features(invalid_features))

        # Test invalid features (out of range)
        out_of_range_features = {
            'glucose': 600.0,  # Too high
            'bmi': 5.0,        # Too low
            'age': 15.0        # Too young
        }
        self.assertFalse(self.extractor.validate_features(out_of_range_features))

if __name__ == '__main__':
    unittest.main() 