import numpy as np
import xgboost as xgb
import shap
import joblib
from typing import Dict, Tuple, List
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'glucose', 'bmi', 'age', 'insulin',
            'systolic', 'diastolic'
        ]
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the XGBoost model with synthetic data."""
        # Generate synthetic data for demonstration
        np.random.seed(42)
        X = np.random.rand(100, 10)  # 100 samples, 10 features
        y = np.random.randint(0, 2, 100)  # Binary classification

        # Train a simple XGBoost model
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X, y)

        # Initialize SHAP explainer with numpy compatibility
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {str(e)}")
            self.explainer = None

    def predict(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Make prediction and generate SHAP values for feature importance.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (probability, feature_importance)
        """
        # Convert features to model input format
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        
        # Get prediction probability
        prob = self.model.predict_proba(X)[0][1]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)[0]
        
        # Create feature importance dictionary
        feature_importance = dict(zip(self.feature_names, np.abs(shap_values)))
        
        return prob, feature_importance

    def get_feature_importance_plot(self, features: Dict[str, float]) -> str:
        """
        Generate SHAP feature importance plot.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Path to saved plot image
        """
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        
        # Create SHAP plot
        plt = shap.force_plot(
            self.explainer.expected_value,
            self.explainer.shap_values(X)[0],
            X,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        # Save plot
        plot_path = os.path.join('data', 'feature_importance.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        return plot_path 