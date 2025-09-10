"""
Equipment Failure Prediction from Logs
Jan 2024 – Feb 2024
Python, XGBoost, Random Forest

Modeled robotics-style sensor logs to predict equipment failures before occurrence.
Applied explainable AI (SHAP) and deployed real-time dashboards for monitoring 
operational health.

Required packages:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import shap
from datetime import datetime, timedelta
import warnings
import joblib
import json
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class EquipmentFailurePredictor:
    """Equipment failure prediction system using sensor logs"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = xgb.XGBClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
    def generate_sensor_logs(self, n_samples=10000):
        """Generate synthetic robotics sensor log data"""
        
        print("Generating synthetic sensor log data...")
        
        # Equipment types
        equipment_types = ['Robot_Arm_A', 'Robot_Arm_B', 'Conveyor_1', 'Conveyor_2', 
                          'Gripper_X', 'Gripper_Y', 'Vision_System']
        
        # Generate timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(hours=i/10) for i in range(n_samples)]
        
        data = []
        failure_probability = 0.05  # 5% failure rate
        
        for i in range(n_samples):
            # Randomly select equipment
            equipment_id = np.random.choice(equipment_types)
            
            # Determine if this is a failure case
            is_failure = np.random.random() < failure_probability
            
            # Generate sensor readings based on failure status
            if is_failure:
                # Failure cases - abnormal readings
                temperature = np.random.normal(85, 15)  # Higher temp
                vibration = np.random.normal(12, 4)     # Higher vibration
                current = np.random.normal(8, 2)        # Higher current
                pressure = np.random.normal(45, 10)     # Lower pressure
                speed = np.random.normal(150, 30)       # Irregular speed
                error_count = np.random.poisson(3)      # More errors
                cycle_time = np.random.normal(25, 8)    # Longer cycle time
            else:
                # Normal operation
                temperature = np.random.normal(65, 8)
                vibration = np.random.normal(5, 2)
                current = np.random.normal(5, 1)
                pressure = np.random.normal(60, 5)
                speed = np.random.normal(200, 10)
                error_count = np.random.poisson(0.5)
                cycle_time = np.random.normal(15, 3)
            
            # Add some temporal degradation patterns
            age_factor = (i / n_samples) * 0.3  # Equipment ages over time
            if not is_failure:  # Only apply to normal cases
                temperature += age_factor * 10
                vibration += age_factor * 2
                current += age_factor * 1
            
            # Additional derived features
            power = current * 220  # Power consumption
            efficiency = 100 / max(cycle_time, 1)  # Efficiency metric
            temp_vibration_ratio = temperature / max(vibration, 0.1)
            
            data.append({
                'timestamp': timestamps[i],
                'equipment_id': equipment_id,
                'temperature': max(temperature, 20),  # Min temp
                'vibration': max(vibration, 0),       # Min vibration
                'current': max(current, 0),           # Min current
                'pressure': max(pressure, 0),         # Min pressure
                'speed': max(speed, 0),               # Min speed
                'error_count': max(error_count, 0),   # Min errors
                'cycle_time': max(cycle_time, 5),     # Min cycle time
                'power': power,
                'efficiency': efficiency,
                'temp_vibration_ratio': temp_vibration_ratio,
                'failure': 1 if is_failure else 0
            })
        
        df = pd.DataFrame(data)
        
        # Add rolling statistics (predictive features)
        df = df.sort_values('timestamp')
        
        for col in ['temperature', 'vibration', 'current', 'pressure', 'speed']:
            df[f'{col}_rolling_mean'] = df.groupby('equipment_id')[col].rolling(window=10).mean().values
            df[f'{col}_rolling_std'] = df.groupby('equipment_id')[col].rolling(window=10).std().values
            df[f'{col}_trend'] = df.groupby('equipment_id')[col].diff().values
        
        # Fill NaN values from rolling calculations
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        
        # Encode categorical variables
        df_processed = df.copy()
        df_processed['equipment_encoded'] = self.label_encoder.fit_transform(df_processed['equipment_id'])
        
        # Time-based features
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        
        # Select features for training
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['timestamp', 'equipment_id', 'failure']]
        
        self.feature_names = feature_cols
        
        X = df_processed[feature_cols]
        y = df_processed['failure']
        
        return X, y
    
    def train_models(self, X, y):
        """Train Random Forest and XGBoost models"""
        
        print("\nTraining machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_prob = self.rf_model.predict_proba(X_test)[:, 1]
        
        # Train XGBoost
        print("Training XGBoost...")
        self.xgb_model.fit(X_train, y_train)
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        
        # Evaluate models
        print("\n=== Model Performance ===")
        
        print("\nRandom Forest:")
        print(classification_report(y_test, rf_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_prob):.4f}")
        
        print("\nXGBoost:")
        print(classification_report(y_test, xgb_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_prob):.4f}")
        
        self.is_trained = True
        self.X_test = X_test
        self.y_test = y_test
        
        return {
            'rf_score': roc_auc_score(y_test, rf_prob),
            'xgb_score': roc_auc_score(y_test, xgb_prob),
            'X_test': X_test,
            'y_test': y_test,
            'rf_prob': rf_prob,
            'xgb_prob': xgb_prob
        }
    
    def explain_predictions(self, X_sample=None, n_samples=100):
        """Generate SHAP explanations for model predictions"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained first!")
        
        print("\nGenerating SHAP explanations...")
        
        # Use test data sample if none provided
        if X_sample is None:
            X_sample = self.X_test.sample(n=min(n_samples, len(self.X_test)))
        
        # Create SHAP explainer for Random Forest
        explainer_rf = shap.TreeExplainer(self.rf_model)
        shap_values_rf = explainer_rf.shap_values(X_sample)
        
        # For binary classification, take positive class
        if isinstance(shap_values_rf, list):
            shap_values_rf = shap_values_rf[1]
        
        # Feature importance from SHAP
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values_rf).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features (SHAP):")
        print(feature_importance.head(10))
        
        return shap_values_rf, feature_importance
    
    def create_monitoring_dashboard_data(self, df, recent_hours=24):
        """Create data for real-time monitoring dashboard"""
        
        # Get recent data
        latest_time = df['timestamp'].max()
        cutoff_time = latest_time - timedelta(hours=recent_hours)
        recent_data = df[df['timestamp'] >= cutoff_time].copy()
        
        # Predict on recent data
        if self.is_trained and len(recent_data) > 0:
            X_recent, _ = self.prepare_features(recent_data)
            recent_data['failure_probability'] = self.rf_model.predict_proba(X_recent)[:, 1]
            recent_data['risk_level'] = pd.cut(recent_data['failure_probability'], 
                                             bins=[0, 0.3, 0.7, 1.0], 
                                             labels=['Low', 'Medium', 'High'])
        
        # Equipment health summary
        equipment_health = recent_data.groupby('equipment_id').agg({
            'failure_probability': 'max',
            'temperature': 'mean',
            'vibration': 'mean',
            'current': 'mean',
            'error_count': 'sum'
        }).round(3)
        
        # Time series data for charts
        time_series = recent_data.groupby(['timestamp', 'equipment_id']).agg({
            'failure_probability': 'first',
            'temperature': 'first',
            'vibration': 'first'
        }).reset_index()
        
        # Alerts (high risk equipment)
        alerts = recent_data[recent_data['failure_probability'] > 0.7][
            ['timestamp', 'equipment_id', 'failure_probability', 'temperature', 'vibration']
        ].sort_values('failure_probability', ascending=False)
        
        dashboard_data = {
            'equipment_health': equipment_health,
            'time_series': time_series,
            'alerts': alerts,
            'summary_stats': {
                'total_equipment': len(recent_data['equipment_id'].unique()),
                'high_risk_count': len(recent_data[recent_data['failure_probability'] > 0.7]),
                'avg_failure_prob': recent_data['failure_probability'].mean(),
                'last_updated': latest_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return dashboard_data
    
    def save_model(self, filename='equipment_failure_model.pkl'):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'rf_model