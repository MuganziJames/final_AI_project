import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

from utils import DataPreprocessor, ModelUtils, validate_data_files, get_model_paths

class ModelTrainer:
    def __init__(self, data_dir: str, models_dir: str):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.preprocessor = DataPreprocessor()
        
        os.makedirs(models_dir, exist_ok=True)
    
    def train_all_models(self):
        print("Starting model training process...")
        
        file_status = validate_data_files(self.data_dir)
        if not all(file_status.values()):
            print("Missing data files:")
            for file, exists in file_status.items():
                if not exists:
                    print(f"  - {file}")
            return False
        
        try:
            self.train_drought_model()
            self.train_flood_model()
            self.train_hunger_model()
            self.train_crop_model()
            
            self.preprocessor.save_preprocessors(self.models_dir)
            print("\nAll models trained and saved successfully!")
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
    
    def train_drought_model(self):
        print("\nTraining enhanced drought prediction model...")
        
        df = self.preprocessor.load_and_clean_data(
            os.path.join(self.data_dir, 'drought_data.csv')
        )
        
        X, y = self.preprocessor.prepare_enhanced_drought_data(df, self.data_dir)
        
        # Check class distribution and handle edge cases
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2 or min(class_counts) < 2:
            print("Warning: Insufficient class distribution for stratified split. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        # Cross-validation evaluation
        cv_results = self.evaluate_model_with_cv(model, X, y)
        
        model_path = get_model_paths(self.models_dir)['drought']
        ModelUtils.save_model(model, model_path)
        
        print(f"Drought model accuracy: {accuracy:.3f}")
        print(f"Cross-validated accuracy: {cv_results['cv_accuracy_mean']:.3f} ± {cv_results['cv_accuracy_std']:.3f}")
        print(f"Saved to: {model_path}")
    
    def train_flood_model(self):
        print("\nTraining enhanced flood prediction model...")
        
        df = self.preprocessor.load_and_clean_data(
            os.path.join(self.data_dir, 'flood_data.csv')
        )
        
        X, y = self.preprocessor.prepare_enhanced_flood_data(df, self.data_dir)
        
        # Check class distribution and handle edge cases
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2 or min(class_counts) < 2:
            print("Warning: Insufficient class distribution for stratified split. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        # Cross-validation evaluation
        cv_results = self.evaluate_model_with_cv(model, X, y)
        
        model_path = get_model_paths(self.models_dir)['flood']
        ModelUtils.save_model(model, model_path)
        
        print(f"Flood model accuracy: {accuracy:.3f}")
        print(f"Cross-validated accuracy: {cv_results['cv_accuracy_mean']:.3f} ± {cv_results['cv_accuracy_std']:.3f}")
        print(f"Saved to: {model_path}")
    
    def train_hunger_model(self):
        print("\nTraining hunger prediction model...")
        
        df = self.preprocessor.load_and_clean_data(
            os.path.join(self.data_dir, 'hunger_data.csv')
        )
        
        X, y = self.preprocessor.prepare_enhanced_hunger_data(df, self.data_dir)
        
        # Check class distribution and handle edge cases
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2 or min(class_counts) < 2:
            print("Warning: Insufficient class distribution for stratified split. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        model = RandomForestClassifier(
            n_estimators=130,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        # Cross-validation evaluation
        cv_results = self.evaluate_model_with_cv(model, X, y)
        
        model_path = get_model_paths(self.models_dir)['hunger']
        ModelUtils.save_model(model, model_path)
        
        print(f"Hunger model accuracy: {accuracy:.3f}")
        print(f"Cross-validated accuracy: {cv_results['cv_accuracy_mean']:.3f} ± {cv_results['cv_accuracy_std']:.3f}")
        print(f"Saved to: {model_path}")
    
    def train_crop_model(self):
        print("\nTraining enhanced crop yield prediction model...")
        
        df = self.preprocessor.load_and_clean_data(
            os.path.join(self.data_dir, 'crop_data.csv')
        )
        
        X, y = self.preprocessor.prepare_enhanced_crop_data(df, self.data_dir)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(
            n_estimators=140,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation evaluation
        cv_results = self.evaluate_model_with_cv(model, X, y, model_type='regression')
        
        model_path = get_model_paths(self.models_dir)['crop']
        ModelUtils.save_model(model, model_path)
        
        print(f"Crop yield model R²: {r2:.3f}")
        print(f"Crop yield model RMSE: {rmse:.3f}")
        print(f"Cross-validated R²: {cv_results['cv_r2_mean']:.3f} ± {cv_results['cv_r2_std']:.3f}")
        print(f"Saved to: {model_path}")

    def evaluate_model_with_cv(self, model, X, y, model_type='classification', cv_folds=5):
        """Evaluate model performance using cross-validation"""
        
        if model_type == 'classification':
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            metric_name = 'accuracy'
        else:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            metric_name = 'r2'
        
        return {
            f'cv_{metric_name}_mean': scores.mean(),
            f'cv_{metric_name}_std': scores.std(),
            f'cv_{metric_name}_scores': scores
        }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    models_dir = os.path.join(script_dir, 'models')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please ensure the data folder contains the required CSV files.")
        return
    
    trainer = ModelTrainer(data_dir, models_dir)
    success = trainer.train_all_models()
    
    if success:
        print("\nModel training completed successfully!")
        print("You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("\nModel training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
