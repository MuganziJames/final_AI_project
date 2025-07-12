import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
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
        print("\nTraining drought prediction model...")
        
        df = self.preprocessor.load_and_clean_data(
            os.path.join(self.data_dir, 'drought_data.csv')
        )
        
        X, y = self.preprocessor.prepare_drought_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        model_path = get_model_paths(self.models_dir)['drought']
        ModelUtils.save_model(model, model_path)
        
        print(f"Drought model accuracy: {accuracy:.3f}")
        print(f"Saved to: {model_path}")
    
    def train_flood_model(self):
        print("\nTraining flood prediction model...")
        
        df = self.preprocessor.load_and_clean_data(
            os.path.join(self.data_dir, 'flood_data.csv')
        )
        
        X, y = self.preprocessor.prepare_flood_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        model_path = get_model_paths(self.models_dir)['flood']
        ModelUtils.save_model(model, model_path)
        
        print(f"Flood model accuracy: {accuracy:.3f}")
        print(f"Saved to: {model_path}")
    
    def train_hunger_model(self):
        print("\nTraining hunger prediction model...")
        
        df = self.preprocessor.load_and_clean_data(
            os.path.join(self.data_dir, 'hunger_data.csv')
        )
        
        X, y = self.preprocessor.prepare_hunger_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        model_path = get_model_paths(self.models_dir)['hunger']
        ModelUtils.save_model(model, model_path)
        
        print(f"Hunger model accuracy: {accuracy:.3f}")
        print(f"Saved to: {model_path}")
    
    def train_crop_model(self):
        print("\nTraining crop yield prediction model...")
        
        df = self.preprocessor.load_and_clean_data(
            os.path.join(self.data_dir, 'crop_data.csv')
        )
        
        X, y = self.preprocessor.prepare_crop_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        model_path = get_model_paths(self.models_dir)['crop']
        ModelUtils.save_model(model, model_path)
        
        print(f"Crop yield model R²: {r2:.3f}")
        print(f"Crop yield model RMSE: {rmse:.3f}")
        print(f"Saved to: {model_path}")

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
