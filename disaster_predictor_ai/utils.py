import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Dict, List, Tuple, Any

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
    
    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = self._clean_dataset(df)
        return df
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df.reset_index(drop=True)
    
    def prepare_drought_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_cols = [
            'elevation', 'slope1', 'slope2', 'slope3', 'aspectN', 'aspectE',
            'WAT_LAND', 'NVG_LAND', 'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].values
        
        drought_threshold = df['SQ1'].quantile(0.7) if 'SQ1' in df.columns else 3
        y = (df['SQ1'] >= drought_threshold).astype(int).values if 'SQ1' in df.columns else np.random.randint(0, 2, len(df))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['drought'] = scaler
        self.feature_columns['drought'] = available_cols
        
        return X_scaled, y
    
    def prepare_flood_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df_flood = df[df['Entity'] == 'Flood'].copy() if 'Entity' in df.columns else df.copy()
        
        if len(df_flood) == 0:
            df_flood = df.copy()
        
        df_flood['Year_normalized'] = (df_flood['Year'] - df_flood['Year'].min()) / (df_flood['Year'].max() - df_flood['Year'].min())
        
        feature_cols = ['Year_normalized']
        if 'Disasters' in df_flood.columns:
            X = df_flood[feature_cols + ['Year']].values
            y = (df_flood['Disasters'] > df_flood['Disasters'].median()).astype(int).values
        else:
            X = df_flood[['Year']].values
            X = np.column_stack([X, (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())])
            y = np.random.randint(0, 2, len(df_flood))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['flood'] = scaler
        self.feature_columns['flood'] = ['Year', 'Year_normalized']
        
        return X_scaled, y
    
    def prepare_hunger_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df_clean = df.dropna(subset=['Global Hunger Index (2021)']) if 'Global Hunger Index (2021)' in df.columns else df
        
        if 'Year' in df_clean.columns:
            df_clean['Year_normalized'] = (df_clean['Year'] - df_clean['Year'].min()) / (df_clean['Year'].max() - df_clean['Year'].min())
            feature_cols = ['Year', 'Year_normalized']
        else:
            df_clean['dummy_feature'] = np.random.random(len(df_clean))
            feature_cols = ['dummy_feature']
        
        X = df_clean[feature_cols].values
        
        if 'Global Hunger Index (2021)' in df_clean.columns:
            ghi_values = df_clean['Global Hunger Index (2021)'].values
            y = np.where(ghi_values < 10, 0, np.where(ghi_values < 20, 1, 2))
        else:
            y = np.random.randint(0, 3, len(df_clean))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['hunger'] = scaler
        self.feature_columns['hunger'] = feature_cols
        
        return X_scaled, y
    
    def prepare_crop_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_cols = [
            'Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT',
            'Extreme_Weather_Events', 'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA',
            'Fertilizer_Use_KG_per_HA', 'Soil_Health_Index'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) == 0:
            available_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        
        X = df[available_cols].values
        y = df['Crop_Yield_MT_per_HA'].values if 'Crop_Yield_MT_per_HA' in df.columns else np.random.random(len(df)) * 5
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['crop'] = scaler
        self.feature_columns['crop'] = available_cols
        
        return X_scaled, y
    
    def save_preprocessors(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(os.path.join(model_dir, 'feature_columns.pkl'), 'wb') as f:
            pickle.dump(self.feature_columns, f)
    
    def load_preprocessors(self, model_dir: str):
        with open(os.path.join(model_dir, 'scalers.pkl'), 'rb') as f:
            self.scalers = pickle.load(f)
        
        with open(os.path.join(model_dir, 'feature_columns.pkl'), 'rb') as f:
            self.feature_columns = pickle.load(f)

class ModelUtils:
    @staticmethod
    def save_model(model: Any, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, model_type: str) -> Dict:
        if model_type == 'regression':
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return {'mse': mse, 'r2': r2, 'rmse': np.sqrt(mse)}
        else:
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(y_true, y_pred)
            return {'accuracy': accuracy}
    
    @staticmethod
    def create_prediction_input(location_data: Dict, hazard_type: str, 
                              feature_columns: List[str]) -> np.ndarray:
        feature_values = []
        
        if hazard_type == 'drought':
            defaults = {
                'elevation': location_data.get('lat', 0) * 100,
                'slope1': 0.1, 'slope2': 0.2, 'slope3': 0.3,
                'aspectN': 0.2, 'aspectE': 0.2,
                'WAT_LAND': 5.0, 'NVG_LAND': 20.0,
                'GRS_LAND': 30.0, 'FOR_LAND': 25.0, 'CULTRF_LAND': 20.0
            }
        elif hazard_type == 'flood':
            defaults = {
                'Year': 2025, 'Year_normalized': 0.8
            }
        elif hazard_type == 'hunger':
            defaults = {
                'Year': 2025, 'Year_normalized': 0.8, 'dummy_feature': 0.5
            }
        else:
            defaults = {
                'Average_Temperature_C': 25.0,
                'Total_Precipitation_mm': 800.0,
                'CO2_Emissions_MT': 15.0,
                'Extreme_Weather_Events': 3,
                'Irrigation_Access_%': 50.0,
                'Pesticide_Use_KG_per_HA': 20.0,
                'Fertilizer_Use_KG_per_HA': 40.0,
                'Soil_Health_Index': 70.0
            }
        
        for col in feature_columns:
            feature_values.append(defaults.get(col, 0.0))
        
        return np.array(feature_values).reshape(1, -1)

class ResponseFormatter:
    @staticmethod
    def format_drought_prediction(prediction: int, confidence: float, location: str) -> str:
        risk_level = "High" if prediction == 1 else "Low"
        return f"Drought risk in {location.title()}: {risk_level} (Confidence: {confidence:.1f}%)"
    
    @staticmethod
    def format_flood_prediction(prediction: int, confidence: float, location: str) -> str:
        risk_level = "High" if prediction == 1 else "Low"
        return f"Flood risk in {location.title()}: {risk_level} (Confidence: {confidence:.1f}%)"
    
    @staticmethod
    def format_hunger_prediction(prediction: int, confidence: float, location: str) -> str:
        risk_levels = {0: "Low", 1: "Moderate", 2: "High"}
        risk_level = risk_levels.get(prediction, "Unknown")
        return f"Hunger risk in {location.title()}: {risk_level} (Confidence: {confidence:.1f}%)"
    
    @staticmethod
    def format_crop_prediction(prediction: float, confidence: float, location: str) -> str:
        return f"Expected crop yield in {location.title()}: {prediction:.2f} MT/HA (Confidence: {confidence:.1f}%)"
    
    @staticmethod
    def format_fallback_message() -> str:
        return "Sorry, we couldn't understand your location. Please use the dropdown below to select your area:"
    
    @staticmethod
    def get_example_query() -> str:
        return "Try asking: 'Will there be drought in Turkana next year?'"

def validate_data_files(data_dir: str) -> Dict[str, bool]:
    required_files = [
        'drought_data.csv',
        'flood_data.csv', 
        'hunger_data.csv',
        'crop_data.csv'
    ]
    
    file_status = {}
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        file_status[file] = os.path.exists(file_path)
    
    return file_status

def get_model_paths(models_dir: str) -> Dict[str, str]:
    return {
        'drought': os.path.join(models_dir, 'drought_model.pkl'),
        'flood': os.path.join(models_dir, 'flood_model.pkl'),
        'hunger': os.path.join(models_dir, 'hunger_model.pkl'),
        'crop': os.path.join(models_dir, 'crop_yield_model.pkl')
    }
