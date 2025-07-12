import os
import numpy as np
from typing import Dict, Tuple, Optional

from utils import DataPreprocessor, ModelUtils, get_model_paths, ResponseFormatter

class ClimatePredictor:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = DataPreprocessor()
        self._load_models()
    
    def _load_models(self):
        try:
            self.preprocessor.load_preprocessors(self.models_dir)
            
            model_paths = get_model_paths(self.models_dir)
            
            for hazard_type, model_path in model_paths.items():
                if os.path.exists(model_path):
                    self.models[hazard_type] = ModelUtils.load_model(model_path)
                    print(f"Loaded {hazard_type} model")
                else:
                    print(f"Warning: {hazard_type} model not found at {model_path}")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Please run train.py first to train the models.")
    
    def predict(self, hazard_type: str, location_data: Dict, time_period: str = None) -> Dict:
        if hazard_type not in self.models:
            return {
                "error": f"Model for {hazard_type} not available",
                "prediction": None,
                "confidence": 0
            }
        
        try:
            model = self.models[hazard_type]
            scaler = self.preprocessor.scalers[hazard_type]
            feature_columns = self.preprocessor.feature_columns[hazard_type]
            
            input_features = ModelUtils.create_prediction_input(
                location_data, hazard_type, feature_columns
            )
            
            input_scaled = scaler.transform(input_features)
            
            if hazard_type == 'crop':
                prediction = model.predict(input_scaled)[0]
                confidence = self._calculate_regression_confidence(model, input_scaled)
            else:
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                confidence = max(prediction_proba) * 100
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "hazard_type": hazard_type,
                "location": location_data,
                "error": None
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "prediction": None,
                "confidence": 0
            }
    
    def _calculate_regression_confidence(self, model, input_features: np.ndarray) -> float:
        try:
            predictions = []
            for estimator in model.estimators_:
                pred = estimator.predict(input_features)[0]
                predictions.append(pred)
            
            std_dev = np.std(predictions)
            mean_pred = np.mean(predictions)
            
            if mean_pred != 0:
                cv = std_dev / abs(mean_pred)
                confidence = max(0, 100 - (cv * 100))
            else:
                confidence = 75.0
            
            return min(confidence, 95.0)
            
        except:
            return 75.0
    
    def get_formatted_prediction(self, hazard_type: str, location_name: str, 
                               location_data: Dict, time_period: str = None) -> str:
        result = self.predict(hazard_type, location_data, time_period)
        
        if result["error"]:
            return f"Error: {result['error']}"
        
        prediction = result["prediction"]
        confidence = result["confidence"]
        
        if hazard_type == "drought":
            return ResponseFormatter.format_drought_prediction(prediction, confidence, location_name)
        elif hazard_type == "flood":
            return ResponseFormatter.format_flood_prediction(prediction, confidence, location_name)
        elif hazard_type == "hunger":
            return ResponseFormatter.format_hunger_prediction(prediction, confidence, location_name)
        elif hazard_type == "crop":
            return ResponseFormatter.format_crop_prediction(prediction, confidence, location_name)
        else:
            return f"Unknown hazard type: {hazard_type}"
    
    def bulk_predict(self, requests: list) -> list:
        results = []
        for request in requests:
            hazard_type = request.get("hazard_type")
            location_data = request.get("location_data")
            location_name = request.get("location_name")
            time_period = request.get("time_period")
            
            prediction_text = self.get_formatted_prediction(
                hazard_type, location_name, location_data, time_period
            )
            
            results.append({
                "request": request,
                "prediction": prediction_text
            })
        
        return results
    
    def get_risk_summary(self, location_name: str, location_data: Dict) -> Dict:
        summary = {
            "location": location_name,
            "risks": {}
        }
        
        for hazard_type in self.models.keys():
            result = self.predict(hazard_type, location_data)
            if not result["error"]:
                summary["risks"][hazard_type] = {
                    "prediction": result["prediction"],
                    "confidence": result["confidence"]
                }
        
        return summary
    
    def validate_models(self) -> Dict[str, bool]:
        model_status = {}
        expected_models = ['drought', 'flood', 'hunger', 'crop']
        
        for model_type in expected_models:
            model_status[model_type] = model_type in self.models
        
        return model_status

def make_prediction(hazard_type: str, location_name: str, location_data: Dict, 
                   models_dir: str, time_period: str = None) -> str:
    predictor = ClimatePredictor(models_dir)
    return predictor.get_formatted_prediction(
        hazard_type, location_name, location_data, time_period
    )

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    
    predictor = ClimatePredictor(models_dir)
    
    model_status = predictor.validate_models()
    print("Model Status:")
    for model_type, loaded in model_status.items():
        status = "✓ Loaded" if loaded else "✗ Missing"
        print(f"  {model_type}: {status}")
    
    if not any(model_status.values()):
        print("\nNo models loaded. Please run train.py first.")
        return
    
    print("\nTesting predictions...")
    
    test_location = {
        "country": "Kenya",
        "type": "city",
        "lat": -1.2921,
        "lon": 36.8219
    }
    
    for hazard_type in ['drought', 'flood', 'hunger', 'crop']:
        if model_status[hazard_type]:
            prediction = predictor.get_formatted_prediction(
                hazard_type, "Nairobi", test_location
            )
            print(f"{hazard_type.title()}: {prediction}")

if __name__ == "__main__":
    main()
