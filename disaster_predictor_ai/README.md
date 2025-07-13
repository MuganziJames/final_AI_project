# AI-Powered Climate Risk & Crop Yield Prediction Tool

A comprehensive Python-based AI system that helps users assess climate-related risks through natural language queries, predicting droughts, floods, hunger risk, and crop yields for regions across Africa.

## Features

- **Natural Language Processing**: Ask questions in plain English
- **Comprehensive African Coverage**: 200+ cities, regions, and countries
- **Multiple Risk Types**: Drought, flood, hunger, and crop yield predictions
- **Smart Fallback**: Dropdown interface when NLP parsing fails
- **Interactive Web Interface**: Clean Streamlit-based UI
- **Offline Operation**: No external APIs required

## Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**

   ```bash
   python train.py
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

## File Structure

```
disaster_predictor_ai/
├── data/                    # CSV datasets
│   ├── drought_data.csv
│   ├── flood_data.csv
│   ├── hunger_data.csv
│   └── crop_data.csv
├── models/                  # Trained ML models
│   ├── drought_model.pkl
│   ├── flood_model.pkl
│   ├── hunger_model.pkl
│   └── crop_yield_model.pkl
├── nlp/                     # Natural language processing
│   └── parser.py
├── train.py                 # Model training script
├── predict.py               # Prediction engine
├── app.py                   # Streamlit web interface
├── utils.py                 # Helper functions
└── requirements.txt         # Python dependencies
```

## Usage Examples

### Natural Language Queries

- "Will there be drought in Turkana next year?"
- "Flood risk in Lagos during rainy season"
- "Food security in Borno state"
- "Maize yield in Rift Valley this season"

### Supported Locations

- **Countries**: Kenya, South Africa, Nigeria, Ethiopia, Ghana, Tanzania, Uganda, and more
- **Regions**: East Africa, Sahel, Horn of Africa, Southern Africa
- **Cities**: Nairobi, Lagos, Cape Town, Addis Ababa, Accra, Kampala
- **Counties/States**: Turkana, Borno, Western Cape, Oromia

## Technical Details

### Machine Learning Models

- **Drought Prediction**: Random Forest Classifier
- **Flood Prediction**: Random Forest Classifier
- **Hunger Assessment**: Random Forest Classifier (3-class)
- **Crop Yield**: Random Forest Regressor

### NLP Capabilities

- Extensive synonym recognition
- Fuzzy location matching
- Multi-word location support
- Confidence scoring system
- Graceful fallback handling

## Data Sources

The system uses four main datasets covering climate, agricultural, and socioeconomic indicators across Africa.

## Contributing

This project focuses on African climate resilience and food security. Contributions for additional regions or improved prediction accuracy are welcome.

## License

Open source project for climate research and humanitarian applications.
