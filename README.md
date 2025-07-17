# 🌍 ClimateWise - AI-Powered Climate Risk Assessment System

## 🚀 **[🌐 Try ClimateWise Live →](https://finalaiproject-fexrvamcmq8p76hma3e5fy.streamlit.app/)**

_Experience ClimateWise's AI-powered climate risk predictions in real-time! Ask questions like "Will there be drought in Kenya next year?" and get instant, actionable insights._

**🌍 Building a Climate-Resilient Africa, One Prediction at a Time**

ClimateWise represents a significant step toward proactive climate risk management, enabling communities across Africa to prepare for and mitigate the impacts of climate change through advanced machine learning and natural language processing technologies.

---

## 🎯 About the Project

### Real-World Impact & UN SDGs

ClimateWise directly contributes to multiple **UN Sustainable Development Goals**:

- **SDG 1 - No Poverty**: Predicts crop failures and food insecurity for proactive interventions
- **SDG 2 - Zero Hunger**: Prevents famine through advanced hunger and crop yield predictions
- **SDG 13 - Climate Action**: Provides actionable climate adaptation intelligence
- **SDG 15 - Life on Land**: Supports sustainable land management through risk assessment

### Key Benefits

- **🚨 Early Warning Systems**: 6-12 month advance warnings for climate disasters
- **📊 Resource Allocation**: Enables pre-emptive deployment of aid and resources
- **🌾 Agricultural Planning**: Informs crop selection and planting decisions
- **💰 Economic Protection**: Reduces losses through proactive climate action

## 🤖 Machine Learning Architecture

### Model Specifications

- **Drought Prediction**: Random Forest Classifier (88.1% accuracy)
- **Flood Risk Assessment**: Random Forest Classifier (90.0% accuracy)
- **Hunger Forecasting**: Multi-class Random Forest (29.1% accuracy)
- **Crop Yield Prediction**: Random Forest Regressor (99.8% R²)

### Why Random Forest?

- **Robust Performance**: Handles complex, non-linear climate relationships
- **Feature Importance**: Identifies key climate variables driving predictions
- **Overfitting Resistance**: Ensemble approach prevents model overfitting
- **Missing Data Handling**: Naturally handles incomplete climate records

## 🗣️ Natural Language Processing

### NLP Capabilities

- **Location Extraction**: Identifies 200+ African cities, regions, and countries
- **Hazard Classification**: Recognizes drought, flood, hunger, and crop-related queries
- **Temporal Understanding**: Processes time references ("next year", "2025")
- **Confidence Scoring**: Provides reliability metrics for query interpretation

### Example Queries

- "Will there be drought in Kenya next year?"
- "What's the flood risk in Lagos, Nigeria?"
- "How will crops perform in Ethiopia this season?"
- "Is there hunger risk in Sudan?"

## 🌍 Data Sources

### Comprehensive Dataset Integration

- **10,000+ records** from Climate Change Impact on Agriculture (2024)
- **4,364 World Bank climate change records**
- **3,109 geographical drought records** with topological features
- **3,000 crop yield observations** with agricultural variables

### Feature Engineering

- **Geographical Features**: Latitude, longitude, elevation, land use patterns
- **Climate Variables**: Temperature, precipitation, CO2 emissions, extreme weather
- **Agricultural Indicators**: Irrigation access, soil health, fertilizer use
- **Economic Factors**: Impact assessments, adaptation strategies

## 🎯 Confidence System

Our system provides reliability indicators for each prediction:

- **🟢 High Confidence (85%+)**: Very reliable with comprehensive data coverage
- **🟡 Medium Confidence (70-84%)**: Moderately reliable with good data quality
- **🔴 Low Confidence (50-69%)**: Use with caution, limited data availability
- **🟤 Poor Confidence (<50%)**: Very low reliability, additional data required

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM recommended

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MuganziJames/final_AI_project.git
   cd final_AI_project
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**:

   ```bash
   python train.py
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### Usage

1. **Web Interface**: Navigate to `http://localhost:8501`
2. **Natural Language Queries**: Ask questions in plain English
3. **Manual Selection**: Use dropdowns to select locations and hazards
4. **Risk Dashboard**: View comprehensive risk assessments

## 🏗️ Project Structure

```
final_AI_project/
├── app.py              # Main Streamlit application
├── train.py            # Model training pipeline
├── predict.py          # Climate prediction engine
├── utils.py            # Data preprocessing utilities
├── requirements.txt    # Python dependencies
├── data/              # Climate and agricultural datasets
├── models/            # Trained machine learning models
├── nlp/               # Natural language processing module
└── README.md          # Project documentation
```

## 🎯 Key Features

### Prediction Capabilities

- **Multi-hazard Assessment**: Drought, flood, hunger, and crop yield predictions
- **Geographic Coverage**: 200+ African locations
- **Temporal Flexibility**: 6-month to 2-year forecasting horizons
- **Confidence Scoring**: Reliability indicators for each prediction

### User Experience

- **Intuitive Interface**: Accessible to both technical and non-technical users
- **Real-time Processing**: Instant prediction generation
- **Visual Dashboards**: Comprehensive risk visualization
- **Dual Input Methods**: Natural language AND manual selection

## 🛠️ Technical Requirements

### Dependencies

```
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.1.0    # Machine learning algorithms
streamlit>=1.20.0      # Web application framework
matplotlib>=3.5.0      # Data visualization
seaborn>=0.11.0        # Statistical visualization
plotly>=5.10.0         # Interactive plotting
```

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum
- **Storage**: 1GB available space
- **Internet**: Required for initial setup

## 🌍 Geographic Coverage

### Supported Countries & Regions

- **Kenya**: 19 locations (counties and major cities)
- **South Africa**: 17 locations (provinces and metropolitan areas)
- **Nigeria**: 15 locations (states and major cities)
- **Ethiopia**: 12 locations (regions and cities)
- **Uganda**: 10 locations (districts and cities)
- **Tanzania**: 8 locations (regions and cities)
- **Ghana**: 6 locations (regions and cities)
- **And more across Africa...**

## 🤝 Contributing

We welcome contributions from the climate science and machine learning communities!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add some amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Priority Areas

- **Additional Data Sources**: Integration of new climate datasets
- **Model Improvements**: Enhanced algorithms and feature engineering
- **Regional Expansion**: Support for additional African countries
- **Mobile Optimization**: Responsive design improvements
- **API Development**: RESTful API for external integrations

## 📈 Future Enhancements

- **🛰️ Satellite Data Integration**: Real-time satellite imagery analysis
- **📱 Mobile Application**: Native mobile app development
- **🔔 Alert Systems**: SMS/email notifications for high-risk predictions
- **🌐 Multi-language Support**: Local language interfaces
- **🤖 Advanced AI**: Deep learning model implementations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UN Sustainable Development Goals** for inspiring our mission
- **World Bank Climate Change Knowledge Portal** for comprehensive datasets
- **Global Hunger Index** for food security data
- **Climate research communities** for foundational knowledge
- **Open source community** for essential tools and libraries

## 📞 Contact

**James Muganzi**  
_Climate AI Developer_  
GitHub: [@MuganziJames](https://github.com/MuganziJames)  
Email: [muganzijames.ai.dev@gmail.com](mailto:muganzijames.ai.dev@gmail.com)

---

**🌍 Join us in building a climate-resilient future for Africa!**

_ClimateWise - Where AI meets Climate Action_
