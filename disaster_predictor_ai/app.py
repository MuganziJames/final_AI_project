import streamlit as st
import os
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlp.parser import ClimateQueryParser
from predict import ClimatePredictor
from utils import ResponseFormatter

class ClimateRiskApp:
    def __init__(self):
        self.setup_page_config()
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        self.parser = ClimateQueryParser()
        self.predictor = ClimatePredictor(self.models_dir)
        
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="Climate Risk Predictor",
            page_icon="🌍",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    def render_header(self):
        st.title("🌍 AI-Powered Climate Risk Assessment")
        st.markdown("Ask questions about climate risks in natural language and get AI-powered predictions.")
        
        with st.expander("💡 How to use this tool"):
            st.markdown("""
            **Example:** *"Will there be drought in Turkana next year?"*
            
            **What you can ask about:**
            - 🌵 **Drought** risk in any African location
            - 🌊 **Flood** risk predictions  
            - 🍽️ **Hunger** and food security levels
            - 🌾 **Crop yield** expectations
            
            **Supported locations:** Cities, regions, and countries across Africa
            """)
    
    def render_main_interface(self):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Ask Your Question")
            
            user_query = st.text_input(
                "Type your climate risk question:",
                placeholder="e.g., Will there be drought in Kisumu next year?",
                key="user_query"
            )
            
            col_predict, col_clear = st.columns([1, 1])
            
            with col_predict:
                predict_button = st.button("🔮 Get Prediction", type="primary", use_container_width=True)
            
            with col_clear:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.user_query = ""
                    st.rerun()
            
            if predict_button and user_query:
                self.process_query(user_query)
        
        with col2:
            self.render_sidebar_content()
    
    def process_query(self, query: str):
        with st.spinner("Analyzing your question..."):
            parsed_result = self.parser.parse(query)
            
            st.session_state.query_history.append({
                "query": query,
                "parsed": parsed_result
            })
            
            if parsed_result["parsed_successfully"]:
                self.show_successful_prediction(parsed_result, query)
            else:
                self.show_fallback_interface(query, parsed_result)
    
    def show_successful_prediction(self, parsed_result: Dict, original_query: str):
        hazard = parsed_result["hazard"]
        location = parsed_result["location"]
        location_details = parsed_result["location_details"]
        time_period = parsed_result["time"]
        confidence = parsed_result["confidence"]
        
        st.success(f"✅ Query understood with {confidence}% confidence")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Type", hazard.title())
        with col2:
            st.metric("Location", location.title())
        with col3:
            st.metric("Time Period", time_period or "Near future")
        
        with st.spinner("Generating prediction..."):
            prediction = self.predictor.get_formatted_prediction(
                hazard, location, location_details, time_period
            )
            
            st.subheader("🎯 Prediction Result")
            
            if "High" in prediction:
                st.error(f"⚠️ {prediction}")
            elif "Moderate" in prediction:
                st.warning(f"⚡ {prediction}")
            else:
                st.success(f"✅ {prediction}")
            
            self.show_location_details(location_details)
    
    def show_fallback_interface(self, original_query: str, parsed_result: Dict):
        st.warning("🤔 " + ResponseFormatter.format_fallback_message())
        
        st.subheader("Manual Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hazard_type = st.selectbox(
                "Select Risk Type:",
                options=["drought", "flood", "hunger", "crop"],
                format_func=lambda x: x.title()
            )
        
        with col2:
            countries = self.parser.get_available_countries()
            selected_country = st.selectbox(
                "Select Country:",
                options=sorted(countries)
            )
        
        with col3:
            time_options = ["Next 6 months", "Next year", "2025", "2026"]
            selected_time = st.selectbox("Select Time Period:", options=time_options)
        
        location_suggestions = [
            loc for loc, details in self.parser.locations.items() 
            if details.get("country") == selected_country
        ]
        
        if location_suggestions:
            selected_location = st.selectbox(
                f"Select Location in {selected_country}:",
                options=sorted(location_suggestions),
                format_func=lambda x: x.title()
            )
            
            if st.button("🔮 Get Prediction", key="fallback_predict"):
                location_details = self.parser.locations[selected_location]
                
                with st.spinner("Generating prediction..."):
                    prediction = self.predictor.get_formatted_prediction(
                        hazard_type, selected_location, location_details, selected_time
                    )
                    
                    st.subheader("🎯 Prediction Result")
                    
                    if "High" in prediction:
                        st.error(f"⚠️ {prediction}")
                    elif "Moderate" in prediction:
                        st.warning(f"⚡ {prediction}")
                    else:
                        st.success(f"✅ {prediction}")
                    
                    self.show_location_details(location_details)
    
    def show_location_details(self, location_details: Dict):
        if location_details:
            with st.expander("📍 Location Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Country:** {location_details.get('country', 'N/A')}")
                    st.write(f"**Type:** {location_details.get('type', 'N/A').title()}")
                
                with col2:
                    if 'lat' in location_details and 'lon' in location_details:
                        st.write(f"**Latitude:** {location_details['lat']:.4f}")
                        st.write(f"**Longitude:** {location_details['lon']:.4f}")
                    
                    region = location_details.get('region') or location_details.get('province') or location_details.get('state')
                    if region:
                        st.write(f"**Region:** {region}")
    
    def render_sidebar_content(self):
        st.subheader("🎯 Quick Actions")
        
        if st.button("🏠 Kenya Regions", use_container_width=True):
            self.show_quick_prediction("drought", "turkana", "Kenya")
        
        if st.button("🌍 South Africa", use_container_width=True):
            self.show_quick_prediction("flood", "western cape", "South Africa")
        
        if st.button("🍽️ Nigeria States", use_container_width=True):
            self.show_quick_prediction("hunger", "borno", "Nigeria")
        
        if st.button("🌾 Ethiopia Regions", use_container_width=True):
            self.show_quick_prediction("crop", "oromia", "Ethiopia")
        
        st.subheader("📊 Model Status")
        model_status = self.predictor.validate_models()
        
        for model_type, loaded in model_status.items():
            status_icon = "✅" if loaded else "❌"
            st.write(f"{status_icon} {model_type.title()} Model")
        
        if st.session_state.query_history:
            st.subheader("📝 Recent Queries")
            for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                    st.write(f"**Question:** {entry['query']}")
                    st.write(f"**Confidence:** {entry['parsed']['confidence']}%")
    
    def show_quick_prediction(self, hazard_type: str, location_name: str, country: str):
        if location_name in self.parser.locations:
            location_details = self.parser.locations[location_name]
            prediction = self.predictor.get_formatted_prediction(
                hazard_type, location_name, location_details
            )
            st.info(f"Quick prediction: {prediction}")
        else:
            st.error(f"Location {location_name} not found in database")
    
    def run(self):
        try:
            self.render_header()
            self.render_main_interface()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.info("Please ensure all models are trained by running: python train.py")

def main():
    app = ClimateRiskApp()
    app.run()

if __name__ == "__main__":
    main()
