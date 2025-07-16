import streamlit as st
import os
import sys
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

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
            page_title="ClimateWise",
            page_icon="🌍",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    def render_header(self):
        st.title("🌍 ClimateWise - AI-Powered Climate Risk Assessment")
        st.markdown("Choose your preferred method: Ask questions in natural language or use manual selection.")
        
        with st.expander("💡 How to use this tool"):
            st.markdown("""
            **Method 1 - Natural Language:** *"Will there be drought in Turkana next year?"*
            
            **Method 2 - Manual Selection:** Use the dropdown menus below
            
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
            st.subheader("🤖 Natural Language Query")
            
            user_query = st.text_input(
                "Type your climate risk question:",
                placeholder="e.g., Will there be drought in Kisumu next year?",
                key="user_query"
            )
            
            predict_button = st.button("🔮 Get NLP Prediction", type="primary", use_container_width=True)
            
            if predict_button and user_query:
                self.process_query(user_query)
            
            st.divider()
            
            self.render_manual_interface()
        
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
                confidence = parsed_result["confidence"]
                st.warning(f"⚠️ Could not fully understand your query (confidence: {confidence}%). Please try the manual selection below or rephrase your question.")
    
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
            self.render_risk_dashboard(location, location_details)
    
    def render_manual_interface(self):
        st.subheader("📋 Manual Selection")
        st.markdown("*Prefer dropdowns? Select your options below:*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hazard_type = st.selectbox(
                "Select Risk Type:",
                options=["drought", "flood", "hunger", "crop"],
                format_func=lambda x: x.title(),
                key="manual_hazard"
            )
        
        with col2:
            countries = self.parser.get_available_countries()
            selected_country = st.selectbox(
                "Select Country:",
                options=sorted(countries),
                key="manual_country"
            )
        
        with col3:
            time_options = ["Next 6 months", "Next year", "2025", "2026"]
            selected_time = st.selectbox(
                "Select Time Period:", 
                options=time_options,
                key="manual_time"
            )
        
        location_suggestions = [
            loc for loc, details in self.parser.locations.items() 
            if details.get("country") == selected_country
        ]
        
        if location_suggestions:
            selected_location = st.selectbox(
                f"Select Location in {selected_country}:",
                options=sorted(location_suggestions),
                format_func=lambda x: x.title(),
                key="manual_location"
            )
            
            if st.button("🎯 Get Manual Prediction", use_container_width=True, key="manual_predict"):
                location_details = self.parser.locations[selected_location]
                
                with st.spinner("Generating prediction..."):
                    prediction = self.predictor.get_formatted_prediction(
                        hazard_type, selected_location, location_details, selected_time
                    )
                    
                    st.subheader("🎯 Manual Prediction Result")
                    
                    if "High" in prediction:
                        st.error(f"⚠️ {prediction}")
                    elif "Moderate" in prediction:
                        st.warning(f"⚡ {prediction}")
                    else:
                        st.success(f"✅ {prediction}")
                    
                    self.show_location_details(location_details)
                    self.render_risk_dashboard(selected_location, location_details)
        else:
            st.info(f"No locations found for {selected_country}. Please select a different country.")
    
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
    
    def render_risk_dashboard(self, location_name: str, location_details: Dict):
        """Render a comprehensive risk dashboard for a specific location"""
        st.subheader(f"🌍 Comprehensive Risk Assessment: {location_name.title()}")
        
        # Add confidence explanation upfront
        with st.expander("❓ Understanding Confidence Levels"):
            st.markdown("""
            **How to interpret confidence levels:**
            - 🟢 **High (85%+)**: Very reliable prediction - strong confidence in the result
            - 🟡 **Medium (70-84%)**: Moderately reliable prediction - good confidence but consider context
            - 🔴 **Low (<70%)**: Less reliable prediction - use with caution and seek additional information
            - 🟤 **Poor (<50%)**: Very low reliability - consider alternative data sources
            
            *Confidence reflects how certain the AI model is about its prediction based on available data.*
            """)
        
        # Get predictions for all hazard types
        all_predictions = {}
        for hazard_type in ['drought', 'flood', 'hunger', 'crop']:
            result = self.predictor.predict(hazard_type, location_details)
            if not result.get('error'):
                all_predictions[hazard_type] = result
        
        if all_predictions:
            # Create metrics row with improved styling
            st.markdown("### 📊 Risk Overview")
            cols = st.columns(len(all_predictions))
            
            for i, (hazard_type, result) in enumerate(all_predictions.items()):
                with cols[i]:
                    pred = result['prediction']
                    conf = result['confidence']
                    
                    # Get confidence icon for visual indication
                    if conf >= 85:
                        conf_icon = "🟢"
                    elif conf >= 70:
                        conf_icon = "🟡"  
                    elif conf >= 50:
                        conf_icon = "🔴"
                    else:
                        conf_icon = "🟤"
                    
                    if hazard_type == 'crop':
                        value = f"{pred:.1f} MT/HA"
                        delta = f"{'Good' if pred >= 2.5 else 'Poor'} {conf_icon}"
                    elif hazard_type == 'hunger':
                        levels = {0: "Low", 1: "Moderate", 2: "High"}
                        value = levels.get(pred, "Unknown")
                        delta = f"{conf:.0f}% {conf_icon}"
                    else:
                        value = "High Risk" if pred == 1 else "Low Risk"
                        delta = f"{conf:.0f}% {conf_icon}"
                    
                    st.metric(
                        label=f"{hazard_type.title()} Risk",
                        value=value,
                        delta=delta
                    )
            
            # Show detailed breakdown with improved styling
            st.markdown("### 📋 Detailed Analysis")
            with st.expander("View Complete Risk Breakdown", expanded=True):
                summary_table = ResponseFormatter.create_risk_summary_table(location_name, all_predictions)
                st.markdown(summary_table)
        else:
            st.warning("Unable to generate comprehensive risk assessment. Some models may not be available.")

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
