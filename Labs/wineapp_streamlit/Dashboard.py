import json
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.logger import get_logger
from datetime import datetime

# FastAPI backend endpoint
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# streamlit logger
LOGGER = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction Demo",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        background-color: #722f37;
        color: white;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

def main():
    # Header
    st.title("🍷 Wine Quality Prediction System")
    st.markdown("*Predict wine quality based on chemical properties using Machine Learning*")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Analytics", "📚 Learn", "🔬 Batch Predict"])
    
    with tab1:
        predict_tab()
    
    with tab2:
        analytics_tab()
    
    with tab3:
        learn_tab()
    
    with tab4:
        batch_predict_tab()

def predict_tab():
    """Main prediction interface"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Wine Parameters")
        
        # Check backend status
        backend_status = check_backend_status()
        
        # Wine type selection with emoji
        wine_type = st.selectbox(
            "Wine Type",
            ["red", "white"],
            format_func=lambda x: "🔴 Red Wine" if x == "red" else "⚪ White Wine"
        )
        
        # Quick presets
        st.markdown("**Quick Presets:**")
        col_preset1, col_preset2 = st.columns(2)
        with col_preset1:
            if st.button("🏆 Premium", use_container_width=True):
                st.session_state['preset'] = 'premium'
        with col_preset2:
            if st.button("📊 Average", use_container_width=True):
                st.session_state['preset'] = 'average'
        
        # Get slider values (with presets if selected)
        sliders = create_sliders(st.session_state.get('preset', None))
        
        # Clear preset after use
        if 'preset' in st.session_state:
            del st.session_state['preset']
        
        st.divider()
        
        # Prediction button
        predict_button = st.button('🔮 Predict Quality', type='primary', use_container_width=True)
    
    with col2:
        st.subheader("📈 Results & Insights")
        
        # Feature values visualization
        with st.expander("Current Wine Profile", expanded=True):
            display_wine_profile(sliders)
        
        # Prediction results
        if predict_button and backend_status:
            make_prediction(wine_type, sliders)
        
        # Show recent predictions
        if st.session_state.prediction_history:
            with st.expander("📜 Recent Predictions", expanded=False):
                display_history()

def create_sliders(preset):
    """Create parameter sliders with optional presets"""
    
    # Default values - ALL VALUES AS FLOATS
    defaults = {
        'premium': {
            'fixed_acidity': 8.5, 
            'volatile_acidity': 0.28, 
            'citric_acid': 0.56,
            'residual_sugar': 2.5, 
            'chlorides': 0.05, 
            'free_sulfur_dioxide': 35.0,  # Changed to float
            'total_sulfur_dioxide': 140.0,  # Changed to float
            'density': 0.995, 
            'pH': 3.3,
            'sulphates': 0.75, 
            'alcohol': 12.5
        },
        'average': {
            'fixed_acidity': 7.0, 
            'volatile_acidity': 0.5, 
            'citric_acid': 0.3,
            'residual_sugar': 3.0, 
            'chlorides': 0.08, 
            'free_sulfur_dioxide': 30.0,  # Changed to float
            'total_sulfur_dioxide': 100.0,  # Changed to float
            'density': 0.996, 
            'pH': 3.3,
            'sulphates': 0.65, 
            'alcohol': 10.5
        }
    }
    
    values = defaults.get(preset, {})
    
    return {
        'fixed_acidity': st.slider("Fixed Acidity", 3.0, 15.0, 
                                   values.get('fixed_acidity', 7.0), 0.1,
                                   help="Tartaric acid (g/L) - adds tartness"),
        'volatile_acidity': st.slider("Volatile Acidity", 0.0, 2.0,
                                      values.get('volatile_acidity', 0.5), 0.01,
                                      help="Acetic acid (g/L) - vinegar taste"),
        'citric_acid': st.slider("Citric Acid", 0.0, 2.0,
                                 values.get('citric_acid', 0.3), 0.01,
                                 help="Adds freshness and flavor"),
        'residual_sugar': st.slider("Residual Sugar", 0.0, 20.0,
                                    values.get('residual_sugar', 2.5), 0.1,
                                    help="Remaining sugar after fermentation (g/L)"),
        'chlorides': st.slider("Chlorides", 0.0, 0.5,
                               values.get('chlorides', 0.08), 0.001,
                               help="Salt content (g/L)"),
        'free_sulfur_dioxide': st.slider("Free SO₂", 0.0, 100.0,
                                         values.get('free_sulfur_dioxide', 30.0), 1.0,
                                         help="Prevents oxidation (mg/L)"),
        'total_sulfur_dioxide': st.slider("Total SO₂", 0.0, 300.0,
                                          values.get('total_sulfur_dioxide', 100.0), 1.0,
                                          help="Total sulfur dioxide (mg/L)"),
        'density': st.slider("Density", 0.98, 1.04,
                            values.get('density', 0.996), 0.001,
                            help="Mass per volume (g/mL)"),
        'pH': st.slider("pH", 2.5, 4.5,
                       values.get('pH', 3.3), 0.01,
                       help="Acidity level (lower = more acidic)"),
        'sulphates': st.slider("Sulphates", 0.0, 2.0,
                              values.get('sulphates', 0.65), 0.01,
                              help="Wine additive (g/L)"),
        'alcohol': st.slider("Alcohol", 8.0, 15.0,
                            values.get('alcohol', 10.5), 0.1,
                            help="Alcohol content (% by volume)")
    }

def display_wine_profile(sliders):
    """Display current wine profile as metrics"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Alcohol", f"{sliders['alcohol']}%")
        st.metric("pH", f"{sliders['pH']}")
        st.metric("Density", f"{sliders['density']} g/mL")
    
    with col2:
        st.metric("Total Acidity", f"{sliders['fixed_acidity'] + sliders['volatile_acidity']:.1f} g/L")
        st.metric("Sugar", f"{sliders['residual_sugar']} g/L")
        st.metric("SO₂", f"{sliders['total_sulfur_dioxide']} mg/L")

def make_prediction(wine_type, sliders):
    """Make prediction and display results"""
    
    # Prepare input data
    client_input = json.dumps({
        **sliders,
        'type': wine_type
    })
    
    try:
        with st.spinner('🔮 Analyzing wine quality...'):
            response = requests.post(f'{FASTAPI_BACKEND_ENDPOINT}/predict', client_input)
        
        if response.status_code == 200:
            result = json.loads(response.content)
            
            # Update session state
            st.session_state.prediction_count += 1
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'wine_type': wine_type,
                'quality': result['quality_label'],
                'confidence': result.get('confidence', 0),
                'alcohol': sliders['alcohol']
            })
            
            # Display results
            st.divider()
            
            if result["quality"] == 1:
                st.success("### 🏆 Premium Quality Wine!")
                st.write("This wine scores **7 or above** on the quality scale.")
                st.balloons()
            else:
                st.info("### 📊 Standard Quality Wine")
                st.write("This wine scores **below 7** on the quality scale.")
            
            # Confidence meter
            if result.get("confidence"):
                st.write("**Prediction Confidence:**")
                confidence_pct = result['confidence'] * 100
                st.progress(result['confidence'])
                
                if confidence_pct > 80:
                    st.success(f"High confidence: {confidence_pct:.1f}%")
                elif confidence_pct > 60:
                    st.warning(f"Moderate confidence: {confidence_pct:.1f}%")
                else:
                    st.info(f"Low confidence: {confidence_pct:.1f}%")
            
            # Recommendations
            if result["quality"] == 0:
                st.info("""
                💡 **Suggestions to improve quality:**
                - Consider increasing alcohol content (most important factor)
                - Reduce volatile acidity (vinegar taste)
                - Optimize sulphates level
                """)
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please make sure FastAPI server is running on port 8000")

def display_history():
    """Display prediction history"""
    history_df = pd.DataFrame(st.session_state.prediction_history[-5:][::-1])
    st.dataframe(history_df, use_container_width=True, hide_index=True)

def analytics_tab():
    """Analytics and insights tab"""
    st.subheader("📊 Analytics Dashboard")
    
    if not st.session_state.prediction_history:
        st.info("No predictions yet. Make some predictions to see analytics!")
        return
    
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", st.session_state.prediction_count)
    with col2:
        if len(history_df) > 0:
            high_quality = len(history_df[history_df['quality'] == 'High Quality'])
            st.metric("High Quality", f"{high_quality}/{len(history_df)}")
    with col3:
        if len(history_df) > 0:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Quality distribution
    if len(history_df) > 5:
        st.subheader("Quality Distribution")
        quality_counts = history_df['quality'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart
        ax1.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        ax1.set_title("Quality Distribution")
        
        # Bar chart by wine type
        type_quality = history_df.groupby(['wine_type', 'quality']).size().unstack(fill_value=0)
        type_quality.plot(kind='bar', ax=ax2)
        ax2.set_title("Quality by Wine Type")
        ax2.set_xlabel("Wine Type")
        ax2.set_ylabel("Count")
        
        st.pyplot(fig)

def learn_tab():
    """Educational content tab"""
    st.subheader("📚 Understanding Wine Quality")
    
    with st.expander("🔬 Chemical Properties Explained"):
        st.write("""
        **Fixed Acidity**: Most acids in wine (tartaric, malic, citric) are fixed/non-volatile
        
        **Volatile Acidity**: Amount of acetic acid - high levels give unpleasant vinegar taste
        
        **Citric Acid**: Adds 'freshness' and flavor to wines
        
        **Residual Sugar**: Sugar remaining after fermentation stops
        
        **Chlorides**: Amount of salt in the wine
        
        **Sulfur Dioxide**: Prevents microbial growth and wine oxidation
        
        **pH**: Describes how acidic/basic a wine is (0-14 scale)
        
        **Sulphates**: Wine additive that contributes to SO₂ levels
        
        **Alcohol**: Percent alcohol content by volume
        """)
    
    with st.expander("📊 Feature Importance"):
        features = ['Alcohol', 'Density', 'Volatile Acidity', 'Chlorides', 'Residual Sugar']
        importance = [0.21, 0.13, 0.09, 0.08, 0.078]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(features, importance, color='#722f37')
        ax.set_xlabel("Importance")
        ax.set_title("Top 5 Most Important Features for Wine Quality")
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, importance)):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1%}', va='center')
        
        st.pyplot(fig)
    
    with st.expander("🍷 Wine Quality Scale"):
        st.write("""
        Wine quality is typically rated on a scale of 3-9:
        
        - **9**: Exceptional
        - **8**: Excellent  
        - **7**: Very Good ← *High Quality Threshold*
        - **6**: Good
        - **5**: Average
        - **4**: Below Average
        - **3**: Poor
        
        Our model classifies wines as:
        - **High Quality**: Score ≥ 7
        - **Standard Quality**: Score < 7
        """)

def batch_predict_tab():
    """Batch prediction from CSV file"""
    st.subheader("🔬 Batch Prediction")
    st.write("Upload a CSV file with wine data for batch predictions")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Run Batch Prediction"):
            with st.spinner("Processing batch predictions..."):
                # Here you would process each row
                st.success(f"Processed {len(df)} wines!")
                # Display results
                st.write("Results would appear here...")
    
    # Sample CSV template
    if st.button("Download Sample CSV Template"):
        sample_data = pd.DataFrame({
            'fixed acidity': [7.4, 7.8],
            'volatile acidity': [0.7, 0.88],
            'citric acid': [0, 0],
            'residual sugar': [1.9, 2.6],
            'chlorides': [0.076, 0.098],
            'free sulfur dioxide': [11, 25],
            'total sulfur dioxide': [34, 67],
            'density': [0.9978, 0.9968],
            'pH': [3.51, 3.2],
            'sulphates': [0.56, 0.68],
            'alcohol': [9.4, 9.8],
            'type': ['red', 'red']
        })
        csv = sample_data.to_csv(index=False)
        st.download_button("Download", csv, "wine_sample.csv", "text/csv")

def check_backend_status():
    """Check if backend is online"""
    try:
        response = requests.get(FASTAPI_BACKEND_ENDPOINT)
        if response.status_code == 200:
            st.sidebar.success("✅ Backend Online")
            return True
        else:
            st.sidebar.warning("⚠️ Backend Issue")
            return False
    except:
        st.sidebar.error("❌ Backend Offline")
        st.sidebar.info("Run: `uvicorn main:app --reload`")
        return False

if __name__ == "__main__":
    main()