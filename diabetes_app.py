# diabetes_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Checker",
    page_icon="🩺",
    layout="wide"
)

# Load models and scaler
@st.cache_resource
def load_models():
    models = {}
    try:
        models['logistic'] = pickle.load(open('diabetes_model_logistic_regression.pkl', 'rb'))
        models['random_forest'] = pickle.load(open('diabetes_model_random_forest.pkl', 'rb'))
        models['xgboost'] = pickle.load(open('diabetes_model_xgboost.pkl', 'rb'))
        scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def create_risk_categories_single(value, feature_type):
    """Create risk categories for single values"""
    if feature_type == 'glucose':
        if value <= 100: return 0
        elif value <= 126: return 1
        elif value <= 200: return 2
        else: return 3
    elif feature_type == 'bmi':
        if value <= 18.5: return 0
        elif value <= 25: return 1
        elif value <= 30: return 2
        else: return 3
    elif feature_type == 'age':
        if value <= 30: return 0
        elif value <= 45: return 1
        elif value <= 60: return 2
        else: return 3

def main():
    st.title("🩺 Diabetes Risk Checker")
    st.write("This app predicts the risk of diabetes based on health parameters using machine learning.")
    
    # Load models
    models, scaler = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check if all model files are available.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("Patient Health Parameters")
    
    # Create input fields
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Normal Ranges:**
    - Glucose: 70-100 mg/dL (fasting)
    - Blood Pressure: < 120/80 mm Hg
    - BMI: 18.5-24.9
    - Insulin: 2-20 μU/mL (fasting)
    """)
    
    # Create feature array
    features = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, diabetes_pedigree, age
    ]])
    
    # Create engineered features
    glucose_bmi_ratio = glucose / bmi if bmi > 0 else 0
    bp_age_ratio = blood_pressure / age if age > 0 else 0
    insulin_glucose_ratio = insulin / glucose if glucose > 0 else 0
    
    glucose_risk = create_risk_categories_single(glucose, 'glucose')
    bmi_risk = create_risk_categories_single(bmi, 'bmi')
    age_risk = create_risk_categories_single(age, 'age')
    
    # Add engineered features
    features_with_engineered = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, diabetes_pedigree, age,
        glucose_bmi_ratio, bp_age_ratio, insulin_glucose_ratio,
        glucose_risk, bmi_risk, age_risk
    ]])
    
    # Scale features
    numerical_features = features_with_engineered[:, :11]  # First 11 features are numerical
    scaled_numerical = scaler.transform(numerical_features)
    
    # Replace the numerical features with scaled ones
    features_scaled = features_with_engineered.copy()
    features_scaled[:, :11] = scaled_numerical
    
    # Main content area
    st.header("Risk Assessment")
    
    # Display current parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Glucose", f"{glucose} mg/dL", 
                 delta="Normal" if glucose <= 100 else "High" if glucose <= 126 else "Very High",
                 delta_color="normal" if glucose <= 100 else "off" if glucose <= 126 else "inverse")
    
    with col2:
        st.metric("BMI", f"{bmi:.1f}", 
                 delta="Underweight" if bmi < 18.5 else "Normal" if bmi <= 25 else "Overweight" if bmi <= 30 else "Obese",
                 delta_color="normal" if 18.5 <= bmi <= 25 else "off" if bmi <= 30 else "inverse")
    
    with col3:
        st.metric("Blood Pressure", f"{blood_pressure} mm Hg",
                 delta="Normal" if blood_pressure < 120 else "Elevated" if blood_pressure < 130 else "High",
                 delta_color="normal" if blood_pressure < 120 else "off" if blood_pressure < 130 else "inverse")
    
    # Prediction section
    st.header("Diabetes Risk Prediction")
    
    if st.button("Check Diabetes Risk", type="primary"):
        with st.spinner("Analyzing health parameters..."):
            # Make predictions with all models
            results = {}
            for model_name, model in models.items():
                prediction = model.predict(features_scaled)
                probability = model.predict_proba(features_scaled)
                results[model_name] = {
                    'prediction': prediction[0],
                    'probability': probability[0][1]  # Probability of diabetes
                }
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Logistic Regression")
            pred = results['logistic']['prediction']
            prob = results['logistic']['probability']
            st.metric(
                label="Risk Level", 
                value="High Risk" if pred == 1 else "Low Risk",
                delta=f"{prob:.1%} probability",
                delta_color="inverse" if pred == 1 else "normal"
            )
        
        with col2:
            st.subheader("Random Forest")
            pred = results['random_forest']['prediction']
            prob = results['random_forest']['probability']
            st.metric(
                label="Risk Level", 
                value="High Risk" if pred == 1 else "Low Risk",
                delta=f"{prob:.1%} probability",
                delta_color="inverse" if pred == 1 else "normal"
            )
        
        with col3:
            st.subheader("XGBoost")
            pred = results['xgboost']['prediction']
            prob = results['xgboost']['probability']
            st.metric(
                label="Risk Level", 
                value="High Risk" if pred == 1 else "Low Risk",
                delta=f"{prob:.1%} probability",
                delta_color="inverse" if pred == 1 else "normal"
            )
        
        # Show consensus and recommendations
        st.subheader("Overall Assessment")
        positive_count = sum(1 for result in results.values() if result['prediction'] == 1)
        total_models = len(results)
        avg_probability = np.mean([result['probability'] for result in results.values()])
        
        if positive_count == total_models:
            st.error("🔴 **High Diabetes Risk Detected**")
            st.warning("""
            **Recommendations:**
            - Consult with a healthcare professional immediately
            - Monitor blood sugar levels regularly
            - Consider lifestyle changes (diet, exercise)
            - Schedule follow-up tests
            """)
        elif positive_count >= total_models / 2:
            st.warning("🟡 **Moderate Diabetes Risk**")
            st.info("""
            **Recommendations:**
            - Schedule a check-up with your doctor
            - Maintain healthy lifestyle habits
            - Monitor your health parameters
            - Consider preventive measures
            """)
        else:
            st.success("🟢 **Low Diabetes Risk**")
            st.info("""
            **Maintenance Tips:**
            - Continue healthy lifestyle habits
            - Regular exercise and balanced diet
            - Annual health check-ups
            - Maintain current healthy parameters
            """)
        
        # Risk probability gauge
        st.subheader("Risk Probability")
        st.progress(avg_probability)
        st.write(f"Average probability of diabetes: **{avg_probability:.1%}**")
    
    # Information section
    st.markdown("---")
    st.header("About This Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Parameters Used:**
        - Pregnancies
        - Glucose Level
        - Blood Pressure
        - Skin Thickness
        - Insulin Level
        - BMI
        - Diabetes Pedigree
        - Age
        """)
    
    with col2:
        st.warning("""
        **Important Disclaimer:**
        This tool is for educational purposes only and is not a substitute for professional medical advice. 
        Always consult with healthcare professionals for medical diagnoses.
        """)

if __name__ == "__main__":
    main()
