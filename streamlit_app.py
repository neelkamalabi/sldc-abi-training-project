"""
Streamlit UI for Iris Classification using FastAPI backend.
"""
import os

import requests
import streamlit as st

# Configuration - use environment variable for Docker compatibility
API_URL = os.getenv("API_BASE", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# Title and description
st.title("🌸 Iris Flower Classifier")
st.markdown("""
This app uses a Machine Learning model to classify Iris flowers into three species:
- **Setosa** 🌼
- **Versicolor** 🌺
- **Virginica** 🌻

Enter the measurements of an Iris flower below to get a prediction.
""")

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("status") == "healthy":
            st.success("✅ API is running and model is loaded")
        else:
            st.error("❌ API is not healthy")
    else:
        st.error("❌ Cannot connect to API")
except requests.exceptions.RequestException:
    st.error("❌ API is not running. Please start the FastAPI server first.")
    st.info("Run: `uvicorn iris_api.app:app --reload`")
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Features")
    
    # Input fields for features
    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.1,
        step=0.1,
        help="Length of the sepal in centimeters"
    )
    
    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.5,
        step=0.1,
        help="Width of the sepal in centimeters"
    )
    
    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=1.4,
        step=0.1,
        help="Length of the petal in centimeters"
    )
    
    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=0.2,
        step=0.1,
        help="Width of the petal in centimeters"
    )
    
    # Predict button
    predict_button = st.button("🔮 Predict Species", type="primary", use_container_width=True)

with col2:
    st.header("Prediction Results")
    
    if predict_button:
        # Prepare request data
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        
        # Make prediction request
        with st.spinner("Making prediction..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display prediction
                    species = result["predicted_species"]
                    confidence = result["confidence"]
                    
                    # Species emoji mapping
                    species_emoji = {
                        "setosa": "🌼",
                        "versicolor": "🌺",
                        "virginica": "🌻"
                    }
                    
                    # Display main prediction
                    st.success(f"### {species_emoji.get(species, '🌸')} Predicted Species: **{species.capitalize()}**")
                    st.metric("Confidence", f"{confidence * 100:.2f}%")
                    
                    # Display probabilities
                    st.subheader("Class Probabilities")
                    probabilities = result["probabilities"]
                    
                    for cls, prob in probabilities.items():
                        emoji = species_emoji.get(cls, "🌸")
                        st.progress(prob, text=f"{emoji} {cls.capitalize()}: {prob * 100:.2f}%")
                    
                    # Display input summary
                    with st.expander("📊 Input Summary"):
                        st.json({
                            "Sepal Length": f"{sepal_length} cm",
                            "Sepal Width": f"{sepal_width} cm",
                            "Petal Length": f"{petal_length} cm",
                            "Petal Width": f"{petal_width} cm"
                        })
                
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {str(e)}")
    else:
        st.info("👈 Adjust the feature values and click **Predict Species** to see results")

# Information section
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    ### Model Information
    - **Algorithm**: Random Forest Classifier
    - **Features**: Sepal Length, Sepal Width, Petal Length, Petal Width
    - **Classes**: Setosa, Versicolor, Virginica
    
    ### Feature Ranges (typical)
    - **Setosa**: Small petals (1-2 cm), medium sepals
    - **Versicolor**: Medium petals (3-5 cm), medium sepals
    - **Virginica**: Large petals (4-7 cm), large sepals
    
    ### API Endpoint
    The predictions are made by calling the FastAPI backend at `{API_URL}/predict`
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit 🎈 and FastAPI ⚡"
    "</div>",
    unsafe_allow_html=True
)
