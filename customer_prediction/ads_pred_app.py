import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

# Page setting
st.set_page_config(page_title="Ad Click Prediction App", layout="centered", page_icon="📈")

st.title("AD CLICK PREDICTION")
try:
    img = Image.open('ads_pic.jpg')
    st.image(img, use_column_width=True)
except Exception as e:
    st.error(f"Error loading image: {e}")

st.text("Fill in the following values to predict whether the user will click on the ad")

# Input features
daily_time_spent_on_site = st.slider('Daily Time Spent on Site (minutes):', 0.0, 500.0, step=0.1, value=60.0)
age = st.slider('Age:', 10, 100, step=1, value=30)
area_income = st.number_input('Area Income (average income of the user’s area):', 0.0, 2000000.0, step=100.0, value=50000.0)
daily_internet_usage = st.slider('Daily Internet Usage (minutes):', 0.0, 1000.0, step=0.1, value=200.0)
gender = st.selectbox('Gender:', ['Male', 'Female'])
topic_of_interest = st.text_input("Topic of Interest (e.g., Technology, Fitness, Travel):")

# Function to encode age into an age group
def findAgeGroup(Age):
    if Age >= 19 and Age < 35:
        return 1
    elif Age >= 35 and Age < 50:
        return 2
    else:
        return 3

# Submit button
btn = st.button("Predict")

if btn:
    if topic_of_interest.strip() == "":
        st.warning("Please enter a topic of interest.")
    else:
        # Load the trained model
        model = joblib.load('ad_click_model.pkl')

        # Preprocessing
        gender_binary = 1 if gender == "Male" else 0
        
        # Encode age into age group
        age_group = findAgeGroup(age)

        # Prepare input data
        input_data = np.array([[daily_time_spent_on_site, age_group, area_income, daily_internet_usage, gender_binary]])

        # Prediction
        prediction = model.predict(input_data)

        # Personalized result
        if prediction[0] == 1:
            st.success(f"The user is likely to click on the next ad about {topic_of_interest}.")
        else:
            st.error(f"The user is unlikely to click on the next ad about {topic_of_interest}.")
