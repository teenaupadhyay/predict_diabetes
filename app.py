import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

df = pd.read_csv(r"./diabetes.csv")

# Update header with enhanced styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem; 
        color: #264653; 
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem; 
        color: #2A9D8F;
        text-align: center;
    }
    .result {
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🌟 Diabetes Prediction App 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Get a diabetes prediction by entering your health data below.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 3px solid #E9C46A;'>", unsafe_allow_html=True)

# Sidebar header with a cleaner look
st.sidebar.markdown("<h2 style='text-align: center;'>🩺 Patient Health Input</h2>", unsafe_allow_html=True)
st.sidebar.write("Please enter the following health details:")

def get_user_input():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI (Body Mass Index)', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin Level (IU/mL)', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age (years)', min_value=21, max_value=88, value=33)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    return pd.DataFrame(user_data, index=[0])

user_data = get_user_input()

# Subheader with modern styling
st.markdown("<h2 style='text-align: center; color: #2A9D8F;'>📝 Your Entered Health Data</h2>", unsafe_allow_html=True)
st.dataframe(user_data)

# Train/test split and model training
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

if st.button('Predict'):
    st.markdown("<h3 style='text-align: center;'>🔄 Running the Prediction...</h3>", unsafe_allow_html=True)
    
    # Smoother progress animation
    progress_bar = st.progress(0)
    for i in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar.progress(i)
    
    # Model prediction
    prediction = rf.predict(user_data)

    # Display result with improved color scheme
    st.markdown("<hr style='border-top: 2px solid #E9C46A;'>", unsafe_allow_html=True)
    result = 'You are not Diabetic' if prediction[0] == 0 else 'You are Diabetic'
    result_color = '#2A9D8F' if prediction[0] == 0 else '#E76F51'
    st.markdown(f"<h2 class='result' style='text-align: center; color: {result_color};'>{result}</h2>", unsafe_allow_html=True)
    
    # Display model accuracy
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.markdown("<h3 style='text-align: center;'>📊 Model Accuracy</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 1.5rem;'>{accuracy:.2f}%</p>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='text-align: center;'>🖱️ Click 'Predict' to see the result</h3>", unsafe_allow_html=True)
