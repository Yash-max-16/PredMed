# importing the required libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pickle as pkl
import numpy as np

# loading the saved models
diabetes_model = pkl.load(open('diabetes_model2.pkl', 'rb'))
cancer_model = pkl.load(open('cancer_model.pkl', 'rb'))
heart_model = pkl.load(open('heart_model.pkl', 'rb'))
parkinsons_model = pkl.load(open('parkinsons_model.pkl', 'rb'))

# making a sidebar for navigating to different disease prediction systems
with st.sidebar:
    selected = option_menu('PredMed',
                           ['Home', 'Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Breast Cancer Prediction'],
                           icons = ['house-fill', 'droplet-fill', 'heart-fill', 'person-square', 'activity'],
                           menu_icon = 'prescription2',
                           default_index = 0)

# Home page
if selected == 'Home':
    # page title
    st.title('PredMed - Your Personal Medical Predictor')
    content = '''

    Welcome to PredMed, your personalized medical predictor powered by cutting-edge machine learning algorithms. PredMed is a user-friendly web application that can assist in predicting the likelihood of four critical medical conditions: diabetes, heart disease, Parkinson's disease, and breast cancer. Empowering users with advanced technology, PredMed aims to promote early detection and proactive health management.

    **How it Works:**
    1. **Easy Input:** Simply enter relevant health features and medical data, and let PredMed do the rest.
    2. **Machine Learning Magic:** Our sophisticated machine learning models analyze your inputs to generate accurate predictions.
    3. **Instant Results:** Within moments, PredMed provides you with actionable insights and risk assessments.

    **Why Use PredMed?**
    1. **Early Detection:** Detect potential health risks early, enabling timely preventive measures.
    2. **Empowering Decisions:** Make informed decisions about your health with evidence-based predictions.

    **Conditions We Predict:**
    1. **Diabetes:** Assess your risk of developing diabetes based on key health indicators.
    2. **Heart Disease:** Predict your risk of heart disease and take proactive measures for a healthier heart.
    3. **Parkinson's Disease:** Determine your likelihood of developing Parkinson's for better health planning.
    4. **Breast Cancer:** Get valuable insights into your breast cancer risk, encouraging early screening.

    **Disclaimer:**
    Please note that PredMed is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for personalized medical guidance.

    **Get Started Now:**
    Navigate to different disease predictors using the option menu at the left end.

    **Contact Us:**
    For any inquiries or feedback, please reach out to us at rattihalliyashas@gmail.com.
    '''

    # Display the content using Streamlit
    st.markdown(content)

# Diabetes Prediction page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction')
    # asking details for prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of pregnancies')
    with col2:
        Glucose = st.number_input('Glucose level')
    with col3:
        BloodPressure = st.number_input('Blood pressure')
    with col1:
        SkinThickness = st.number_input('Skin Thickness')
    with col2:
        Insulin = st.number_input('Insulin value')
    with col3:
        BMI = st.number_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.number_input('Age')
    # processing the features
    features = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    features = np.array(features)
    scale = pkl.load(open('diabetes_scale.pkl', 'rb'))
    scaled_features = (features - scale['mean']) / scale['std_dev']
    # making the prediction
    diab_diagnosis = ''
    if st.button('Predict'):
        diab_pred = diabetes_model.predict([scaled_features])
        if(diab_pred[0] == 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
    st.success(diab_diagnosis)

# Heart Disease Prediction page
if selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction')
    # asking details for prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age')
    with col2:
        options_sex = ['female', 'male']
        sex = st.selectbox('Select sex', options_sex)
        sex = options_sex.index(sex)
    with col3:
        cp = st.number_input('Chest pain types')
    with col1:
        trestbps = st.number_input('Resting blood pressure')
    with col2:
        chol = st.number_input('Serum cholestral in mg/dL')
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dL')
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.number_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
    with col3:
        options_thal = ['unknown', 'normal', 'fixed defect', 'reversable defect']
        thal = st.selectbox('Select thal', options_thal)
        thal = options_thal.index(thal)
    with col1:
        ca = st.number_input('Major vessels colored by flourosopy')
    # processing the features
    features = [age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal]
    features = np.array(features)
    scale = pkl.load(open('heart_scale.pkl', 'rb'))
    min_values = np.array(scale['min'])
    max_values = np.array(scale['max'])
    scaled_features = (features - min_values) / (max_values - min_values)
    # making the prediction
    heart_diagnosis = ''
    if st.button('Predict'):
        heart_pred = heart_model.predict([scaled_features])
        if (heart_pred[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person is not having heart disease'
    st.success(heart_diagnosis)

# Parkinsons Prediction page
if selected == 'Parkinsons Prediction':
    # page title
    st.title('Parkinsons Prediction')
    # asking details for prediction
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.number_input('MDVP:RAP')
    with col2:
        PPQ = st.number_input('MDVP:PPQ')
    with col3:
        DDP = st.number_input('Jitter:DDP')
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5')
    with col3:
        APQ = st.number_input('MDVP:APQ')
    with col4:
        DDA = st.number_input('Shimmer:DDA')
    with col5:
        NHR = st.number_input('NHR')
    with col1:
        HNR = st.number_input('HNR')
    with col2:
        RPDE = st.number_input('RPDE')
    with col3:
        DFA = st.number_input('DFA')
    with col4:
        spread1 = st.number_input('spread1')
    with col5:
        spread2 = st.number_input('spread2')
    with col1:
        D2 = st.number_input('D2')
    with col2:
        PPE = st.number_input('PPE')
    # processing the features
    features = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,
                Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]
    features = np.array(features)
    scale = pkl.load(open('parkinsons_scale.pkl', 'rb'))
    min_values = np.array(scale['min'])
    max_values = np.array(scale['max'])
    scaled_features = (features - min_values) / (max_values - min_values)
    # making the prediction
    parkinsons_diagnosis = ''
    if st.button('Predict'):
        parkinsons_pred = parkinsons_model.predict([scaled_features])
        if (parkinsons_pred[0] == 1):
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
    st.success(parkinsons_diagnosis)

# Breast Cancer Prediction page
if selected == 'Breast Cancer Prediction':
    # page title
    st.title('Breast Cancer Prediction')
    # asking details for prediction
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        radius_mean = st.number_input('Radius Mean')
    with col2:
        texture_mean = st.number_input('Texture Mean')
    with col3:
        perimeter_mean = st.number_input('Perimeter Mean')
    with col4:
        area_mean = st.number_input('Area Mean')
    with col5:
        smoothness_mean = st.number_input('Smoothness Mean')
    with col1:
        compactness_mean = st.number_input('Compactness Mean')
    with col2:
        concavity_mean = st.number_input('Concavity Mean')
    with col3:
        concave_points_mean = st.number_input('Concave Points Mean')
    with col4:
        symmetry_mean = st.number_input('Symmetry Mean')
    with col5:
        fractal_dimension_mean = st.number_input('Fractal Dimension Mean')
    with col1:
        radius_se = st.number_input('Radius SE')
    with col2:
        texture_se = st.number_input('Texture SE')
    with col3:
        perimeter_se = st.number_input('Perimeter SE')
    with col4:
        area_se = st.number_input('Area SE')
    with col5:
        smoothness_se = st.number_input('Smoothness SE')
    with col1:
        compactness_se = st.number_input('Compactness SE')
    with col2:
        concavity_se = st.number_input('Concavity SE')
    with col3:
        concave_points_se = st.number_input('Concave Points SE')
    with col4:
        symmetry_se = st.number_input('Symmetry SE')
    with col5:
        fractal_dimension_se = st.number_input('Fractal Dimension SE')
    with col1:
        radius_worst = st.number_input('Radius Worst')
    with col2:
        texture_worst = st.number_input('Texture Worst')
    with col3:
        perimeter_worst = st.number_input('Perimeter Worst')
    with col4:
        area_worst = st.number_input('Area Worst')
    with col5:
        smoothness_worst = st.number_input('Smoothness Worst')
    with col1:
        compactness_worst = st.number_input('Compactness Worst')
    with col2:
        concavity_worst = st.number_input('Concavity Worst')
    with col3:
        concave_points_worst = st.number_input('Concave Points Worst')
    with col4:
        symmetry_worst = st.number_input('Symmetry Worst')
    with col5:
        fractal_dimension_worst = st.number_input('Fractal Dimension Worst')
    # processing the features
    features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
                texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
                concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
                concave_points_worst, symmetry_worst, fractal_dimension_worst]
    features = np.array(features)
    scale = pkl.load(open('cancer_scale.pkl', 'rb'))
    scaled_features = (features - scale['mean']) / scale['std_dev']
    # making the prediction
    cancer_diagnosis = ''
    if st.button('Predict'):
        cancer_pred = cancer_model.predict([scaled_features])
        if (cancer_pred[0] == 1):
            cancer_diagnosis = 'The cancer is Malignant'
        else:
            cancer_diagnosis = 'The cancer is Benign'
    st.success(cancer_diagnosis)