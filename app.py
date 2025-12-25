import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Model ---
model = joblib.load('census_model.pkl')

# --- 2. Title and Description ---
st.title("Census Income Prediction")
st.markdown("""
This app predicts whether a person's income is **<=50K** or **>50K** based on 1994 Census data.
""")

# --- 3. Sidebar Inputs ---
st.sidebar.header("User Details")

def user_input_features():
    # Numeric Inputs
    age = st.sidebar.slider("Age", 17, 90, 30)
    hours_per_week = st.sidebar.slider("Hours Worked per Week", 1, 99, 40)
    capital_gain = st.sidebar.number_input("Capital Gain", value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", value=0)

    # Categorical Inputs
    workclass = st.sidebar.selectbox("Workclass", 
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
         'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])

    education = st.sidebar.selectbox("Education Level", 
        ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
         'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
         '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])

    marital_status = st.sidebar.selectbox("Marital Status", 
        ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 
         'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])

    occupation = st.sidebar.selectbox("Occupation", 
        ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
         'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
         'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
         'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
         'Armed-Forces'])

    relationship = st.sidebar.selectbox("Relationship", 
        ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])

    race = st.sidebar.selectbox("Race", 
        ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])

    sex = st.sidebar.selectbox("Sex", ['Female', 'Male'])

    native_country = st.sidebar.selectbox("Native Country", 
        ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 
         'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 
         'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
         'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 
         'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 
         'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 
         'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

    # Combine into a DataFrame
    # Note: 'fnlwgt' and 'education.num' are included as placeholders 
    # because the pipeline expects these columns to exist, even if it drops them.
    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': 0, 
        'education': education,
        'education.num': 0, 
        'marital.status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
        'hours.per.week': hours_per_week,
        'native.country': native_country
    }
    return pd.DataFrame(data, index=[0])

# --- 4. Main Page Display ---
input_df = user_input_features()

st.subheader("User Parameters")
st.write(input_df)

# --- 5. Prediction Logic ---
if st.button("Predict Income"):
    # The pipeline handles all preprocessing (scaling, encoding, etc.)
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result:")
    if prediction == '<=50K':
        st.success(f"Income Prediction: {prediction}")
    else:
        st.warning(f"Income Prediction: {prediction}")
