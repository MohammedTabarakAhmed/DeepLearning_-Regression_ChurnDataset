import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

#load the model
model=tf.keras.models.load_model('model.h5')

#encode and scaler
#load encoder & scaler
with open('geo_encoder_df.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#streamlit app
st.title('Customer Churn Predcition')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])  ## → array(['France', 'Germany', 'Spain']) the array is the first [0] so everything
gender = st.selectbox('Gender', label_encoder_gender.classes_) #the UI with male female
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10) #how many year they have been working in the bank
num_of_products = st.slider('Number of Products', 1, 4) #loan,mutual funds etcetra
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]], #ML model doesnt understand male female so convert it into 0 and 1s "backend not like UI like above"
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba >= 0.5:
    print('Customer likely to churn')
else:
    print('Not Churn')