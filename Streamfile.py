#!/usr/bin/env python
# coding: utf-8

# In[1]:



import streamlit as st
import joblib
import numpy as np
import time
# import pandas as pd
import random
import warnings

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
# import matplotlib.pyplot as plt
import plotly.express as px

filename = 'STD_LR_model.joblib'
loaded_model = joblib.load(filename)
# df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>Get Your Credit Card 💳</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# st.title("Get Your Credit Card 💳")

MAPPINGS = {
    "gender" :{'Female': 0, 'Male': 1},
    "marital_status" : {'Married': 0, 'Single': 1},
    "age_group" : {'21-24': 0, '25-34': 1, '35-45': 2, '45+': 3},
    "city" : {'Bengaluru': 0, 'Chennai': 1, 'Delhi NCR': 2, 'Hyderabad': 3, 'Mumbai': 4},
    "occupation" : {'Business Owners': 0, 'Freelancers': 1, 'Government Employees': 2, 'Salaried IT Employees': 3, 'Salaried Other Employees': 4}
}

with st.form("my_form"):
    income=st.number_input(label='Monthly Income (INR)',step=100,min_value=25000,max_value=100000)
    spends=st.number_input(label='Monthly Spends (INR)',step=100,min_value=5000,max_value=50000)
    credit_card_spends=st.number_input(label='Monthly Credit Card Bill (INR)',step=100,min_value=0,max_value=20000, value=0, help="If you already have a credit card")
    income_utilization_perc = round((spends/income)*100,2)
    if credit_card_spends > 0:
        credit_card_spends_perc = round((credit_card_spends/spends)*100,2)
    else:
        credit_card_spends_perc = round(random.uniform(26, 50),2)

    gender = st.radio("Gender",["Male", "Female"],horizontal=True)

    marital_status = st.radio("Marital Status",["Single", "Married"],horizontal=True)

    age_group = st.radio("Age Group you belong to",["21-24", "25-34", "35-45", "45+"],horizontal=True)

    city = st.selectbox("City you live in",["Bengaluru", "Chennai", "Delhi NCR", "Hyderabad", "Mumbai"])

    occupation = st.selectbox("Occupation",["Business Owners", "Freelancers", "Government Employees", "Salaried IT Employees", "Salaried Other Employees"])
    
    
    submitted = st.form_submit_button("Submit")

if submitted:

    with st.spinner("Processing..."):

        gender = MAPPINGS["gender"][gender]
        marital_status = MAPPINGS["marital_status"][marital_status]
        age_group = MAPPINGS["age_group"][age_group]
        city = MAPPINGS["city"][city]
        occupation = MAPPINGS["occupation"][occupation]
        DTI = round((spends * (credit_card_spends_perc/100))/ income, 2)
        print(DTI)
        credit_limit_range = (income * DTI * 2.5, income * DTI * 5.5)
        data=[income,spends,credit_card_spends, income_utilization_perc, credit_card_spends_perc,gender,marital_status,age_group,city,occupation]


        # data = [69308, 29267, 13698, 42.23, 46.80, 1, 0, 1, 4, 0]
        # print(data)
        data = np.array(data).reshape(1, -1)
        pred = loaded_model.predict(data)

        # Simulate processing delay
        time.sleep(3)

    
    st.success(f'The customer belongs to cluster-{pred.item()+1}')
    st.success(f'💳 Recommended Limit Range: {credit_limit_range[0]:.2f} INR to  {credit_limit_range[1]:.2f} INR')
    # print(pred.item())
    # print(max(loaded_model.predict_proba(data)[0]))
    percentages = loaded_model.predict_proba(data)[0]
    cluster_names = [f'Cluster-{i+1}' for i in range(len(percentages))]
    
   # Create a pie chart using Plotly Express
    # fig = px.pie(
    #     values=percentages*100,
    #     names=cluster_names,
    #     title='Probability Distribution',
    #     labels={'values': 'Probability'},
    # )

    fig = px.bar(
        x=cluster_names,
        y=percentages,
        title='Probability Distribution',
        labels={'y': 'Probability', 'x': 'Cluster'},
        text=[f'{percent:.2%}' for percent in percentages]
    )

    fig.update_layout( width=600, height=400)

    # Display the pie chart using Streamlit
    st.plotly_chart(fig)

 # Add footer text
    
# Add content to the sidebar, including profile information
with st.sidebar:
    st.write("## New Credit Card Launch ")
    # st.image("https://placekitten.com/80/80", caption="My Profile Picture", use_column_width=True, class_="profile-img")
    st.write("*****")
    st.write("*****")
    st.write("*****")
   
st.markdown(
    """
    
    <div style="text-align: center;">
        <hr style="border: 1px solid #ddd; width: 100%;">
        <h5>This can be used by Customers or Bank </a></h5>
    </div>
    
    """,
    unsafe_allow_html=True
)


# In[ ]:




