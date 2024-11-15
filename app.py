import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
st.set_page_config(page_title="Customer Churn Prediction and EDA", page_icon=":guardsman:")
df = pd.read_csv('train.csv')
rf = joblib.load('churn_model.pkl')

label_encoders = {}
label_encoders['state'] = LabelEncoder()
label_encoders['state'].fit(df['state'])
label_encoders['gender'] = LabelEncoder()
label_encoders['gender'].fit(df['gender'])

st.title("Customer Churn Prediction and EDA")

tabs = st.radio("Select Page", ['Exploratory Data Analysis', 'Churn Prediction'])

if tabs == 'Exploratory Data Analysis':
    st.header("Basic Information and Statistics")
    st.write(df.info())
    st.write(df.describe())
    # st.write("Missing Values:")
    # st.write(df.isnull().sum())

    df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
    df['state'] = df['state'].fillna(df['state'].mode()[0])

    st.subheader("Univariate Analysis")

    st.write("State Distribution:")
    fig, ax = plt.subplots()
    sns.countplot(x='state', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Gender Distribution:")
    fig, ax = plt.subplots()
    sns.countplot(x='gender', data=df, ax=ax)
    st.pyplot(fig)

    numerical_columns = ['credit_score', 'age', 'tenure', 'balance', 'no_of_products', 'estimated_salary']
    for col in numerical_columns:
        st.write(f"Distribution of {col}:")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Bivariate Analysis")

    st.write("Credit Score by Gender:")
    fig, ax = plt.subplots()
    sns.boxplot(x='gender', y='credit_score', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Balance by State:")
    fig, ax = plt.subplots()
    sns.boxplot(x='state', y='balance', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Outlier Detection")
    for col in numerical_columns:
        st.write(f"Boxplot of {col}:")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Boxplot of {col}')
        st.pyplot(fig)

    df['churn_binary'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    st.write("Churn column created.")

elif tabs == 'Churn Prediction':
    st.header("Churn Prediction App")

    credit_score = st.number_input('Credit Score', min_value=int(df['credit_score'].min()), max_value=int(df['credit_score'].max()), step=1)
    state = st.selectbox('State', options=df['state'].unique())
    gender = st.selectbox('Gender', options=df['gender'].unique())
    age = st.number_input('Age', min_value=int(df['age'].min()), max_value=int(df['age'].max()), step=1)
    tenure = st.number_input('Tenure (Years)', min_value=int(df['tenure'].min()), max_value=int(df['tenure'].max()), step=1)
    balance = st.number_input('Balance', min_value=0, step=1)
    no_of_products = st.number_input('Number of Products', min_value=1, max_value=4, step=1)
    credit_card = st.selectbox('Credit Card', options=df['credit_card'].unique())
    active = st.selectbox('Active', options=df['active'].unique())
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=0.1)

    state = label_encoders['state'].transform([state])[0]
    gender = label_encoders['gender'].transform([gender])[0]

    user_input = pd.DataFrame({
        'credit_score': [credit_score],
        'state': [state],
        'gender': [gender],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'no_of_products': [no_of_products],
        'credit_card': [credit_card],
        'active': [active],
        'estimated_salary': [estimated_salary]
    })

    if st.button('Predict Churn'):
        prediction = rf.predict(user_input)[0]
        
        if prediction == 1:
            st.write("The customer is likely to churn.")
        else:
            st.write("The customer is unlikely to churn.")
