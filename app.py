import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.title('End-to-End Regression Model')

# Load data
st.write('Loading data...')
df = pd.read_csv('https://storage.googleapis.com/dataset-uploader/california_housing.csv')

# Split data into train and test sets
st.write('Splitting data into train and test sets...')
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
st.write('Training model...')
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
st.write('Evaluating model...')
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write('Mean Absolute Error:', mae)

# Save model
st.write('Saving model...')
model.save('model.pkl')
