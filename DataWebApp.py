import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

'''Writing in Markdown format'''
st.write("""
#Iris Flower Prediction App

This app will predict the Iris Flower type!
""")

'''Creating a side-bar for user input'''
st.sidebar.header("User Input Parameters:")

'''Function that takes input from user using slider in the sidebar'''
def taking_input_parameters():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4) #Slider_name | min_value | max_value | default_value
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4) #Slider_name | min_value | max_value | default_value
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3) #Slider_name | min_value | max_value | default_value
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2) #Slider_name | min_value | max_value | default_value

    data = {
        "Sepal Length" : sepal_length,
        "Sepal Width" : sepal_width,
        "Petal Length" : petal_length,
        "Petal Width" : petal_width
    }

    features = pd.DataFrame(data, index = [0])
    return features

'''Displaying a dataframe of user inputs'''
df = taking_input_parameters()

st.subheader("User Input Parameters:")
st.write(df)

'''Loading data to train model'''
iris = load_iris()
X = iris.data
Y = iris.target

'''Fitting data into the model'''
clf = KNeighborsClassifier()
clf.fit(X,Y)

'''Making Predictions on user input'''
prediction = clf.predict(df)
'''Probability of predicton'''
prediction_proba = clf.predict_proba(df)

'''Legend'''
st.subheader("Class labels and their corresponding index number:")
st.write(iris.target_names)

'''Displaying Prediction'''
st.subheader("Prediction")
st.write(iris.target_names[prediction])

'''Displaying Prediction Probability'''
st.subheader("Prediction Probability")
st.write(prediction_proba)