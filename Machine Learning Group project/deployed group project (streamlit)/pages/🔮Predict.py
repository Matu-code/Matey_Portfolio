import streamlit as st
from functions import *

import logging
logging.basicConfig(filename='./logs/0.log', encoding='utf-8', level=logging.INFO)

logging.info('Loading model')
model = import_model()
logging.info('Loaded')
logging.info('Loading data')
X, y = load_data()
logging.info('Loaded')

logging.info('Setting feature and class names')
feature_names = X.columns.tolist()
class_names = ['Low crime','Mid crime','High crime']

logging.info('Display coefficients')
st.write('# **:green[Feature importance]**')
plot_coefficients_separate(model.coef_, 0, feature_names, class_names, 'red')
plot_coefficients_separate(model.coef_, 1, feature_names, class_names, 'green')
plot_coefficients_separate(model.coef_, 2, feature_names, class_names, 'blue')


st.markdown("### **The final dataframe which contains information such as income level, education level and age of the citizens per neighborhood**")
logging.info('Display dataframe**')
st.dataframe(X)


logging.info('Create input fields')
custom = []
st.markdown("### :green[Number of people with different age groups]")
custom.append(st.number_input('Number of people < 30 year old', value = X['< 30 jaar'].iloc[0]))
custom.append(st.number_input('Number of people 30-44 year old', value = X['30-44 jaar'].iloc[0]))
custom.append(st.number_input('Number of people 45-64 year old', value = X['45-64 jaar'].iloc[0]))
custom.append(st.number_input('Number of people 65-74 year old', value = X['65-74 jaar'].iloc[0]))
custom.append(st.number_input('Number of people >= 75 year old', value = X['>= 75 jaar'].iloc[0]))
st.markdown("### :green[Percentage of low and high income]")
custom.append(st.slider('Percentage of high income households '))
custom.append(st.slider('Percentage of low income households percentage'))
st.markdown("### :green[Different income levels [High,Medium and Low]]")
custom.append(st.number_input('Number of people with high education', value = X['Hoog'].iloc[0]))
custom.append(st.number_input('Number of people with low education', value = X['Laag'].iloc[0]))
custom.append(st.number_input('Number of people with medium education', value = X['Midden'].iloc[0]))
custom.append(st.number_input('QoL', value = X['2019'].iloc[0]))
custom = np.array(custom)
custom = custom.reshape(1, -1)

if st.button('Predict'):
    logging.info('Button "Predict" pressed')
    prediction = predict_m(model, custom)
    st.write(prediction)




