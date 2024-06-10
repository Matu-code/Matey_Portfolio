#This file contains functions used by streamlit
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.svm import SVC
import logging
logging.basicConfig(filename='./logs/0.log', encoding='utf-8', level=logging.INFO)

def import_model(model='class', path='pages/'):
    '''
    @Author Hubert
    Import the clustering or classification model
    
    Parameters
    ----------
    model: string
        'class' for classification model.
        'clust' for clustering model.
    
    Returns
    model
    '''
    if model == 'class':
        logging.info('Loading classification model')
        filename = path+'model_class.sav'
    elif model == 'clust':
        logging.info('Loading clustering model')
        filename = path+'model_clust.sav'
    else:
        logging.critical('Model not specified')
        return 1
    
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

#Not realy needed
def predict_m(model, data):
    '''
    @Author Hubert
    Perform prediction utilizing specified model on provided data
    
    Parameters
    ----------    -------

    model: model
        Model that will perform prediction.
    data: DataFrame
        DataFrame containing data to perform prediction.
    
    Returns
    -------
    prediction: string
    '''
    logging.info('Predicting crime rate')
    prediction = model.predict(data)
    if prediction == 0:
        prediction = 'Low crime'
    elif prediction == 1:
        prediction = 'Medium crime'
    elif prediction == 2:
        prediction = 'High crime'
    else:
        logging.critical('Crime rate out of class range')
        return 1
    return prediction

st.cache_data()
def load_data(path='data/'):
    '''
    @Author Hubert
    Load DataFrame
    path: string
        path to X.csv and y.csv

    Returns
    -------
    X: DataFrame
    y: DataFrame
    '''
    X = pd.read_csv(path+'X.csv')
    y = pd.read_csv(path+'y.csv')
    X.set_index('Buurten', inplace=True)
    y.set_index(['Unnamed: 0'], inplace=True)
    
    return X,y



st.cache_data()
def plot_coefficients_separate(coef, number, feature_names, labels, color='blue'):
    """
    @Author Maikel
    Display a barhplot for all feature and each class separate
    
    Parameters
    ----------
    coef: model.coef_
        Coefficitent values from the model.
    number: int
        Which classification class to display.
    feature_names: List
        List of names of features in the model.
    labels: List
        Class names.
    color: string
        Colour of the bars.
    
    Returns
    -------
    None
    
    Notes
    -----
    Classification model needs to provide coef_ for this function to work.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.5  # Width of each bar
    space_between_bars = 0.05  # Space between bars for different classes

    # Calculate the x-position for each bar, adding appropriate spacing
    x_pos = np.arange(len(feature_names))

    # Plot the horizontal bars for the current class
    ax.barh(x_pos, coef[number], height=bar_width, align='center', color=color)

    # Set the y-ticks and labels to the feature names
    ax.set_yticks(x_pos)
    ax.set_yticklabels(feature_names)

    ax.set_xlabel('Coefficient Value')  # Set the x-axis label
    ax.set_title('Coefficients for Class: {}'.format(labels[number]))  # Set the plot title

    plt.xticks(np.linspace(coef.min(), coef.max(), 7))  # Set the x-axis ticks so all the plots are the same and easy to read.
    st.pyplot(fig)  # Display the plot



def plot_coefficients(model, feature_names, class_names):
    """
    @Author Hubert
    Display a barhplot for each feature and class

    Parameters
    ----------
    coef : array-like, shape (n_classes, n_features)
        Coefficient values for each feature and class.
        Note: Classification model needs to provide coef_
    feature_names : list
        List of feature names.
    class_names: list
        List of class names.
    class_names : list
        List of class names.
        
    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2  # Width of each bar
    space_between_bars = 0.05  # Space between bars for different classes

    # Iterate over each class
    for i, class_name in enumerate(class_names):
        # Calculate the x-position for each bar, adding appropriate spacing
        x_pos = np.arange(len(feature_names)) + (bar_width + space_between_bars) * i
    
        # Plot the horizontal bars for the current class
        ax.barh(x_pos, model.coef_[i], height=bar_width, align='center', label=class_name)

    # Set the y-ticks and labels to the feature names
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)

    ax.legend()  # Add a legend to the plot
    ax.set_xlabel('Coefficient Value')  # Set the x-axis label
    ax.set_title('Coefficients for Each Class')  # Set the plot title

    plt.xticks(np.linspace(model.coef_.min(), model.coef_.max(), 7))
    st.pyplot(fig)  # Display the plot


st.cache_data()
def load_data_eda(path='pages/data/'):
    '''
    @Author Raphael
    Import the data for EDA
    
    
    
    Returns
    -------
    number_of_crime: DataFrame
    population: DataFrame
    educational_level: DataFrame
    income_level: DataFrame
    qol: DataFrame
    '''

    numbers_of_crime = pd.read_csv(path + 'numbers_of_crime.csv')
    population = pd.read_csv(path + 'population_2.csv')
    educational_level = pd.read_csv(path + 'educational_level_neighbourhood_of_Gemeente_Breda_processed.csv')
    income_level = pd.read_csv(path + 'income_level_neighbourhood_of_Gemeente_Breda_processed.csv')
    qol = pd.read_csv(path + 'QOL_of_Gemeente_Breda_processed.csv')

    return numbers_of_crime, population, educational_level, income_level, qol

def select_and_filter(select, data, key=''):
    
    year = data['Year'].unique()
    unique = data[select].unique()

    selected_year = st.selectbox(f'Select a year {key}', year)
    selected = st.selectbox(f'Select a {select}', unique)

    filter_year = data[data['Year'] == selected_year]
    final_filter = filter_year[filter_year[select] == selected]

    return final_filter