import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from functions import load_data_eda, select_and_filter


numbers_of_crime, population, educational_level, income_level, qol = load_data_eda()

st.set_page_config(page_title='Exploratory Data Analysis', layout='centered')

st.title(':green[Exploratory Data Analysis]')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Intro', 'Crime data', 'Population data', 'Educational level data', 'Income level data', 'Quality of life data'])

with tab1:
    st.subheader('*:violet[Intro]*')
    st.write("""This page is going to be dedicated on the exploratory data analysis, which was performed on the datasets, which were used for the purpose of the project.
            For the EDA were used 5 datasets - numbers of crime, population, educational level, income level and quality life as all of them are for Breda and focus on each
            neighbourhoods (Except Emer and Hazeldonk, because in some of the datasets there was not much data for them).""")

with tab2:
    st.subheader(':violet[Numbers of crime]')
    st.write("""This section is going to be focused on the numbers of crimes in Breda and each neighbourhood in it. The dataset can be found in the official website of the Dutch
                police. Originally, the table is made of 4 columns - 'Soort misdrijf' (type of crime), 'Perioden' (Period), 'Wijken en buurten' (District and neighbourhoods) and
                'Geregistreerde misdrijven (aantal)' (Registered crime). To make the data suitable for visualizations, it was checked for missing values and duplicates. The latter was not found, but for the 
                former there was some - null values were spotted all together with '.' values in the column for 'Registered crimes'. That either means that certain crimes are not occuring in the neighbourhood/
                area or just crimes were not reported. It was made the assumption that those crimes were not appearing, so the null values were kept. Furthermore, the 'Perioden', which contains months and
                year, was split into two so it can be plotted a chart, which shows the trend of different crimes through the years.""")

    selected_crime = st.selectbox('Select a crime', numbers_of_crime['Soort misdrijf'].unique())

    total_crime_rate = numbers_of_crime[numbers_of_crime['Soort misdrijf'] == selected_crime]

    selected_neighbourhood = st.selectbox('Select a neighbourhood', total_crime_rate['Wijken en buurten'].unique())

    filtered = total_crime_rate[total_crime_rate['Wijken en buurten'] == selected_neighbourhood]

    total_crime_year_per_year = filtered.groupby(['Soort misdrijf', 'Year'], as_index=False)['Geregistreerde misdrijven (aantal)'].mean().sort_values('Year')

    st.markdown('*:violet[Crime rate per year]*')
    st.line_chart(data=total_crime_year_per_year, x='Year', y='Geregistreerde misdrijven (aantal)')

    types_of_crime = numbers_of_crime.groupby('Soort misdrijf', as_index=False)['Geregistreerde misdrijven (aantal)'].mean().sort_values('Geregistreerde misdrijven (aantal)', ascending=False)
    types_of_crime = types_of_crime[types_of_crime['Soort misdrijf'] != 'Totaal misdrijven']
    types_of_crime_top_10 = types_of_crime.head(10)

    st.markdown('*:violet[The top 10 most common crimes (rate)]*')
    st.bar_chart(data=types_of_crime_top_10, x='Soort misdrijf', y='Geregistreerde misdrijven (aantal)')

    crime_per_neighbourhood = total_crime_rate.groupby('Wijken en buurten', as_index=False)['Geregistreerde misdrijven (aantal)'].mean().sort_values('Geregistreerde misdrijven (aantal)', ascending=False)
    crime_per_neighbourhood = crime_per_neighbourhood[crime_per_neighbourhood['Wijken en buurten'] != 'City']
    crime_per_neighbourhood_top_10 = crime_per_neighbourhood.head(10)

    st.markdown('*:violet[The top 10 neighborhoods with the highest crime rate]*')
    st.area_chart(data=crime_per_neighbourhood_top_10, x='Wijken en buurten', y='Geregistreerde misdrijven (aantal)')

with tab3:
    st.subheader('*:violet[Population]*')

    st.write("""The population contains data about the different age groups in each neighbourhood in Breda. There are five columns in the dataset - 'Buurten'(Neighbourhoods), 'Year', 'Age', 'Gender' 
                and 'value'. The 'Age' column is made of 5 age groups - < 30 jaar, 30-44 jaar, 45-64 jaaar, 65-74 jaar and >= 75 jaar. Missing values were spot only in the 'value' column as it was only one
                for one specific year for the neighbourhood Hazeldonk. Changes were not necessary as only one was done - the 'value' column was renamed to 'number_of_people', so it can be more clear what
                values it contains. """)

    population = population.rename({'value':'number of people'}, axis=1)

    year_and_gender_filter = select_and_filter('Gender', population)

    st.bar_chart(data=year_and_gender_filter, x='Age', y='number of people')

with tab4:
    st.subheader('*:violet[Educational level]*')

    st.write("""The educational level dataset gives information about the people with different level of education - low, medium, high. The table is made four columns - 'Buurten', 'Year', 'level_of_education'
                and 'value'. The 'value' column contained some missing values along with '?' values as the latter was removed from the data. Moreover, the same column was renamed to 'number_of_students',
                since the previous name did not give better idea of the content of it.""")

    buurten_year_filter = select_and_filter('Buurten', educational_level)

    st.bar_chart(buurten_year_filter, x='level_of_education', y='number_of_students')

    level_of_education_year_filter = select_and_filter('level_of_education', educational_level, ' ')

    st.bar_chart(level_of_education_year_filter, x='Buurten', y='number_of_students')

with tab5:
    st.subheader('*:violet[Income level]*')

    st.write("""The income level data provides information about the level of income in each neighbourhood of Breda in the period of 2009 to 2021. The income level is divided in two categories - high and low.
                It contains four columns - 'Buurten', 'Year', 'level_of_income' and 'value'. As in the educational level data, there were some '?' along with 'NaN' as the rows with the latter were removed.
                Also, the 'value' column was renamed to 'number_of_people'.""")

    Buurten_and_year_income = select_and_filter('Buurten', income_level)

    st.bar_chart(Buurten_and_year_income, x='level_of_income', y='number_of_people')

    income_level_buurtens = select_and_filter('level_of_income', income_level, ' ')

    st.area_chart(income_level_buurtens, x='Buurten', y='number_of_people')

with tab6:
    st.subheader('*:violet[Quality of life]*')

    st.write("""The dataset for quality of life in Breda gives information about the livability index in each neighbourhood of Breda for three years - 2014, 2018 and 2020. The dataset consists of 3 columns -
                'Buurten', 'Year' and 'value'. Like some of the other datasets, there were missing values in the 'value' column as the same one was renamed to 'livability_index'. Furthermore, the rows with
                '?' were removed from the table.""")

    years = qol['Year'].unique()

    selected_year = st.selectbox('Select a year', years)
    selected_sort = st.selectbox('Select sorting', ('Ascending', 'Descending'))

    filter_year = qol[qol['Year'] == selected_year]

    if selected_sort == 'Ascending':

        qol_sorted = filter_year.sort_values('livability_index')
        top_10 = qol_sorted.head(10)

    else:

        qol_sorted = filter_year.sort_values('livability_index', ascending=False)
        top_10 = qol_sorted.head(10)

    st.bar_chart(data=top_10, x='Buurten', y='livability_index')
