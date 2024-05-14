# Done by Amy
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_extras.app_logo import add_logo
import matplotlib.pyplot as plt
from PIL import Image
st.set_page_config(
    page_title="The Panic Room",
    page_icon="👋",
)
st.markdown (" # *Team 3 : The Panic Room* ")
image = Image.open("images/maxresdefault.jpg") 
image_size = (600,350)
image = image.resize(image_size)
st.image(image)
st.markdown ( "## :violet[The Crime Crystal Ball🔮]")

st.sidebar.success("Select a button.")

st.markdown("# :red[What is our project about?]")
st.markdown("""### *Our project aims to provide users with a comprehensive understanding of the factors that can influence crime rates in the municipality of Breda. The interactive app allows users to manipulate and explore various features, enabling them to assess whether a neighborhood has a high, medium, or low crime rate.Furthermore, our dashboard includes an analysis of the datasets used in this project, providing users with exploratory data analysis (EDA) insights. Additionally, we have incorporated ethical frameworks to ensure that our project adheres to professional standards and ethical considerations.By enhancing the professionalism of our project, we aim to deliver a robust and reliable tool for users to gain valuable insights into crime rates and make informed decisions.*] ###""")

from streamlit_extras.let_it_rain import rain
rain(
    emoji="✨",
    font_size=17,
    falling_speed=10,
    animation_length="infinite",
)

AI_canvas = Image.open("images/AI canvas.png")
st.image(AI_canvas)













