import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import colorsys


# Set page config first - MUST be the first Streamlit command
st.set_page_config(
    page_icon="assets/logo.png",
    page_title="EDA TOOL",
    layout="wide",  # This ensures wide layout
    initial_sidebar_state="expanded"  # Makes sure sidebar is expanded by default
)

def add_logo(logo_path):
    """Read and return a resized logo"""
    with open(logo_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    
    logo_html = f'''
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url("data:image/png;base64,{data}");
                background-repeat: no-repeat;
                background-position: 60px 0px;
                background-size: 180px auto;
                padding-top: 140px;
            }}
            
            
        </style>
    '''
    
    st.markdown(logo_html, unsafe_allow_html=True)

# Add the logo
add_logo("assets/AI-POWERED-EDA-TOOL.png")

# Define your pages
about_page = st.Page(
    page="views/about.py",
    title="About",
    icon=":material/home:",
    default=True
)

eda_1_page = st.Page(
    page="views/upload.py",
    title = "Upload dataset",
    icon=":material/upload_file:"
)

eda_2_page = st.Page(
    page="views/datatype.py",
    title="Dataset Analysis",
    icon=":material/bar_chart:"
)

eda_3_page = st.Page(
    page = "views/plots.py",
    title = "Plots",
    icon=":material/query_stats:"
)

eda_4_page = st.Page(
    page="views/matrix.py",
    title="Correlation Matrix",
    #icon=":material/grid_view:"
)

eda_5_page = st.Page(
    page="views/data_qa.py",
    title="ChatBot",
    #icon="material/question_answer:"
)

# ai_eda_page = st.Page(
#     page="views/ai_eda.py",
#     title="AI Report",
#     icon=":material/insights:"
# )
# REMOVED: Duplicate Data QA page that was causing the conflict
# Now we only have one reference to the data_qa.py file

# Set up navigation
pg = st.navigation(
    {
        "Info":[about_page],
        "EDA": [eda_1_page, eda_2_page, eda_3_page, eda_4_page, eda_5_page],
    }
)

pg.run()

# Your existing CSS with modifications for wide layout
page_bg_img = """
<style>
[data-testid="stSidebar"] > div:first-child {
    background-repeat: no-repeat;
    background-attachment: fixed;
    background: rgb(18 18 18 / 0%);
}

.st-emotion-cache-1gv3huu {
    position: relative;
    top: 2px;
    background-color: #000;
    z-index: 999991;
    min-width: 244px;
    max-width: 300px;  /* Control sidebar width */
    transform: none;
    transition: transform 300ms, min-width 300ms, max-width 300ms;
}

.st-emotion-cache-1jicfl2 {
    width: 100%;
    padding: 1rem;  /* Reduced padding */
    min-width: auto;
    max-width: none;  /* Allow full width */
}

/* Ensure content spans full width */
.stApp {
    max-width: 100%;
}


/* Rest of your existing styles... */

.st-emotion-cache-4uzi61 {
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background: rgb(240 242 246);
    box-shadow: 0 5px 8px #6c757d;
    width: 100%;  /* Ensure full width */
}

/* Additional styles to ensure wide layout */
.row-widget {
    width: 100%;
}

.stDataFrame {
    width: 100%;
}

/* Make plots take full width */
.js-plotly-plot {
    width: 100%;
}

/* Rest of your existing styles... */
</style>
"""

# Apply CSS styling
st.markdown(page_bg_img, unsafe_allow_html=True)