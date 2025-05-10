import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import colorsys



st.markdown("""
    <style>
    .title {
        font-size: 2.5rem;
        color: #4B89DC;
        text-align: center;
        font-weight: bold;
    }
    .subheader {
        font-size: 1.5rem;
        color: #1F77B4;
        margin-top: 20px;
        font-weight: bold;
    }
    .description, .features, .audience {
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 20px;
        text-align: justify;
    }
    .key-features ul {
        list-style-type: disc;
        margin-left: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def show_description():
    st.markdown('<div class="title">About This App</div>', unsafe_allow_html=True)
    
    # Description of the app
    st.markdown('<div class="subheader">Welcome to the EDA Web App</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="description">
    This web application is designed to help you perform interactive Exploratory Data Analysis (EDA) with ease. 
    With a simple and intuitive interface, users can upload datasets, create dynamic visualizations, and explore key data trends effortlessly.
    </div>
    """, unsafe_allow_html=True)
    
    # Features of the app
    st.markdown('<div class="subheader">Key Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="features key-features">
    <ul>
        <li><b>Upload Your Dataset:</b> Easily upload datasets in CSV format and get started with analysis.</li>
        <li><b>Dynamic Visualizations:</b> Create histograms, scatter plots, box plots, and heatmaps with customizable colors.</li>
        <li><b>Instant Results:</b> View results in real-time as you interact with the app.</li>
        <li><b>User-Friendly Interface:</b> Designed for both beginners and advanced users in data analysis.</li>
        <li><b>Question Answering  ChatBot: Engage with the RAG-enabled Q&A chatbot for information.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Target audience
    st.markdown('<div class="subheader">Who is this app for?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="audience">
    This tool is ideal for:
    <ul>
        <li><b>Data Analysts:</b> Seeking to quickly explore data trends and create insightful visualizations.</li>
        <li><b>Students & Researchers:</b> Working on data-centric projects or academic research.</li>
        <li><b>Business Professionals:</b> Looking for an easy, no-code solution for visualizing datasets.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="description">
    Whether you are looking to explore relationships between variables, analyze data distribution, or visualize correlations, this app 
    offers a fast and intuitive way to perform in-depth data analysis.
    </div>
    """, unsafe_allow_html=True)
    
page_bg_img = f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
    background-repeat: no-repeat;
    background-attachment: fixed;
    background: rgb(18 18 18 / 0%);
}}

.st-emotion-cache-1gv3huu {{
    position: relative;
    top: 2px;
    background-color: #000;
    z-index: 999991;
    min-width: 244px;
    max-width: 550px;
    transform: none;
    transition: transform 300ms, min-width 300ms, max-width 300ms;
}}

.st-emotion-cache-1jicfl2 {{
    width: 100%;
    padding: 4rem 1rem 4rem;
    min-width: auto;
    max-width: initial;

}}


.st-emotion-cache-4uzi61 {{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background: rgb(240 242 246);
    box-shadow: 0 5px 8px #6c757d;
}}

.st-emotion-cache-1vt4y43 {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: #ffc107;
    border: 1px solid rgba(49, 51, 63, 0.2);
}}

.st-emotion-cache-qcpnpn {{
    border: 1px solid rgb(163, 168, 184);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background-color: rgb(38, 39, 48);
    MARGIN-TOP: 9PX;
    box-shadow: 0 5px 8px #6c757d;
}}


.st-emotion-cache-15hul6a {{
    user-select: none;
    background-color: #ffc107;
    border: 1px solid rgba(250, 250, 250, 0.2);
    
}}

.st-emotion-cache-1hskohh {{
    margin: 0px;
    padding-right: 2.75rem;
    color: rgb(250, 250, 250);
    border-radius: 0.5rem;
    background: #000;
}}

.st-emotion-cache-12pd2es {{
    margin: 0px;
    padding-right: 2.75rem;
    color: #f0f2f6;
    border-radius: 0.5rem;
    background: #000;
}}

p, ol, ul, dl {{
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 1rem;
    font-weight: 400;
    color: whitesmoke;
}}

.st-emotion-cache-1v7f65g .e1b2p2ww15 {{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    background: #212121;
    color: white;
}}

.st-emotion-cache-1aehpvj {{
    color: #f5deb3ab;
    font-size: 12px;
    line-height: 1.25;
}}

.st-emotion-cache-1ny7cjd {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: #FFA000;
    border: 1px solid rgba(49, 51, 63, 0.2);
}}

.st-cg {{
    caret-color: rgb(23 24 27);
 
    background: #bdbdbdc4;

}}

.st-emotion-cache-1jicfl2 {{
    width: 100%;
    padding: 2rem 1rem 4rem;
    min-width: auto;
    max-width: initial;
}}

.st-emotion-cache-ocqkz7 {{
    display: flex;
    flex-wrap: wrap;
    -webkit-box-flex: 1;
    flex-grow: 1;
    -webkit-box-align: stretch;
    align-items: stretch;
    gap: 1rem;
    padding: 20px;
}}

# .st-emotion-cache-ue6h4q {{
#     font-size: 14px;
#     color: rgb(49, 51, 63);
#     display: flex;
#     visibility: visible;
#     margin-bottom: 0.25rem;
#     height: auto;
#     min-height: 1.5rem;
#     vertical-align: middle;
#     flex-direction: row;
#     -webkit-box-align: center;
#     align-items: center;
#     display: none;
# }}


</style>
"""

# Apply CSS styling to the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)

show_description()

st.session_state['show_description'] = show_description

