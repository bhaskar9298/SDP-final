import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import colorsys


def Line_Break(width):
        line_code=f"""

            <hr style="border: none; height: 2px;width: {width}%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />


            """
        st.markdown(line_code,unsafe_allow_html=True)


st.title('Upload the Dataset')
# Check if 'uploaded_file' is already in session state
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

Line_Break(100)

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

# If a new file is uploaded, process and store it
if uploaded_file is not None:
    # Store the uploaded file in session state
    st.session_state['uploaded_file'] = uploaded_file

    # Read the uploaded file based on file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    # Store the DataFrame in session state
    st.session_state['uploaded_data'] = df
    st.success("Dataset uploaded successfully!")
    
# If a dataset is already in session state, use it without requiring re-upload
elif st.session_state['uploaded_file'] is not None:
    uploaded_file = st.session_state['uploaded_file']

    # Check if the dataset is already stored in session state
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        st.success("Using previously uploaded dataset!")
    else:
        # Process the file if 'uploaded_data' is missing for some reason
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        # Store the DataFrame in session state
        st.session_state['uploaded_data'] = df
else:
    st.warning("Please upload a dataset to proceed.")

# Now 'df' contains the DataFrame for further use, whether from a fresh upload or session state.

# Display a sample of the dataset within the specified range
if 'uploaded_data' in st.session_state:
    # Ask user for the range to display the sample
    st.write("Specify the row range to display the sample of the dataset.")
    
    # Create two columns for start_row and end_row side by side
    col1, col2 = st.columns(2)
    
    with col1:
        start_row = st.number_input("Start row", min_value=0, max_value=len(df)-1, value=0)
    
    with col2:
        end_row = st.number_input("End row", min_value=0, max_value=len(df), value=min(5, len(df)))
    
    # Validate the range and display the sample
    if start_row < end_row:
        st.write(f"Displaying rows from {start_row} to {end_row}:")
        st.dataframe(df.iloc[start_row:end_row])
    else:
        st.warning("End row must be greater than the start row.")

