import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import colorsys



df = st.session_state.get('uploaded_data',None)

def Line_Break(width):
    line_code=f"""

        <hr style="border: none; height: 2px;width: {width}%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />


        """
    st.markdown(line_code,unsafe_allow_html=True)
    

def create_correlation_heatmap(numerical_dataset):
    # Get all numerical columns
    numerical_columns = numerical_dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Create callback to update session state
    def on_change():
        st.session_state.correlation_selected_columns = st.session_state.correlation_multiselect

    # Initialize the session state if it doesn't exist
    if 'correlation_selected_columns' not in st.session_state:
        st.session_state.correlation_selected_columns = numerical_columns

    # Create the multiselect with the callback
    selected_attributes = st.multiselect(
        "Select attributes for correlation analysis",
        options=numerical_columns,
        default=st.session_state.correlation_selected_columns,
        key='correlation_multiselect',
        on_change=on_change
    )

    if len(selected_attributes) < 2:
        st.warning("Please select at least two attributes for correlation analysis.")
        return

    correlation_matrix = numerical_dataset[selected_attributes].corr()

    # Define the available colorscales
    colorscales = {
        'Viridis': 'Viridis',
        'RdBu': 'RdBu',
        'Rainbow': 'Rainbow',
        'Plasma': 'Plasma',
        'Inferno': 'Inferno'
    }

    fig = go.Figure()

    for colorscale_name, colorscale in colorscales.items():
        heatmap = go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale=colorscale,
            zmin=-1, zmax=1,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={'size': 12},
            name=colorscale_name,
            visible=(colorscale_name == 'Viridis'),
            showscale=True,
            colorbar=dict(
                title='Correlation',
                #titleside='right',
                thickness=15,
                len=0.75,
            )
        )
        fig.add_trace(heatmap)

    # Define buttons
    buttons = []
    for idx, colorscale_name in enumerate(colorscales.keys()):
        visibility = [i == idx for i in range(len(colorscales))]
        buttons.append(
            dict(
                label=colorscale_name,
                method='update',
                args=[{'visible': visibility}],
            )
        )

    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title='Columns',
        yaxis_title='Columns',
        width=900,
        height=600,
        showlegend=True,
        updatemenus=[dict(
            type='buttons',
            showactive=True,
            buttons=buttons,
            x=1.25,
            y=0.7,
            xanchor='left',
            yanchor='middle',
            direction='down',
            pad={"r": 10, "t": 10},
            bgcolor='lightgray',
            font=dict(color='black'),
            bordercolor='gray',
        )]
    )

    # Add margin to prevent cutoff and adjust paper bgcolor
    fig.update_layout(
        margin=dict(r=150),
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )

    # Display the heatmap
    st.plotly_chart(fig, use_container_width=True)



    
def extract_numerical_columns(df):

    # Select columns with numerical data types
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    return numerical_columns



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

if 'uploaded_data' in st.session_state:
    st.title('Correlation Matrix')

    numrical_col=extract_numerical_columns(df)

    numrical_dataset=df[numrical_col]


    create_correlation_heatmap(numrical_dataset)

else:
    st.warning("Upload the dataset to view the plots")
