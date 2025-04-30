import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import colorsys

data = st.session_state.get('uploaded_data',None)

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

def Line_Break(width):
        line_code=f"""

            <hr style="border: none; height: 2px;width: {width}%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />


            """
        st.markdown(line_code,unsafe_allow_html=True)

def Line_Break_start(width):
        line_code=f"""

            <hr style="border: none; height: 2px;width: {width}%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />


                    """
        st.markdown(line_code,unsafe_allow_html=True)

def missing_values_count(data):
        # Calculate the count of missing values for each column
        missing_counts = data.isnull().sum()
        return missing_counts.sum()

def duplicate_Values_count(df):
    # Identify duplicate rows
    num_duplicates = df.duplicated(keep=False).sum()

    return num_duplicates

def plot_data_type_distribution(dataframe):
    # Initialize counters and lists for different data types
    float_count = 0
    int_count = 0
    object_count = 0
    numeric_columns = []
    categorical_columns = []

    # Count columns by data type
    for col in dataframe.columns:
        dtype = dataframe[col].dtype
        if dtype == "float64":
            float_count += 1
            numeric_columns.append(col)
        elif dtype == "int64":
            int_count += 1
            numeric_columns.append(col)
        elif dtype == "object":
            object_count += 1
            categorical_columns.append(col)

    # Calculate total number of columns
    total_columns = len(dataframe.columns)

    # Calculate numerical and categorical feature counts and percentages
    numerical_count = float_count + int_count
    numerical_percentage = (numerical_count / total_columns) * 100
    categorical_percentage = (object_count / total_columns) * 100

    # Data for Pie Chart
    labels = ['Numerical Features', 'Categorical Features']
    values = [numerical_percentage, categorical_percentage]

    # Prepare hover text with column names
    numeric_text = f"Numeric Columns ({numerical_count}):<br>" + "<br>".join(numeric_columns)
    categorical_text = f"Categorical Columns ({object_count}):<br>" + "<br>".join(categorical_columns)
    hover_text = [numeric_text, categorical_text]

    # Create Pie Chart with custom colors and hover information
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.4,
        marker=dict(colors=['#FFA500', '#1E90FF']),  # Custom colors: orange and blue
        textinfo='label+percent',  # Show labels and percentages on the chart
        hovertext=hover_text,
        hoverinfo='text',  # Show custom hover text
        textfont=dict(size=16),  # Increase text font size for readability
    )])

    # Update Layout
    fig.update_layout(
        title=dict(
            text='Distribution of Numerical and Categorical Features',
            x=0.5,  # Center the title
            xanchor='center',
            font=dict(size=22)  # Increase title font size
        ),
        annotations=[dict(
            text='Data Types', 
            x=0.5, y=0.5, 
            font_size=18, 
            showarrow=False
        )],
        legend=dict(
            font=dict(size=14),  # Increase legend font size
            orientation='h',  # Horizontal legend
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=80, b=50, l=50, r=50),  # Adjust margins for spacing
        width=700,
        height=500,
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)




def plot_data_type_distribution_barchart(dataframe):

    # Initialize counters
    float_count = 0
    int_count = 0
    object_count = 0

    # Count columns by data type
    for col in dataframe.columns:
        dtype = dataframe[col].dtype
        if dtype == "float64":
            float_count += 1
        elif dtype == "int64":
            int_count += 1
        elif dtype == "object":
            object_count += 1

    # Data for Bar Chart
    labels = ['Float', 'Integer', 'Object']
    counts = [float_count, int_count, object_count]

    # Create Bar Chart
    fig = go.Figure(data=[go.Bar(x=labels, y=counts, marker_color=['blue', 'green', 'red'])])

    # Update Layout
    fig.update_layout(
        title='Count of Each Data Type',
        xaxis_title='Data Type',
        yaxis_title='Count',
        width=600,
        height=400
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

def plot_missing_values_percentage(dataframe):
    # Calculate the percentage of missing values in each column
    missing_percentages = dataframe.isnull().mean() * 100

    # Filter out columns with 0% missing values
    missing_percentages = missing_percentages[missing_percentages > 0]

    # Data for Pie Chart
    labels = missing_percentages.index
    values = missing_percentages.values

    # Create Pie Chart
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.5,
        marker=dict(colors=['orange', 'blue', 'red', 'green', 'purple'])  # Customize colors
    )])

    # Update Layout
    fig.update_layout(
        title='Percentage of Missing Values in Each Column',
        annotations=[dict(text='Missing Data', x=0.5, y=0.5, font_size=15, showarrow=False)],
        legend={'font': {'size': 14}},  # Legend font size
        margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for better spacing
        width=600,
        height=400
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

def plot_missing_values_count_barchart(dataframe):

    # Calculate the count of missing values for each column
    missing_counts = dataframe.isnull().sum()

    # Filter out columns with zero missing values
    missing_counts = missing_counts[missing_counts > 0]

    # Data for Bar Chart
    labels = missing_counts.index
    counts = missing_counts.values

    # Only plot if there are columns with missing values
    if len(labels) > 0:
        # Generate a dynamic color palette based on the number of columns
        num_columns = len(labels)
        colors = px.colors.qualitative.Plotly * (num_columns // len(px.colors.qualitative.Plotly) + 1)  # Ensure enough colors

        # Create Bar Chart with dynamic colors
        fig = go.Figure(data=[go.Bar(x=labels, y=counts, marker_color=colors[:num_columns])])

        # Update Layout
        fig.update_layout(
            title='Count of Missing Values in Each Column',
            xaxis_title='Columns',
            yaxis_title='Count of Missing Values',
            xaxis=dict(
                tickfont=dict(size=14),
                titlefont=dict(size=16)
            ),
            yaxis=dict(
                tickfont=dict(size=14),
                titlefont=dict(size=16)
            ),
            width=500,
            height=600
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)
    else:
        st.write("No missing values in the dataset!")
        
        
def plot_duplicate_vs_unique_pie_chart(df):
    # Identify duplicate rows
    num_duplicates = df.duplicated(keep=False).sum()
    num_unique = len(df) - num_duplicates

    # Prepare data for pie chart
    data_summary = pd.DataFrame({
        'Type': ['Unique Rows', 'Duplicate Rows'],
        'Count': [num_unique, num_duplicates]
    })

    # Create a pie chart with Plotly
    fig = px.pie(
        data_summary,
        names='Type',
        values='Count',
        title='Percentage of Unique vs Duplicate Rows',
        color_discrete_sequence=['blue', 'yellow'],
        hole=0.4  # Donut style
    )

    # Display the pie chart in Streamlit
    st.plotly_chart(fig)



def plot_duplicate_vs_unique_bar_chart(df):
    # Identify duplicate rows
    num_duplicates = df.duplicated(keep=False).sum()
    num_unique = len(df) - num_duplicates

    # Prepare data for bar chart
    data_summary = pd.DataFrame({
        'Type': ['Unique Rows', 'Duplicate Rows'],
        'Count': [num_unique, num_duplicates]
    })

    # Create a bar chart with Plotly
    fig = px.bar(
        data_summary,
        x='Type',
        y='Count',
        title='Count of Unique vs Duplicate Rows',
        color='Type',
        color_discrete_sequence=['orange', '#EF553B']
    )

    # Display the bar chart in Streamlit
    st.plotly_chart(fig)


if 'uploaded_data' in st.session_state:
    
    st.title('Data Analytics Dashboard')
    rows=data.shape[0]
    Features=data.shape[1]

    # Get the data types of each column
    data_types = data.dtypes
    # Convert to a set to get unique data types
    unique_data_types = set(data_types)
    null_values = data.isnull().sum()
    null_values=null_values.sum()
    # Get the number of unique data types
    num_unique_data_types = len(unique_data_types)
    # Inject custom CSS
    st.markdown(
        """
        <style>
        /* Change the background color of the tab container */
        div[data-baseweb="tab-list"] {
            background-color: #00BCD4;
            padding: 5px;
            border-radius: 20px;
        }
        /* Change the color of the selected tab */
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #0008ff;
        color: white;
        border-radius: 20px;
        padding: 10px;
        border: none;
        }
        /* Change the color of non-selected tabs */
        div[data-baseweb="tab-list"] button {
            # background-color: #ffd54fdb;
            # color: black;
            border-radius: 10px;
            padding: 10px;
            border: none;
            margin: 0 5px;
            color: #fff;
            background-color: #007bff;
            border-color: #007bff;
        }
        /* Change the hover color of non-selected tabs */
        div[data-baseweb="tab-list"] button:hover {
            background-color: #ffcc00;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if missing_values_count(data)  > 1 and duplicate_Values_count(data) >1:
        Line_Break(100)
        tab1, tab2,tab3 = st.tabs(["DataTypes Analysis", "Missing Values Analysis","Duplicate Values Analysis"])



        with tab1:
            Line_Break(100)

            st.subheader('DataTypes Analysis')
            graph1, graph2 = st.columns([2, 1])
            with graph1:
                plot_data_type_distribution(data)
            with graph2:
                plot_data_type_distribution_barchart(data)
        Line_Break(100)
        with tab2:
            Line_Break(100)

            st.subheader('Missing Values Analysis')
            graph3, graph4 = st.columns([2, 1])
            with graph3:
                plot_missing_values_percentage(data)
            with graph4:
                plot_missing_values_count_barchart(data)
        with tab3:
            Line_Break(100)

            st.subheader('Duplicate Values Analysis')
            graph3, graph4 = st.columns([2, 1])
            with graph3:
                plot_duplicate_vs_unique_pie_chart(data)
            with graph4:
                plot_duplicate_vs_unique_bar_chart(data)
        # Function to create boxplots

    elif  missing_values_count(data)  > 1:
        Line_Break(100)
        tab1, tab2 = st.tabs(["DataTypes Analysis", "Missing Values Analysis"])
        with tab1:
            Line_Break(100)
            st.subheader('DataTypes Analysis')
            graph1, graph2 = st.columns([2, 1])
            with graph1:
                plot_data_type_distribution(data)
            with graph2:
                plot_data_type_distribution_barchart(data)
        Line_Break(100)
        with tab2:
            Line_Break(100)

            st.subheader('Missing Values Analysis')
            graph3, graph4 = st.columns([2, 1])
            with graph3:
                plot_missing_values_percentage(data)
            with graph4:
                plot_missing_values_count_barchart(data)
    else:
        Line_Break(100)
        # Create a single tab
        tab1 = st.tabs(["DataTypes Analysis"])[0]  # Access the first tab
        

        with tab1:
            Line_Break(100)
            st.subheader('DataTypes Analysis')
            
            graph1, graph2 = st.columns([2, 1])

            with graph1:
                plot_data_type_distribution(data)

            with graph2:
                plot_data_type_distribution_barchart(data)

        Line_Break(100)
        # Function to create boxplots
        def create_boxplots(data, columns):
            st.write("Here you can display a boxplot or any other analysis for", columns[0])
            # Example plot (you can replace this with actual boxplot code)
            st.line_chart(data[columns[0]])
            Line_Break(100)

else:
    st.warning("Upload the dataset to view the dataset analysis")