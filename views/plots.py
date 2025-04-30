import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import colorsys

data = st.session_state.get('uploaded_data',None)
def Line_Break(width):
    line_code=f"""

        <hr style="border: none; height: 2px;width: {width}%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />


        """
    st.markdown(line_code,unsafe_allow_html=True)
        
        
def styled_paragraph(content, color="#37474F", font_size="14px"):


    # Define the CSS style block with dynamic values
    css_style = f"""
    <style>
        .custom-paragraph {{
            color: {color};
            font-size: {font_size};
        
        }}
    </style>
    """

    # Insert the CSS into the Streamlit app
    st.markdown(css_style, unsafe_allow_html=True)

    # Define the HTML content with the custom class
    html_content = f"""
    <p class="custom-paragraph">
        {content}
    </p>
    """

    # Display the HTML content in Streamlit
    st.markdown(html_content, unsafe_allow_html=True)        
        

# def create_correlation_heatmap(numrical_dataset, col_name):
#     st.write("##### Correlation Heatmap")
    
#     # Get all numerical columns
#     numerical_columns = numrical_dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
#     selected_attributes = st.multiselect(
#         "Select attributes for correlation analysis",
#         options=numerical_columns,
#         default=numerical_columns,
#         key=f"attribute_selector_{col_name}"
#     )
    
#     if len(selected_attributes) < 2:
#         st.warning("Please select at least two attributes for correlation analysis.")
#         return
        
#     correlation_matrix = numrical_dataset[selected_attributes].corr()
    
#     colorscales = {
#         'Viridis': 'Viridis',
#         'RdBu': 'RdBu',
#         'Rainbow': 'Rainbow',
#         'Plasma': 'Plasma',
#         'Inferno': 'Inferno'
#     }
    
#     fig = go.Figure()
    
#     for colorscale_name, colorscale in colorscales.items():
#         heatmap = go.Heatmap(
#             z=correlation_matrix.values,
#             x=correlation_matrix.columns,
#             y=correlation_matrix.index,
#             colorscale=colorscale,
#             zmin=-1, zmax=1,
#             text=correlation_matrix.values,
#             texttemplate='%{text:.2f}',
#             textfont={'size': 12},
#             name=colorscale_name,
#             visible=(colorscale_name == 'Viridis'),
#             showscale=True,
#             colorbar=dict(
#                 title='Correlation',
#                 titleside='right',
#                 thickness=15,
#                 len=0.75,
#             )
#         )
#         fig.add_trace(heatmap)
    
#     buttons = []
#     for idx, colorscale_name in enumerate(colorscales.keys()):
#         visibility = [i == idx for i in range(len(colorscales))]
#         buttons.append(
#             dict(
#                 label=colorscale_name,
#                 method='update',
#                 args=[{'visible': visibility}],
#                 visible=True
#             )
#         )
    
#     fig.update_layout(
#         title="Correlation Heatmap",
#         xaxis_title='Columns',
#         yaxis_title='Columns',
#         width=900,
#         height=600,
#         showlegend=True,
#         legend=dict(
#             title='Color Scales',
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=1.02,
#             bgcolor='rgba(255, 255, 255, 0.8)',
#             bordercolor='rgba(0, 0, 0, 0.2)',
#             borderwidth=1
#         ),
#         updatemenus=[dict(
#             type='buttons',
#             showactive=True,
#             buttons=buttons,
#             x=1.25,
#             y=0.7,
#             xanchor='left',
#             yanchor='middle',
#             direction='down',
#             pad={"r": 10, "t": 10},
#             # bgcolor='white',
#             bordercolor='white',
#             borderwidth=2,
#             font=dict(
#                 size=12,
#                 color='white'
#             ),
#             active=0
#         )]
#     )
    
#     # Add margin to prevent cutoff
#     fig.update_layout(margin=dict(r=150))
    
#     # Display the heatmap
#     st.plotly_chart(fig, use_container_width=True)
#     Line_Break(100)

            
def plot_class_distribution_pie_chart(df, column_name):
    # Calculate percentage of each class
    class_counts = df[column_name].value_counts(normalize=True) * 100
    class_percentages = class_counts.reset_index()
    class_percentages.columns = ['Class', 'Percentage']
    
    # Define custom colors (you can change these to any colors you like)
    custom_colors = ['orange', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2']

    # Create a pie chart with Plotly Express
    fig = px.pie(class_percentages, names='Class', values='Percentage', 
                 title='Class Distribution Percentage',
                 color_discrete_sequence=custom_colors)
    
    # Display the pie chart in Streamlit
    st.plotly_chart(fig)



def plot_class_distribution_bar_chart(df, column_name):
    # Calculate count of each class
    class_counts = df[column_name].value_counts()
    class_counts_df = class_counts.reset_index()
    class_counts_df.columns = ['Class', 'Count']
    
    # Create a bar chart with Plotly Express, assigning different colors to each class
    fig = px.bar(class_counts_df, x='Class', y='Count', 
                 title='Class Distribution Count', 
                 labels={'Class': 'Class', 'Count': 'Count'},
                 text='Count',
                 color='Class',  # Color by 'Class'
                 color_discrete_sequence=px.colors.qualitative.Plotly)  # Use Plotly's color sequence
    
    # Display the bar chart in Streamlit
    st.plotly_chart(fig)


def check_datatype(dataframe,colmun):
    dtype = dataframe[colmun].dtype
    return dtype

def categorical_variable(df, column_name):
    Line_Break(100)
    col_name = column_name[0]
    st.subheader(f'{col_name} Feature Analysis')

    graph1, graph2 = st.columns([2, 1])

    with graph1:
        plot_class_distribution_pie_chart(df, column_name)

    with graph2:
        plot_class_distribution_bar_chart(df, column_name)
        
    Line_Break(100)
    
    # Options for graph type
    graph_options = ["Line Graph", "Box Plot", "Pivot Chart"]
    graph_type = st.selectbox("Select Graph Type", graph_options, key=f"graph_type_{col_name}")

    # Line Graph
    if graph_type == "Line Graph":
        st.write(f"##### Line Graph of {col_name}")

        # Select y-axis from numerical columns
        y_options = [col for col in df.columns if col != col_name and df[col].dtype in ['int64', 'float64']]

        if not y_options:
            st.warning("There are no numerical columns available for the y-axis. Please select a different column or graph type.")
        else:
            # Create columns for controls
            col1, col2, col3 = st.columns(3)

            with col1:
                y_axis = st.selectbox("Select Y-axis", y_options, key=f"y_axis_{col_name}")

            with col2:
                # Add aggregation function selection
                agg_function = st.selectbox(
                    "Select Aggregation Method",
                    ["Mean", "Sum", "Count", "Min", "Max", "Median"],
                    key=f"agg_function_{col_name}"
                )

            with col3:
                line_color = st.color_picker("Select color for Line Graph", "#2ca02c", key=f"line_color_{col_name}")

            # Map aggregation function to pandas methods
            agg_map = {
                "Mean": "mean", "Sum": "sum", "Count": "count",
                "Min": "min", "Max": "max", "Median": "median"
            }
            grouped_df = df.groupby(col_name).agg({y_axis: agg_map[agg_function]}).reset_index()
            
            # Create the line graph using Plotly
            line_fig = px.line(
                grouped_df, 
                x=col_name, 
                y=y_axis,
                title=f"Line Graph of {y_axis} by {col_name} ({agg_function})",
                color_discrete_sequence=[line_color],
                width=800, 
                height=600
            )
            line_fig.update_traces(
                hovertemplate=f"{col_name}: %{{x}}<br>{y_axis} ({agg_function}): %{{y}}<extra></extra>"
            )

            # Option to show markers
            show_markers = st.checkbox("Show markers", key=f"show_markers_{col_name}")
            if show_markers:
                line_fig.update_traces(mode='lines+markers')

            line_fig.update_layout(
                xaxis_title=col_name,
                yaxis_title=f"{y_axis} ({agg_function})",
                showlegend=False,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#31333F',
                    zerolinecolor='rgba(128, 128, 128, 0.2)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#31333F',
                    zerolinecolor='rgba(128, 128, 128, 0.2)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(line_fig)

    # Box Plot
    elif graph_type == "Box Plot":
        st.write(f"##### Box Plot for {col_name}")
        
        # Create columns for controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Determine if the current column is categorical
            is_categorical = not pd.api.types.is_numeric_dtype(df[col_name])
            
            if is_categorical:
                # If categorical, allow selecting a numerical variable for the box plot values
                numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if numerical_cols:
                    value_col = st.selectbox(
                        "Select numerical variable for values", 
                        numerical_cols,
                        key=f"box_value_select_{col_name}_cat"
                    )
                else:
                    st.warning("No numerical columns available for box plot")
                    value_col = None
            else:
                # If the current column is numerical, allow selecting a categorical variable for grouping
                categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
                if categorical_cols:
                    group_by = st.selectbox(
                        "Group by (categorical variable)", 
                        categorical_cols,
                        key=f"box_group_select_{col_name}_num"
                    )
                else:
                    st.warning("No categorical columns available for grouping")
                    group_by = None
        
        with col2:
            # Option to show all points
            show_points = st.selectbox(
                "Show individual points", 
                ["No Points", "Outliers Only", "All Points"],
                key=f"box_points_select_{col_name}_cat"
            )
            
            points_map = {
                "All Points": "all",
                "Outliers Only": "outliers",
                "No Points": False
            }
        
        # with col3:
        #     box_color = st.color_picker(
        #         "Select default color", 
        #         "#1f77b4", 
        #         key=f"box_default_color_{col_name}_cat"
        #     )
        
        # Create the box plot based on selected options
        if (is_categorical and value_col is not None) or (not is_categorical and group_by is not None):
            if is_categorical:
                box_fig = px.box(
                    df,
                    x=col_name,
                    y=value_col,
                    title=f"Box Plot of {value_col} by {col_name}",
                    color=col_name,
                    points=points_map[show_points],
                    width=800,
                    height=600,
                    template="plotly_dark"
                )
            else:
                box_fig = px.box(
                    df,
                    x=group_by,
                    y=col_name,
                    title=f"Box Plot of {col_name} by {group_by}",
                    color=group_by,
                    points=points_map[show_points],
                    width=800,
                    height=600,
                    template="plotly_dark"
                )
            
            # Update layout with dark theme
            box_fig.update_layout(
                font=dict(color='white'),
                showlegend=True,
                legend_title_text="Groups",
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#31333F',
                    zerolinecolor='rgba(128, 128, 128, 0.2)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#31333F',
                    zerolinecolor='rgba(128, 128, 128, 0.2)'
                )
            )
            
            # Add statistical annotations if requested
            show_stats = st.checkbox("Show statistical annotations", key=f"box_stats_check_{col_name}_cat")
            if show_stats:
                if is_categorical:
                    stats = df.groupby(col_name)[value_col].describe()
                else:
                    stats = df.groupby(group_by)[col_name].describe()
                
                annotations = []
                for idx, row in stats.iterrows():
                    annotation_text = (
                        f"Group: {idx}<br>"
                        f"Mean: {row['mean']:.2f}<br>"
                        f"Median: {row['50%']:.2f}<br>"
                        f"Std: {row['std']:.2f}"
                    )
                    
                    annotations.append(
                        dict(
                            x=idx,
                            y=row['max'],
                            text=annotation_text,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='white',
                            font=dict(size=10, color='white'),
                            bgcolor='rgba(0,0,0,0.7)',
                            bordercolor='white',
                            borderwidth=1,
                            borderpad=4,
                            align='left'
                        )
                    )
                
                box_fig.update_layout(annotations=annotations)
            
            st.plotly_chart(box_fig)
        else:
            st.warning("Please select appropriate numerical and categorical columns for the box plot.")
            
            
    elif graph_type == "Pivot Chart":
        st.write(f"##### Pivot Chart for {col_name}")
        
        # Select values to aggregate (y-axis)
        numeric_columns = list(df.select_dtypes(include=['float64', 'int64']).columns)
        
        if not numeric_columns:
            st.warning("There are no numerical columns available for the y-axis. Please select a different column or graph type.")
        else:
            # Create columns for controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                values_col = st.selectbox(
                    "Select Y-Axis Values", 
                    numeric_columns,
                    key=f"pivot_values_{col_name}"
                )
            
            with col2:
                # Select columns for grouping
                column_options = ["None"] + [col for col in df.columns if col not in [col_name, values_col]]
                columns_col = st.selectbox(
                    "Select Column Groups", 
                    column_options,
                    key=f"pivot_columns_{col_name}"
                )
            
            with col3:
                # Select aggregation function
                agg_functions = ["mean", "sum", "count", "min", "max"]
                agg_function = st.selectbox(
                    "Select Aggregation", 
                    agg_functions,
                    key=f"agg_function_{col_name}"
                )

            try:
                # Create pivot table
                if columns_col == "None":
                    pivot_data = df.groupby(col_name)[values_col].agg(agg_function).reset_index()
                    
                    # Ensure the data is properly formatted
                    pivot_fig = px.bar(
                        pivot_data,
                        x=col_name,
                        y=values_col,
                        title=f"Pivot Chart: {agg_function.capitalize()} of {values_col} by {col_name}",
                        template="plotly_dark"
                    )
                else:
                    # Create pivot table with error handling
                    pivot_data = df.pivot_table(
                        values=values_col,
                        index=col_name,
                        columns=columns_col,
                        aggfunc=agg_function,
                        fill_value=0  # Fill NaN values with 0
                    ).reset_index()
                    
                    # Melt the pivot table to long format for proper plotting
                    melted_data = pivot_data.melt(
                        id_vars=[col_name],
                        value_vars=pivot_data.columns[1:],
                        var_name=columns_col,
                        value_name=values_col
                    )
                    
                    pivot_fig = px.bar(
                        melted_data,
                        x=col_name,
                        y=values_col,
                        color=columns_col,
                        title=f"Pivot Chart: {agg_function.capitalize()} of {values_col} by {col_name} and {columns_col}",
                        template="plotly_dark",
                        barmode='group'
                    )

                # Update layout with dark theme
                pivot_fig.update_layout(
                    xaxis_title=col_name,
                    yaxis_title=f"{agg_function.capitalize()} of {values_col}",
                    font=dict(color='white'),
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='#31333F',
                        zerolinecolor='rgba(128, 128, 128, 0.2)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='#31333F',
                        zerolinecolor='rgba(128, 128, 128, 0.2)'
                    ),
                    showlegend=True,
                    legend_title_text=columns_col if columns_col != "None" else "",
                    width=800,
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                # Display the chart
                st.plotly_chart(pivot_fig)
                
            except Exception as e:
                st.error(f"Error creating pivot chart: {str(e)}")
                st.write("Please try different columns or aggregation functions.")


    
    

    
    
    
    
    
def extract_numerical_columns(df):

    # Select columns with numerical data types
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    return numerical_columns




def numarical_Features(df, column_name):
    col_name = column_name[0]

    st.title(f"Analysis for {col_name}")

    graph1, graph2 = st.columns([2, 1])
    
    # Initialize session state for graph type if it doesn't exist
    if f"graph_type_{col_name}" not in st.session_state:
        st.session_state[f"graph_type_{col_name}"] = "Scatter Plot"

    with graph1:
        st.write(f"##### Distribution Of {col_name}")
        hist_color = st.color_picker("Select color for Histogram", "#1f77b4", key=f"hist_color_{col_name}")
        hist_fig = px.histogram(df, x=col_name, nbins=30, title=f"Histogram of {col_name}", color_discrete_sequence=[hist_color])
        
        # Update layout for consistency with dark theme
        hist_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            )
        )
        
        st.plotly_chart(hist_fig)


    with graph2:
        st.write(f"##### Outliers in {col_name}")
        box_color = st.color_picker("Select color for Box Plot", "#ff7f0e", key=f"box_color_{col_name}")
        box_fig = px.box(df, y=column_name, title=f"Box Plot of {col_name}", color_discrete_sequence=[box_color])
        st.plotly_chart(box_fig)

    Line_Break(100)

    # Define options list
    graph_options = ["Scatter Plot","Line Graph","Box Plot","Pivot Chart"]
    
    # Get the index of the current selection from session state
    current_index = graph_options.index(st.session_state[f"graph_type_{col_name}"])

    # Use selectbox with index
    graph_type = st.selectbox(
        "Select Graph Type",
        graph_options,
        index=current_index,
        key=f"graph_type_{col_name}"
    )


    if graph_type == "Line Graph":
        st.write(f"##### Line Graph of {col_name}")

        # Create columns for controls
        col1, col2, col3 = st.columns(3)

        with col1:
            # Exclude current column from selection for X-axis
            x_options = [col for col in df.columns if col != col_name]
            x_axis = st.selectbox(
                "Select X-axis", 
                x_options, 
            )

        with col2:
            # Add aggregation function selection
            agg_function = st.selectbox(
                "Select Aggregation Method",
                ["None", "Mean", "Sum", "Count", "Min", "Max", "Median"]
                
            )

        with col3:
            line_color = st.color_picker(
                "Select color for Line Graph", 
                "#2ca02c"
            )

        # Process data based on aggregation selection
        if agg_function != "None":
            # Convert aggregation function name to pandas function
            agg_map = {
                "Mean": "mean",
                "Sum": "sum",
                "Count": "count",
                "Min": "min",
                "Max": "max",
                "Median": "median"
            }

            # Group by x_axis and aggregate y values
            grouped_df = df.groupby(x_axis).agg({col_name: agg_map[agg_function]}).reset_index()

            # Create line graph with aggregated data
            line_fig = px.line(
                grouped_df, 
                x=x_axis, 
                y=col_name,
                title=f"Line Graph of {col_name} ({agg_function})",
                color_discrete_sequence=[line_color],
                width=800, 
                height=600
            )

            # Add hover data to show aggregated values
            line_fig.update_traces(
                hovertemplate=f"{x_axis}: %{{x}}<br>{col_name} ({agg_function}): %{{y}}<extra></extra>"
            )

        else:
            # Create line graph with original data
            line_fig = px.line(
                df, 
                x=x_axis, 
                y=col_name,
                title=f"Line Graph of {col_name}",
                color_discrete_sequence=[line_color],
                width=800, 
                height=600
            )

        # Add option for showing/hiding markers
        show_markers = st.checkbox("Show markers")
        if show_markers:
            line_fig.update_traces(mode='lines+markers')

        # Customize layout
        line_fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=col_name,
            showlegend=False,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#31333F'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#31333F')
        )

        st.plotly_chart(line_fig)

    
    elif graph_type == "Scatter Plot":
        st.write(f"##### Scatter Plot for {col_name}")
        
        # Create three columns for controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Exclude current column from selection for Y-axis
            y_options = [col for col in df.columns if col != col_name]
            y_axis = st.selectbox(
                "Select Y-axis", 
                y_options,
                key=f"scatter_y_{col_name}"
            )
        
        with col2:
            hue_options = ["None"] + [col for col in df.columns if col != col_name]
            hue_column = st.selectbox(
                "Select a column for hue (optional)", 
                hue_options,
                key=f"hue_column_{col_name}"
            )
        
        with col3:
            scatter_color = st.color_picker(
                "Select color (when no hue)", 
                "#7CF5FF", 
                key=f"scatter_color_{col_name}"
            )

        # Set graph size (width=800, height=600)
        if hue_column == "None":
            scatter_fig = px.scatter(
                df, 
                x=col_name, 
                y=y_axis, 
                title=f"Scatter Plot of {col_name} vs {y_axis}", 
                color_discrete_sequence=[scatter_color],
                width=800, 
                height=600,
                template="plotly_dark"  # Use dark template
            )
        else:
            scatter_fig = px.scatter(
                df, 
                x=col_name, 
                y=y_axis, 
                color=hue_column,
                title=f"Scatter Plot of {col_name} vs {y_axis} (Hue: {hue_column})",
                width=800, 
                height=600,
                template="plotly_dark"  # Use dark template
            )

        # Update layout with dark theme
        scatter_fig.update_layout(
            xaxis_title=col_name,
            yaxis_title=y_axis,
            font=dict(
                color='white'  # White text
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#31333F',  # Subtle grid lines
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#31333F',  # Subtle grid lines
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            )
        )

        st.plotly_chart(scatter_fig)
        
    elif graph_type == "Box Plot":
        st.write(f"##### Box Plot for {col_name}")
        
        # Create columns for controls
        col1, col2, col3= st.columns(3)
        
        with col1:
            # Determine if current column is numerical
            is_numeric = pd.api.types.is_numeric_dtype(df[col_name])
            
            if is_numeric:
                # If numeric, allow selecting categorical variable for grouping
                categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
                if categorical_cols:
                    group_by = st.selectbox(
                        "Group by (categorical)", 
                        categorical_cols,
                        key=f"box_group_select_{col_name}"
                    )
                else:
                    st.warning("No categorical columns available for grouping")
                    group_by = None
            else:
                # If categorical, allow selecting numerical variable for values
                numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if numerical_cols:
                    value_col = st.selectbox(
                        "Select numerical variable", 
                        numerical_cols,
                        key=f"box_value_select_{col_name}"
                    )
                else:
                    st.warning("No numerical columns available for box plot")
                    value_col = None
        
        with col2:
            # Option to show all points
            show_points = st.selectbox(
                "Show individual points", 
                ["No Points", "Outliers Only", "All Points"],
                key=f"box_points_select_{col_name}"
            )
            
            points_map = {
                "All Points": "all",
                "Outliers Only": "outliers",
                "No Points": False
            }
        
        # with col3:
        #     box_color = st.color_picker(
        #         "Select default color", 
        #         "#1f77b4", 
        #         key=f"box_default_color_{col_name}"
        #     )
        
        # Create box plot based on data types
        if (is_numeric and group_by is not None) or (not is_numeric and value_col is not None):
            if is_numeric:
                box_fig = px.box(
                    df,
                    x=group_by,
                    y=col_name,
                    title=f"Box Plot of {col_name} by {group_by}",
                    color=group_by,
                    points=points_map[show_points],
                    width=800,
                    height=600,
                    template="plotly_dark"
                )
            else:
                box_fig = px.box(
                    df,
                    x=col_name,
                    y=value_col,
                    title=f"Box Plot of {value_col} by {col_name}",
                    color=col_name,
                    points=points_map[show_points],
                    width=800,
                    height=600,
                    template="plotly_dark"
                )
            
            # Update layout with dark theme
            box_fig.update_layout(
                font=dict(color='white'),
                showlegend=True,
                legend_title_text="Groups",
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#31333F',
                    zerolinecolor='rgba(128, 128, 128, 0.2)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#31333F',
                    zerolinecolor='rgba(128, 128, 128, 0.2)'
                )
            )
            
            # Add statistical annotations if requested
            show_stats = st.checkbox("Show statistical annotations", key=f"box_stats_check_{col_name}")
            if show_stats:
                if is_numeric:
                    stats = df.groupby(group_by)[col_name].describe()
                else:
                    stats = df.groupby(col_name)[value_col].describe()
                
                annotations = []
                y_pos = 1.1  # Position above the plot
                
                for idx, row in stats.iterrows():
                    annotation_text = (
                        f"Group: {idx}<br>"
                        f"Mean: {row['mean']:.2f}<br>"
                        f"Median: {row['50%']:.2f}<br>"
                        f"Std: {row['std']:.2f}"
                    )
                    
                    annotations.append(
                        dict(
                            x=idx,
                            y=row['max'],
                            text=annotation_text,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='white',
                            font=dict(size=10, color='white'),
                            bgcolor='rgba(0,0,0,0.7)',
                            bordercolor='white',
                            borderwidth=1,
                            borderpad=4,
                            align='left'
                        )
                    )
                
                box_fig.update_layout(annotations=annotations)
            
            st.plotly_chart(box_fig)
        else:
            st.warning("Please select appropriate numerical and categorical columns for the box plot")
    
    elif graph_type == "Pivot Chart":
        st.write(f"##### Pivot Chart for {col_name}")
        
        # Create columns for controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Select values to aggregate
            # Exclude col_name from numeric columns
            numeric_columns = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col != col_name]
            values_col = st.selectbox(
                "Select Values", 
                numeric_columns,
                key=f"pivot_values_{col_name}"
            )
        
        with col2:
            # Select columns for grouping
            column_options = ["None"] + [col for col in df.columns if col not in [col_name, values_col]]
            columns_col = st.selectbox(
                "Select Column Groups", 
                column_options,
                key=f"pivot_columns_{col_name}"
            )
        
        with col3:
            # Select aggregation function
            agg_functions = ["mean", "sum", "count", "min", "max"]
            agg_function = st.selectbox(
                "Select Aggregation", 
                agg_functions,
                key=f"agg_function_{col_name}"
            )

        # Create pivot table
        if columns_col == "None":
            pivot_data = df.pivot_table(
                values=values_col,
                index=col_name,
                aggfunc=agg_function
            )
            # Convert to regular DataFrame for single-column pivot
            pivot_data = pd.DataFrame(pivot_data)
        else:
            pivot_data = df.pivot_table(
                values=values_col,
                index=col_name,
                columns=columns_col,
                aggfunc=agg_function
            )

        # Create bar chart
        pivot_fig = px.bar(
            pivot_data,
            barmode='group',
            title=f"Pivot Chart: {agg_function.capitalize()} of {values_col} by {col_name}" + 
                (f" and {columns_col}" if columns_col != "None" else ""),
            width=800,
            height=600,
            template="plotly_dark"
        )

        # Update layout with dark theme
        pivot_fig.update_layout(
            xaxis_title=col_name,
            yaxis_title=f"{agg_function.capitalize()} of {values_col}",
            font=dict(
                color='white'
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#31333F',
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#31333F',
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            showlegend=True,
            legend_title_text=columns_col if columns_col != "None" else ""
        )

        # Display the chart
        st.plotly_chart(pivot_fig)


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




if 'uploaded_data' in st.session_state:
    
    st.title('Plots')
    # selected_columns = st.multiselect(
    #                     label="Select Columns for Analysis",
    #                     options=data.columns 
    # )
    
    # Initialize session state for selected columns if it doesn't exist
    # Place this at the beginning of your script
    if 'selected_columns_state' not in st.session_state:
        st.session_state.selected_columns_state = []

    # Function to update session state
    def update_selected_columns():
        st.session_state.selected_columns_state = st.session_state.selected_columns_widget

    # Your column selection widget
    selected_columns = st.multiselect(
        "Select Columns for Analysis",
        options=data.columns.tolist(),
        default=st.session_state.selected_columns_state,
        key='selected_columns_widget',
        on_change=update_selected_columns
    )

    if selected_columns:
        Line_Break(100)
        tabs = st.tabs(selected_columns)
        for tab, column in zip(tabs, selected_columns):
            with tab:
                colmuns_datatype = check_datatype(data, column)

                if colmuns_datatype == "object":
                    categorical_variable(data, [column])
                elif colmuns_datatype == "float64" or colmuns_datatype == "int64":
                    numarical_Features(data, [column])
        
                    
    
else:
    st.warning("Upload the dataset to view the plots")