import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tempfile
import os
from PIL import Image
import base64
import plotly.express as px
import plotly.graph_objects as go

# Set up environment check for Google API
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    if api_key := os.environ.get("GOOGLE_API_KEY"):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        GEMINI_AVAILABLE = True
except ImportError:
    st.warning("For AI features, install required packages: `pip install google-generativeai python-dotenv`")
except Exception as e:
    st.warning(f"Error initializing Gemini: {e}")

def Line_Break(width):
    line_code = f"""
    <hr style="border: none; height: 2px;width: {width}%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
    """
    st.markdown(line_code, unsafe_allow_html=True)

def save_fig(fig):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(f.name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return f.name

def df_info_string(df, max_rows=5):
    # Get schema info
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()
    
    # Get missing value info
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = "No missing values." if missing.empty else str(missing)
    
    return schema, missing_info

def ai_text_analysis(prompt_type, df_context):
    if not GEMINI_AVAILABLE: 
        return "Gemini AI not available. Please set up your GOOGLE_API_KEY environment variable."

    prompts = {
        "plan": f"You are a data analyst. Suggest a concise data analysis plan:\n{df_context}",
        "final": f"Summarize insights from the following dataset:\n{df_context}"
    }

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        res = model.generate_content(
            prompts.get(prompt_type), 
            generation_config=genai.types.GenerationConfig(max_output_tokens=500, temperature=0.3)
        )
        return res.text if res.parts else "⚠️ Gemini response blocked."
    except Exception as e:
        return f"❌ Gemini error: {e}"

def ai_vision_analysis(img_paths):
    if not GEMINI_AVAILABLE: 
        return [("AI Vision", "Gemini not available.")]

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    results = []

    for title, path in img_paths:
        try:
            img = Image.open(path)
            res = model.generate_content(
                [f"Explain this '{title}'", img],
                generation_config=genai.types.GenerationConfig(max_output_tokens=200, temperature=0.2)
            )
            results.append((title, res.text if res.parts else "⚠️ Blocked or empty response."))
        except Exception as e:
            results.append((title, f"❌ Error: {e}"))
    
    return results

def generate_visuals(df):
    visualizations = []
    saved_files = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in df.select_dtypes('object') if 1 < df[col].nunique() < 30]

    try:
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            path = save_fig(fig)
            visualizations.append(("Correlation Heatmap", path))
            saved_files.append(path)

        if len(numeric_cols) >= 3:
            sns.set(style="ticks")
            fig = sns.pairplot(df[numeric_cols[:5]].dropna()).fig
            fig.suptitle("Pairplot of Numeric Features", y=1.02)
            path = save_fig(fig)
            visualizations.append(("Pairplot", path))
            saved_files.append(path)

        for col in numeric_cols[:3]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=df, y=col, ax=ax)
            ax.set_title(f"Violin Plot for {col}")
            path = save_fig(fig)
            visualizations.append((f"Violin Plot - {col}", path))
            saved_files.append(path)

    except Exception as e:
        st.error(f"❌ Visualization error: {e}")
        plt.close('all')

    return visualizations, saved_files

def cleanup(files):
    for f in files:
        try: 
            os.remove(f)
        except: 
            pass

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def show_image(image_path, caption):
    img_base64 = image_to_base64(image_path)
    st.markdown(
        f'<figure><img src="data:image/png;base64,{img_base64}" alt="{caption}">'
        f'<figcaption>{caption}</figcaption></figure>',
        unsafe_allow_html=True
    )

# Main app functionality
st.title('AI-Powered Data Analysis Report')
Line_Break(100)

if 'uploaded_data' in st.session_state:
    df = st.session_state['uploaded_data']
    
    # Display dataset information
    st.header("Dataset Overview")
    st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Show data sample
    with st.expander("View Data Sample"):
        st.dataframe(df.head())
    
    # Get schema and missing info
    schema, missing_info = df_info_string(df)
    
    with st.expander("Dataset Schema Information"):
        st.text(schema)
    
    with st.expander("Missing Values Information"):
        st.write(missing_info)
    
    # Generate AI Analysis Plan if Gemini is available
    if GEMINI_AVAILABLE:
        st.header("AI Analysis Plan")
        with st.spinner("Generating AI analysis plan..."):
            df_context = f"Dataset shape: {df.shape}, Columns: {', '.join(df.columns.tolist())}\nSchema: {schema}\nMissing: {missing_info}"
            plan = ai_text_analysis("plan", df_context)
            st.markdown(plan)
    
    # Generate visualizations
    st.header("Data Visualizations")
    with st.spinner("Generating visualizations..."):
        visualizations, saved_paths = generate_visuals(df)
        
        if visualizations:
            for title, path in visualizations:
                st.subheader(title)
                st.image(path)
                
                # Generate AI insights for each visualization if available
                if GEMINI_AVAILABLE:
                    with st.spinner(f"Analyzing {title}..."):
                        insight = ai_vision_analysis([(title, path)])
                        if insight and insight[0][1]:
                            with st.expander(f"AI Insight for {title}"):
                                st.markdown(insight[0][1])
        else:
            st.warning("Could not generate visualizations. Check if your dataset contains appropriate numeric columns.")
    
    # Generate final AI report if Gemini is available
    if GEMINI_AVAILABLE:
        st.header("Final AI Report")
        with st.spinner("Generating comprehensive AI report..."):
            df_context = f"Dataset shape: {df.shape}, Columns: {', '.join(df.columns.tolist())}\nSchema: {schema}\nMissing: {missing_info}"
            final = ai_text_analysis("final", df_context)
            st.markdown(final)
    
    # Cleanup temporary files
    cleanup(saved_paths)

else:
    st.warning("Please upload a dataset first from the 'Upload dataset' section.")
    st.info("Once you've uploaded a dataset, return to this page to see the AI-powered analysis report.")