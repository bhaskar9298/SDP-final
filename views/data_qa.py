import os
import pandas as pd
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
# Updated import for dataframe agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chains import RetrievalQA

load_dotenv()

# Use OpenAI API key for Euriai (assuming compatible API)
os.environ["OPENAI_API_KEY"] = os.getenv("EURI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.euron.one/api/v1/euri/alpha"  # Set custom base URL for Euriai

def create_dataframe_description(df: pd.DataFrame) -> str:
    """
    Create a comprehensive description of the dataframe
    """
    # Generate basic dataframe info
    info = []
    info.append(f"Dataframe Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    info.append(f"Column Names: {', '.join(df.columns.tolist())}")
    
    # Data types
    info.append("\nData Types:")
    for col, dtype in df.dtypes.items():
        info.append(f"- {col}: {dtype}")
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        info.append("\nNumeric Column Statistics:")
        stats = df[numeric_cols].describe().T
        for col in stats.index:
            info.append(f"- {col}: min={stats.loc[col, 'min']:.2f}, max={stats.loc[col, 'max']:.2f}, mean={stats.loc[col, 'mean']:.2f}, std={stats.loc[col, 'std']:.2f}")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        info.append("\nCategorical Column Information:")
        for col in cat_cols:
            unique_vals = df[col].nunique()
            info.append(f"- {col}: {unique_vals} unique values")
            if unique_vals < 10:  # Show examples if not too many
                info.append(f"  Example values: {', '.join(map(str, df[col].unique()[:5]))}")
    
    # Sample data
    info.append("\nSample Data (first 5 rows):")
    info.append(df.head(5).to_string())
    
    return "\n".join(info)

def setup_rag_system(df: pd.DataFrame):
    """Set up RAG system with dataframe information"""
    # Create a detailed text description of the dataframe
    df_description = create_dataframe_description(df)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(df_description)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store from text chunks
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

def setup_df_agent(df: pd.DataFrame):
    """Set up LangChain dataframe agent"""
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.2,
    )
    
    # Create a Pandas DataFrame agent with updated parameters
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True,
        # Updated to work with newer LangChain format
        agent_kwargs={"handle_parsing_errors": True}
    )
    
    return agent

def ask_question_with_rag(df: pd.DataFrame, vector_store, question: str) -> str:
    """
    Gets answers to questions about the data using LangChain with RAG
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Setup LLM
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.3,
    )
    
    # Create prompt template
    template = """
    You are a data analysis assistant. Answer questions about the data clearly and accurately.
    
    Here is information about the dataset:
    {context}
    
    Please answer the following question about this data: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        # Execute RAG pipeline
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        # Fall back to direct analysis when RAG fails
        try:
            # Create a simple analysis prompt
            analysis_prompt = f"""
            Analyze this dataframe and answer the question: {question}
            
            Here's information about the dataframe:
            {create_dataframe_description(df)}
            """
            
            # Use direct LLM call as fallback
            result = llm.invoke(analysis_prompt)
            return result.content
        except Exception as e2:
            return f"Error: Unable to answer question. {str(e2)}"

def data_qa_interface():
    """
    Creates a Streamlit interface for asking questions about the data using LangChain and RAG
    """
    st.subheader("Ask Questions About Your Data (LangChain + RAG)")
    
    # Check if uploaded_data exists in session state
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload a dataset on the 'Upload dataset' page first!")
        return
    
    # Get dataframe from session state
    df = st.session_state.uploaded_data
    
    # Show preview of the data
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Setup RAG system if not already in session state
    if 'vector_store' not in st.session_state:
        with st.spinner("Setting up knowledge base..."):
            st.session_state.vector_store = setup_rag_system(df)
    
    # Sample questions based on column detection
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_columns = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    
    sample_questions = []
    
    if len(numeric_columns) > 0:
        sample_questions.append(f"What is the average of {numeric_columns[0]}?")
        if len(numeric_columns) > 1:
            sample_questions.append(f"What's the correlation between {numeric_columns[0]} and {numeric_columns[1]}?")
    
    if len(categorical_columns) > 0:
        sample_questions.append(f"What are the most common values in {categorical_columns[0]}?")
    
    if len(date_columns) > 0:
        sample_questions.append(f"What trends can you see over time in this data?")
    
    sample_questions.append("What insights can you provide from this dataset?")
    sample_questions.append("What are the key takeaways from this data?")
    
    # Display sample questions
    if sample_questions:
        with st.expander("Sample questions you can ask"):
            for q in sample_questions:
                st.markdown(f"- {q}")
    
    # Question input
    user_question = st.text_input("Enter your question about the data:", key="data_question")
    
    # Chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Submit button
    if st.button("Ask") and user_question:
        with st.spinner("Getting answer..."):
            answer = ask_question_with_rag(df, st.session_state.vector_store, user_question)
            st.session_state.chat_history.append({"question": user_question, "answer": answer})
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, exchange in enumerate(st.session_state.chat_history):
            st.markdown(f"**Question {i+1}:** {exchange['question']}")
            st.markdown(f"**Answer:** {exchange['answer']}")
            st.markdown("---")
    
    # Clear chat history button
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Main function that runs when the page is loaded
def main():
    st.title("Data QA ChatBot")
    data_qa_interface()

# Run the app
# Execute the main function when the page is loaded
main()