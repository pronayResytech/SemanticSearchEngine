import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from utils.utils import *
from params import *
import numpy as np
import pandas as pd
import streamlit as st
from MyCustomAgent import RunAgentPipeline

LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY_ = os.getenv('GOOGLE_API_KEY')
# try:
#     uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")
#     print(uploaded_file.name)
#     save_path = "./example_data/"+uploaded_file.name
#     print(f"save_path: {save_path}")


#     bytes_data = uploaded_file.getvalue()
#     with open(save_path, "wb") as f:
#         f.write(bytes_data)
#     FILE_PATH = save_path
#     MODEL_NAME = "models/embedding-001"

#     user_query = st.text_input("Please Enter your Prompt: ")
#     # USER_QUERY = "How many distribution centers does Nike have in the US?"
#     with st.spinner("Getting Semantic Search Engine Results..."):
#         MyResults = RunAgentPipeline(uploaded_file,MODEL_NAME,user_query)
#         st.table(MyResults)
# except Exception as e:
#     print(f"Ran into an Exception : {e}")
#     pass

st.title("📄 PDF Semantic Search")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# --- 2. Main Logic: Proceed only if a file is uploaded ---
if uploaded_file is not None:
    # Define folder and file path
    save_folder = "example_data"
    save_path = os.path.join(save_folder, uploaded_file.name)
    MODEL_NAME = "models/embedding-001"
    
    try:
        # --- 3. Save the file to disk ---
        # Create the directory if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        
        # Write the file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.success(f"File '{uploaded_file.name}' saved successfully!")

        # --- 4. Get User Input ---
        user_query = st.text_input("Please enter your prompt:")

        # --- 5. Run the agent on button click ---
        if st.button("Get Results"):
            if not user_query:
                st.warning("Please enter a prompt to continue.")
            else:
                with st.spinner("Getting Semantic Search Engine Results...",show_time=True):
                    # Pass the saved file path to the agent
                    MyResults = RunAgentPipeline(save_path, MODEL_NAME, user_query)
                    st.dataframe(MyResults)

    except Exception as e:
        # Display a user-friendly error in the app if something goes wrong
        st.error(f"An error occurred: {e}")


