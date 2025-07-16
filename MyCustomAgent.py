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

LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY_ = os.getenv('GOOGLE_API_KEY')

def RunAgentPipeline(FILE_PATH,MODEL_NAME,USER_QUERY):
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    documents = [
        Document(
            page_content=docs[11].page_content,
            metadata={"source": FILE_PATH},
        ),
    ]
    print("DocumentLoader Successful!!!")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=200,
                                                add_start_index=True)
    print("text_splitter Successful!!!")
    all_splits = text_splitter.split_documents(docs)
    print("All Splits Successful!!!")
    embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_NAME)
    print("Embeddings Successful!!!")
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)
    print("Get Vectors Successful!!!")
    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)
    print("Get Doc Ids Successful!!!")
    results = vector_store.similarity_search_with_score(USER_QUERY)
    #results = vector_store.similarity_search_with_score(user_query)
    print("Vector Store Results Successful!!!")
    print(f"#"*100)
    print(f"Error Handling:")
    # print(results[0])
    print(f"#"*100)
    doc_id_list = []
    doc_metadata_list = []
    doc_resp_page_content_list = []
    doc_sim_score_list = []
    for index__, page_content__ in enumerate(results):
        try:
            print(f"*"*50)
            #print(f"At index {index__} the page content is : {results[index__].page_content}")
            #print(f"Error Handling:")
            #print(results[index__])
            doc_resp,score = results[index__]
            doc_id = doc_resp.id
            doc_metadata=  doc_resp.metadata
            doc_resp_page_content = doc_resp.page_content
            doc_sim_score = score
            doc_sim_score = np.round(doc_sim_score*100,2 )
            print(f"Details for Doc ID:  {doc_id} \n")
            print(f"doc_metadata:  {doc_metadata} \n")
            print(f"doc_resp_page_content:  {doc_resp_page_content} \n")
            print(f"doc_sim_score:  {doc_sim_score}")
            doc_id_list.append(doc_id)
            doc_metadata_list.append(doc_metadata)
            doc_resp_page_content_list.append(doc_resp_page_content)
            doc_sim_score_list.append(doc_sim_score)
        except Exception as e:
            print(f"The Index {index__} ran into an Exception {e}")
            pass
    
    result_dict = {"doc_id":doc_id_list,
                 "metadata":doc_metadata_list,
                 "page_contnet":doc_resp_page_content_list,
                 "sim_score":doc_sim_score_list}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("./Results/SemanticResults.csv", index=False)
    return result_df

    
if __name__ =="__main__":
    FILE_PATH = "./example_data/nke-10k-2023.pdf"
    MODEL_NAME = "models/embedding-001"
    USER_QUERY = "How many distribution centers does Nike have in the US?"
    MyResults = RunAgentPipeline(FILE_PATH,MODEL_NAME,USER_QUERY)
    print(MyResults)
