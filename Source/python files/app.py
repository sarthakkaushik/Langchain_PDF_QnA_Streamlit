import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from tempfile import NamedTemporaryFile

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import document_loaders
from langchain.document_loaders import DirectoryLoader
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
  
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.chains.question_answering import load_qa_chain
import uuid
import textwrap
os.environ["OPENAI_API_KEY"] ="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"
model_name='gpt-3.5-turbo'
llm=OpenAI(model_name=model_name)
chain=load_qa_chain(llm,chain_type='stuff')

#initialize variables in session state
if 'query_result' not in st.session_state:
    st.session_state['query_result'] = ''
if 'token_tracking' not in st.session_state:
    st.session_state['token_tracking'] = ''
st.session_state['curr_dir'] = os.path.dirname(__file__)
st.session_state['agent_created_files_folder'] = rf"{st.session_state['curr_dir']}\agent_created_files"

# Add your existing functions

def make_dir():
    st.session_state['query_id'] = str(uuid.uuid4())
    st.session_state['session_folder'] = f"{st.session_state['agent_created_files_folder'] }\\{st.session_state['query_id']}"
    if not os.path.exists(st.session_state['session_folder']):
                os.makedirs(st.session_state['session_folder'])



def load_docs(directory):
  loader=DirectoryLoader(directory)
  documents=loader.load()
  return documents



def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
  docs=text_splitter.split_documents(documents)
  return docs

def create_embeddings(query):
  query_embeddings=embeddings.embed_query(query)
  return query_embeddings

# def create_db_from_PDF(docs,embeddings):
#     print(f"Embeddings: {embeddings}, Length: {len(embeddings)}")
#     db = FAISS.from_documents(docs, embeddings)
#     return db

def create_db_from_PDF(docs, embeddings_obj):
    db = FAISS.from_documents(docs, embeddings_obj)
    return db



def get_answer(query):
  similar_docs=get_response_from_query(db,query)
  answer=chain.run(input_documents=similar_docs,question=query)
  return answer



####################################################################

# Set up the Streamlit app
st.set_page_config(page_title="Langchain PDF QnA", layout="wide")

st.subheader("OpenAI API Key")

# st.session_state['openai_api_key'] = st.text_input('Enter your OpenAI API key:', placeholder ='sk-...')
# os.environ['OPENAI_API_KEY'] = st.session_state['openai_api_key']


#UI upload file
uploaded_file = st.file_uploader('upload a pdf', type=['pdf'])

if uploaded_file is not None:
    make_dir() #create a new folder
    
    with open(f"{st.session_state['session_folder']}\\source.pdf", 'wb') as f:
        f.write(uploaded_file.getbuffer())
        st.success(f"file has been saved to {st.session_state['session_folder']}\\source.pdf")
    st.write(uploaded_file)
    directory = os.path.join(st.session_state['session_folder'], "source.pdf")
    documents = load_docs(directory)
    # loader = PyPDFLoader(f"{st.session_state['session_folder']}\\source.pdf")
    # pages = loader.load_and_split()
    # page_range = st.slider('Select a range of pages to ', 1, len(pages), (3, 8))
    # st.write(page_range)

# pages_to_use = []
create_embedding_button = st.button('Learn data')

#UI button - create index and agent
# if create_embedding_button:
#     docs= split_docs(documents,chunk_size=1000,chunk_overlap=20)
#     directory=f"{st.session_state['session_folder']}\\source.pdf"
#     documents=load_docs(directory)
#     docs=split_docs(documents)
#     embeddings=OpenAIEmbeddings()
#     db=create_db_from_PDF(docs,embeddings)
    
if create_embedding_button:
    directory = f"{st.session_state['session_folder']}\\source.pdf"
    documents = load_docs(directory)
    docs = split_docs(documents)
    embeddings = OpenAIEmbeddings(llm)
    db = create_db_from_PDF(docs, embeddings)

    
    

#Create a query box for users to input their query:
st.subheader("Ask a Question")
query = st.text_input("Enter your question:")

#Display the query results:
if query:
    answer = get_answer(query)
    st.subheader("Answer")
    st.write(answer)
elif query:
    st.warning("Please upload a PDF file or enter a PDF URL.")

