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
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.chains.question_answering import load_qa_chain

import textwrap

# Add your existing functions

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
  docs=text_splitter.split_documents(documents)
  return docs

def create_embeddings(query):
  query_embeddings=embeddings.embed_query(query)
  return query_embeddings

def create_db_from_PDF(docs,embeddings):
    db = FAISS.from_documents(docs, embeddings)
    return db

# directory='/content/data'

def load_docs(directory):
  loader=DirectoryLoader(directory)
  documents=loader.load()
  return documents

documents=load_docs(directory)

model_name='gpt-3.5-turbo'
llm=OpenAI(model_name=model_name)
chain=load_qa_chain(llm,chain_type='stuff')


def get_answer(query):
  similar_docs=get_response_from_query(db,query)
  answer=chain.run(input_documents=similar_docs,question=query)
  return answer



####################################################################

# Set up the Streamlit app
st.set_page_config(page_title="Langchain PDF QnA", layout="wide")

st.subheader("OpenAI API Key")
api_key = st.text_input("Enter your OpenAI API Key:")
os.environ["OPENAI_API_KEY"] = api_key

#Create an option to upload a PDF or URL:
st.subheader("Upload PDF or Enter URL")
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
pdf_url = st.text_input("Enter a PDF URL:")

#Integrate the uploaded PDF or URL with your existing code:
if pdf_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.getvalue())
        # Update the directory variable with the path to the temporary PDF file
        directory = temp_pdf.name
        documents = load_docs(directory)

elif pdf_url:
    response = requests.get(pdf_url)
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(response.content)
        # Update the directory variable with the path to the temporary PDF file
        directory = temp_pdf.name
        documents = load_docs(directory)
        
# Now split the documents and create embeddings and database
if pdf_file is not None or pdf_url:
    split_documents = split_docs(documents)
    embeddings = OpenAIEmbeddings(llm)
    db = create_db_from_PDF(split_documents, embeddings)

#Create a query box for users to input their query:
st.subheader("Ask a Question")
query = st.text_input("Enter your question:")

#Display the query results:
if query and (pdf_file is not None or pdf_url):
    answer = get_answer(query)
    st.subheader("Answer")
    st.write(answer)
elif query:
    st.warning("Please upload a PDF file or enter a PDF URL.")

