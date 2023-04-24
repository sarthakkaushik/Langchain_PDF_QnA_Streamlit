# -*- coding: utf-8 -*-
"""LLM Model on PDF data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n6Oavn1MTVBO9bdXdgBNapLHqeLqm79i
"""

! pip install --upgrade langchain openapi -q
!pip install --upgrade pillow
!pip install tiktoken
!pip install unstructured -q
!pip install unstructured[local-inference] -q
!pip install "unstructured[local-inference]"
!pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@e2ce8dc#egg=detectron2"
!apt-get install poppler-utils
!pip install openai
!pip install pinecone-client -q
!pip install faiss-cpu

import os
# api_keys = dotenv.dotenv_values()
os.environ["OPENAI_API_KEY"] ="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"

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

# documents[1]

"""### Splitting the data in chunks"""

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
  docs=text_splitter.split_documents(documents)
  return docs

# docs[27].page_content

"""Creating Embeddings"""

def create_embeddings(query):

  query_embeddings=embeddings.embed_query(query)
  return query_embeddings
  # len(query_embeddings)

# """## Storing embeddings in Vectore Database - PINCONE"""

# import pinecone
# from langchain.vectorstores import Pinecone
# #initalize pinecone
# pinecone.init(
# 	api_keys="d75c1f9c-a6be-46f8-a900-edf3fec97c12",
# 	enviorment="asia-northeast1-gcp"
# )

# index_name="pdf-doc-langchain"

# index=Pinecone.from_documents(docs,embeddings, index_name=index_name)

"""## Storing embeddings in Vectore Database - FAISS"""

def create_db_from_PDF(docs,embeddings):
    db = FAISS.from_documents(docs, embeddings)
    return db

"""## Intializing LLM model to answer the query"""

# def Chat_output(query, docs_page_content):
#     chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

#     # Template to use for the system message prompt
#     template = """
#     You are a helpful assistant that that can answer questions about input PDF files.

#     Only use the factual information from the PDF to answer the question.

#     If you feel like you don't have enough information to answer the question, say "I don't know".

#     Your answers should be verbose and detailed.
#     """

#     system_message_prompt = SystemMessagePromptTemplate.from_template(template)

#     # Human question prompt
#     human_template = "Answer the following question: {question}"
#     human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

#     chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

#     chain = LLMChain(llm=chat, prompt=chat_prompt)

#     response = chain.run(question=query, docs=docs_page_content)
#     response = response.replace("\n", "")
#     return response, docs

"""## Getting similar Context"""

def get_response_from_query(db,query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes 
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # response, docs = Chat_output(query,docs_page_content)
    # return response, docs,
    return docs

directory='/content/data'

def load_docs(directory):
  loader=DirectoryLoader(directory)
  documents=loader.load()
  return documents


documents=load_docs(directory)

# len(documents),documents

# docs=split_docs(documents)

# embeddings=OpenAIEmbeddings()
# db=create_db_from_PDF(docs,embeddings)

# query = "Whis the Managing Director of Motilal Oswal Services?"
# response, docs = get_response_from_query(db, query)
# # print(len(docs))

# print(textwrap.fill(response, width=50))



model_name='gpt-3.5-turbo'
llm=OpenAI(model_name=model_name)
chain=load_qa_chain(llm,chain_type='stuff')


def get_answer(query):
  similar_docs=get_response_from_query(db,query)
  answer=chain.run(input_documents=similar_docs,question=query)
  return answer

query="What is the Quarter on Quarter results ?"
get_answer(query)