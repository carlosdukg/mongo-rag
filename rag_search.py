import os
import pymongo
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

load_dotenv(override=True)

# Add an environment file to the notebook root directory called .env with MONGO_URI="xxx" to load these environment variables

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "doc_vector_search"

EMBEDDING_FIELD_NAME = "embedding"
client = pymongo.MongoClient("mongodb+srv://carlosdaboin:tg2yFWrEazgqve1V@clusterukgrapidsearch.v9zv1e4.mongodb.net/?retryWrites=true&w=majority")
db = client.CustomsVectorSearch
mdbcollection = db.SpecDocuments

loader = PyPDFLoader("SPEC_GEN1014_SR417944_Web Automated Kronos New Hire Import_V16.0.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
docs = text_splitter.split_documents(data)

print(docs[0])

# insert the documents in MongoDB Atlas Vector Search
x = MongoDBAtlasVectorSearch.from_documents(
documents=docs, embedding=OpenAIEmbeddings(disallowed_special=()), collection=mdbcollection, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
   "mongodb+srv://carlosdaboin:tg2yFWrEazgqve1V@clusterukgrapidsearch.v9zv1e4.mongodb.net/?retryWrites=true&w=majority",
   OpenAIEmbeddings(disallowed_special=()),
   index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

qa_retriever = vector_search.as_retriever(
   search_type="similarity",
   search_kwargs={
       "k": 200,
       "post_filter_pipeline": [{"$limit": 25}]
   }
)
from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.


{context}


Question: {question}
"""
PROMPT = PromptTemplate(
   template=prompt_template, input_variables=["context", "question"]
)


qa = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff", retriever=qa_retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})


docs = qa({"query": "New Hire Import Process"})

print(docs["result"])
print(docs['source_documents'])


