import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
)
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.mongodb_atlas import MongoDBAtlasTranslator
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv
from os import listdir
from os.path import isfile, join

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
API_VERSION = os.getenv("OPENAI_API_VERSION")
COSMOSDB_CONNECTION_STR = os.getenv("COSMOS_DB_CONNECTION_STRING")
EMBEDDINGS_DEPLOYMENT = os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT")
COMPLETIONS_DEPLOYMENT = os.getenv("OPENAI_COMPLETIONS_DEPLOYMENT")

client: MongoClient = MongoClient(COSMOSDB_CONNECTION_STR)
NAMESPACE = os.getenv("NAMESPACE")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")
COLLECTION = client[DB_NAME][COLLECTION_NAME]
INDEX_NAME = os.getenv("INDEX_NAME")

NUM_LISTS = 100
DIMENSIONS = 1536
SIMILARITY_ALGORITHM = CosmosDBSimilarityType.COS


def get_embedding_function():
    openai_embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        azure_deployment = EMBEDDINGS_DEPLOYMENT,
        openai_api_version = API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )
    return openai_embeddings

def store_scripts():
    onlyfiles = [join("powershellScripts/", f) for f in listdir("powershellScripts/") if isfile(join("powershellScripts/", f))]
    for file in onlyfiles:
        docs = split_document(file, False)
        add_to_vector_store(docs)

def split_document(filename, chunk=True):
    loader = TextLoader(filename)
    documents = loader.load()
    
    if chunk:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)
    return documents

def get_vector_store(collection = COLLECTION, index_name = INDEX_NAME):
    return AzureCosmosDBVectorSearch(
        embedding = get_embedding_function(),
        collection = collection,
        index_name = index_name
    )

def add_to_vector_store(docs, collection = COLLECTION, index_name = INDEX_NAME):
    vectorstore = AzureCosmosDBVectorSearch.from_documents(
        docs,
        embedding=get_embedding_function(),
        collection=collection,
        index_name=index_name,
    )

    vectorstore.create_index(NUM_LISTS, DIMENSIONS, SIMILARITY_ALGORITHM)
    return vectorstore

def get_retriever(vectorstore = get_vector_store()):
    return VectorStoreRetriever(
        vectorstore=vectorstore
    )

def get_metadata_info():
    return [
        AttributeInfo(
            name="name",
            description="The name of the file",
            type="string",
        ),
        AttributeInfo(
            name="fileSystemInfo",
            description="The creation and upload time of the file",
            type="object",
        ),
        AttributeInfo(
            name="source",
            description="The link to the uploaded file",
            type="string",
        ),
    ]
    
def get_self_query_retriever(vectorstore = get_vector_store()):
    metadata_field_info = get_metadata_info()
    document_content_description = "Powershell scripts and modules"
    llm = AzureChatOpenAI(
        api_key = API_KEY,
        api_version = API_VERSION,
        azure_endpoint = AZURE_ENDPOINT,
        azure_deployment=COMPLETIONS_DEPLOYMENT,
    )
    return SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info, MongoDBAtlasTranslator(), verbose=True)

if __name__ == "__main__":

    vectorstore = get_vector_store()

    docs = vectorstore.similarity_search("GatheredLogName")
    
    print([doc.metadata['name'] for doc in docs])
    print(len(docs))
    print(docs)

    # for doc in docs:
    #     print(doc.page_content)
    #     print(doc.metadata['name'])
    # store_scripts()
    # retriver = get_self_query_retriever()
    # results = retriver.invoke("What does the datepicker file do")
    # for result in results:
    #     print( result.metadata['source'])
    #     print(result.page_content)

