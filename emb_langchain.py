import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore

# Credenciales de OpenAI
with open("openai.txt") as archivo:
    os.environ["OPENAI_API_KEY"] = archivo.read().strip()

# Credenciales de Elasticsearch
with open("elasticstore.txt") as archivo:
    key_elastic = archivo.read().strip()

# Carga y chunking
loader = PyPDFLoader(file_path="procedimiento_reembolsos.pdf")
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

separadores = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    " "
]

pdf = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=separadores, is_separator_regex=False)
all_splits = text_splitter.split_documents(pdf)

# Carga a Elasticsearch
vector_store = ElasticsearchStore.from_documents(
    all_splits,
    embedding,
    es_url="http://34.45.156.122:9200",
    es_user="elastic",
    es_password=key_elastic,
    index_name="reembolsos_001_urp")
vector_store.client.indices.refresh(index="reembolsos_001_urp")

print("Indexación finalizada.")