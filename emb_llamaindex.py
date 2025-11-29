import os
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter

# Credenciales de OpenAI
with open("openai.txt") as archivo:
    os.environ["OPENAI_API_KEY"] = archivo.read().strip()

# Credenciales de Elasticsearch
with open("elasticstore.txt") as archivo:
    key_elastic = archivo.read().strip()

# Información de Wikipedia
reader = WikipediaReader()
documents = reader.load_data(pages=["Seguro médico"])

# Chunks
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)

# Embedding
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Carga a Elasticsearch
vector_store = ElasticsearchStore(
    es_url="http://34.72.210.195:9200",
    es_user="elastic",
    es_password=key_elastic,
    index_name="reembolsos_001_urp")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context,
    embed_model=embed_model
)

print("Indexación finalizada.")