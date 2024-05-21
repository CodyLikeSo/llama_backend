from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter






loader = TextLoader("/home/cody/Code/React/backend/back1/File.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=50, separator=' ' )
docs = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(
    model='all-minilm'
)
db = FAISS.from_documents(docs, embeddings)

db_local = db.save_local("faiss_index")