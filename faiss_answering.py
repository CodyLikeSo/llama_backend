from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain.memory import ChatMessageHistory, ConversationBufferMemory




async def get_text(query):
    embeddings = OllamaEmbeddings(
        model='all-minilm'
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    model = Ollama(model='llama3',
                   temperature=0.1,
                   keep_alive=-1,
                   mirostat_tau=2.0,
                   mirostat_eta=0.9,
                #    top_k=15, 
                #    top_p=0.7, 
                #    repeat_penalty=1.5, 
                #    num_predict=60
            
                )

    docs = db.similarity_search(query=query, k=6)

    answer = model.invoke(f'You are Arseniy. Answer the question strictly - 1-2 sentences. Question: {query}. Find the answer here: {docs}.')

    return answer




def get_text1(query):
    embeddings = OllamaEmbeddings(
        model='all-minilm'
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    model = Ollama(model='llama3',
                   temperature=0.1,
                   keep_alive=-1,
                   mirostat_tau=2.0,
                   mirostat_eta=0.9,
                #    top_k=15, 
                #    top_p=0.7, 
                #    repeat_penalty=1.5, 
                #    num_predict=60
            
                )

    docs = db.similarity_search(query=query, k=6)

    answer = model.invoke(f'You are Arseniy. Answer the question strictly - 1-2 sentences. Question: {query}. Find the answer here: {docs}.')

    return answer


a = get_text1('do you use docker before?')
print(a)