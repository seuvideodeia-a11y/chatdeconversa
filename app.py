import streamlit as st
import tempfile
import os

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Importações corrigidas para versões recentes do LangChain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Suporte a Obra - Assistente IA", layout="wide", page_icon="🏗️")
st.title("🏗️ Suporte a Obra - Assistente Técnico")

# --- INICIALIZAÇÃO DA SESSÃO (STATE) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "connected" not in st.session_state:
    st.session_state.connected = False

# --- BARRA LATERAL: CONEXÃO E CONFIGURAÇÕES ---
with st.sidebar:
    st.header("🔑 Conexão Azure OpenAI / Foundry")
    
    azure_api_key = st.text_input("API Key (Chave de Acesso)", type="password", key="api_key")
    azure_endpoint = st.text_input("Endpoint (Ex: https://seu-recurso.openai.azure.com/)", key="endpoint")
    azure_deployment = st.text_input("Nome do Deployment / Modelo LLM", key="deployment")
    azure_api_version = st.text_input("Versão da API", value="2024-02-15-preview", key="api_version")
    
    # Campo opcional para embeddings caso utilize modelo de embedding diferente na Azure
    embedding_deployment = st.text_input("Nome do Deployment de Embedding (Opcional)", value="", help="Deixe em branco se desejar usar RAG simples ou preencha se tiver modelo de embeddings configurado.")
    
    st.divider()
    st.header("🎭 Persona do Assistente")
    persona_default = (
        "Você é um engenheiro civil sênior especialista em suporte a obras, "
        "normas técnicas (NBR), procedimentos operacionais e segurança do trabalho. "
        "Responda às dúvidas com clareza, precisão e foco prático na rotina de canteiro de obras."
    )
    system_persona = st.text_area("Instruções do Sistema (Persona)", value=persona_default, height=150)
    
    st.divider()
    st.header("📁 Documentos de Apoio (RAG)")
    uploaded_files = st.file_uploader(
        "Upload de arquivos de orientação", 
        type=["pdf", "txt", "csv"], 
        accept_multiple_files=True
    )

    # Botão para Inicializar Conexão
    if st.button("🚀 Iniciar / Atualizar Sessão", type="primary"):
        if not azure_api_key or not azure_endpoint or not azure_deployment:
            st.error("Por favor, preencha a API Key, Endpoint e Modelo antes de iniciar.")
        else:
            with st.spinner("Processando documentos e conectando ao modelo..."):
                try:
                    # 1. Configurar o LLM Azure OpenAI
                    llm = AzureChatOpenAI(
                        azure_deployment=azure_deployment,
                        azure_endpoint=azure_endpoint,
                        api_key=azure_api_key,
                        api_version=azure_api_version,
                        temperature=0.2
                    )

                    documents = []
                    # 2. Leitura dos Arquivos Carregados
                    if uploaded_files:
                        for file in uploaded_files:
                            file_ext = os.path.splitext(file.name)[1].lower()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                                tmp_file.write(file.getvalue())
                                tmp_path = tmp_file.name

                            if file_ext == ".pdf":
                                loader = PyPDFLoader(tmp_path)
                            elif file_ext == ".txt":
                                loader = TextLoader(tmp_path, encoding="utf-8")
                            elif file_ext == ".csv":
                                loader = CSVLoader(tmp_path, encoding="utf-8")
                            
                            documents.extend(loader.load())
                            os.remove(tmp_path)

                    # 3. Construção do RAG ou Cadeia Simples
                    if documents and embedding_deployment:
                        # Divisão em partes para criação da base vetorial
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        splits = text_splitter.split_documents(documents)
                        
                        embeddings = AzureOpenAIEmbeddings(
                            azure_deployment=embedding_deployment,
                            azure_endpoint=azure_endpoint,
                            api_key=azure_api_key,
                            api_version=azure_api_version
