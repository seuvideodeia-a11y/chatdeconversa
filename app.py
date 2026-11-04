import streamlit as st
import tempfile
import os

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
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
                        )
                        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                        # Prompt de Contextualização do Histórico
                        contextualize_q_system_prompt = (
                            "Dada uma conversa recente e a última pergunta do usuário, "
                            "formule uma pergunta independente que possa ser entendida sem o histórico."
                        )
                        contextualize_q_prompt = ChatPromptTemplate.from_messages([
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ])
                        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

                        # Prompt de Resposta Principal com RAG + Persona
                        qa_system_prompt = (
                            f"{system_persona}\n\n"
                            "Utilize os seguintes trechos de contexto recuperados para responder à pergunta. "
                            "Se não souber a resposta com base no contexto, informe claramente.\n\n"
                            "Contexto:\n{{context}}"
                        )
                        qa_prompt = ChatPromptTemplate.from_messages([
                            ("system", qa_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ])
                        
                        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                        st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                    else:
                        # Modo sem RAG vetorial (Chat direto com Persona)
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", system_persona),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ])
                        st.session_state.rag_chain = prompt | llm

                    st.session_state.connected = True
                    st.success("Conexão estabelecida com sucesso!")

                except Exception as e:
                    st.error(f"Erro ao conectar com Azure OpenAI: {str(e)}")

# --- ÁREA PRINCIPAL DO CHAT ---

# Cabeçalho de Ações da Sessão
col_title, col_reset = st.columns([4, 1])
with col_reset:
    if st.button("🔄 Novo Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Exibição das mensagens anteriores
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Campo de entrada de texto
user_input = st.chat_input("Digite sua dúvida técnica ou solicitação sobre a obra...")

if user_input:
    if not st.session_state.connected or st.session_state.rag_chain is None:
        st.warning("⚠️ Insira as credenciais da Azure na barra lateral e clique em 'Iniciar / Atualizar Sessão' antes de conversar.")
    else:
        # Exibe mensagem do usuário
        with st.chat_message("user"):
            st.markdown(user_input)

        # Processa e exibe resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Analisando solicitação..."):
                try:
                    if isinstance(st.session_state.rag_chain, dict) or hasattr(st.session_state.rag_chain, "invoke"):
                        # Verifica se é uma cadeia RAG (retorna dict com 'answer') ou Chain simples
                        response = st.session_state.rag_chain.invoke({
                            "input": user_input,
                            "chat_history": st.session_state.chat_history
                        })
                        
                        answer = response["answer"] if isinstance(response, dict) and "answer" in response else response.content
                    
                    st.markdown(answer)

                    # Atualiza histórico na sessão
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=answer))

                except Exception as e:
                    st.error(f"Erro ao processar resposta: {str(e)}")
