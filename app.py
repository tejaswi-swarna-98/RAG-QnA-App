from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
import streamlit as st
import os
import tempfile

# Streamlit UI components
st.title("RAG QnA System 🗣")
openai_api_key = st.sidebar.text_input("Copy and Paste your OpenAI API Key and press 'ENTER'", type='password')

# Validate OpenAI API Key
if not openai_api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
    st.stop()  # Stop execution if API key is missing

# Set the OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = openai_api_key
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF Document to ask Questions from")

if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name  # Get the file path of the temporary file

    # Load and split document using the temporary file path
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Vector store and retriever setup using FAISS
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever()
        st.success("FAISS Vector Store initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize FAISS: {e}")
        st.stop()  # Stop if FAISS setup fails

    # Prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_input = st.chat_input("What would you like to ask about your document?")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response from the agent
        try:
            response = rag_chain.invoke({"input": user_input})

            # Add assistant response to chat history
            answer = response.get('answer', "No answer found.")
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as e:
            st.error(f"Error: {e}")

    # Cleanup: Remove the temporary file after processing
    os.remove(temp_path)
else:
    st.info("Upload a document to get started.")
