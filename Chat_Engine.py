import streamlit as st
import os
from pathlib import Path
from typing import Optional
import time
from uuid import uuid4

from langchain_core.documents import Document

from src.tools.rag_agent.document_loader import DocumentService
from src.tools.rag_agent.retriever import Retriever
from custom_logger import GLOBAL_LOGGER as logger

from src.workflow.agent_workflow import ReActAgent


def get_file_extension(filename: str) -> str:
    """Extract file extension without the dot"""
    return Path(filename).suffix[1:].lower()


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to a temporary location and return the path"""
    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"File saved temporarily at: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return None


def process_and_add_document(file_path: str, filename: str, retriever: Retriever) -> Optional[list]:
    """Process document and add to vector store, returns list of document UUIDs"""
    try:
        file_extension = get_file_extension(file_path)

        supported_types = ['pdf', 'txt', 'docx', 'doc', 'md']
        if file_extension not in supported_types:
            st.warning(f"File type '.{file_extension}' may not be supported.")
            return None

        document_service = DocumentService()
        retrieved_docs = document_service.process_single_file(file_path, file_extension)

        if not retrieved_docs:
            return None

        # Generate UUIDs and add to vector store
        uuids = [str(uuid4()) for _ in range(len(retrieved_docs))]
        retriever.vector_store.add_documents(documents=retrieved_docs, ids=uuids)

        logger.info(f"Successfully added {len(retrieved_docs)} chunks to vector store")

        return uuids

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None


def store_chat_history_in_vstore(retriever, chat_history: list):
    """
    Store chat history messages in the vector store.

    Each message (user/assistant) becomes a Document with simple metadata.
    """
    if not chat_history:
        return

    # Convert chat messages to LangChain Documents
    docs = []
    for msg in chat_history:
        docs.append(
            Document(
                page_content=msg["content"],
                metadata={
                    "role": msg["role"],
                    "type": "chat_history"
                },
            )
        )

    # Generate unique IDs for each message and add them
    uuids = [str(uuid4()) for _ in range(len(docs))]
    retriever.vector_store.add_documents(documents=docs, ids=uuids)
    logger.info("Successfully Added chat history messages to vector store")
    return uuids


def initialize_session_state():
    """Initialize session state variables"""
    if "retriever" not in st.session_state:
        with st.spinner(":rocket: Initializing Vector Store..."):
            st.session_state.retriever = Retriever()

    if "agent" not in st.session_state:
        with st.spinner(":rocket: Initializing AI Agent..."):
            st.session_state.agent = ReActAgent(retriever=st.session_state.retriever)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_session"

    if "documents_metadata" not in st.session_state:
        # Store: {filename: {"uuids": [...], "upload_time": "...", "chunks": int}}
        st.session_state.documents_metadata = {}


def welcome_page():
    """Display welcome page with instructions"""
    st.markdown("## 🚀 Welcome to OmniChat")
    st.markdown("---")

    st.markdown("""
    ### How to Use This App

    OmniChat is an intelligent AI assistant powered by RAG (Retrieval-Augmented Generation) technology. 
    To get started, simply enter your API keys in the sidebar, upload your documents, and start chatting! 
    The AI will use the uploaded documents as context to provide accurate and relevant answers to your questions.

    ---

    ### 🔑 Getting Started

    **Step 1:** Get your API keys:

    1. **[OpenAI API Key](https://platform.openai.com/api-keys)** 
       - Sign up for OpenAI account
       - Go to API Keys section
       - Create new secret key

    2. **[OpenWeather API Key](https://openweathermap.org/api)**
       - Create free account
       - Get API key from dashboard
       - Free tier available

    **Step 2:** Enter your API keys in the sidebar ← 

    **Step 3:** Upload documents (PDF, TXT, DOCX) for the AI to reference

    **Step 4:** Ask questions and get intelligent responses!

    ---

    ### 💡 Pro Tips
    - All processing happens securely - keys are only stored during your session
    - Upload multiple documents to build a comprehensive knowledge base
    - The AI remembers conversation context throughout your session
    - Use the "Clear Chat History" button to start fresh conversations
    """)


def chat_page():
    """Main chat interface page"""
    st.markdown("### :man_astronaut: Engage with the AI")

    # Display chat history
    # chat_container = st.container(height=500)

    # with chat_container:
    if not st.session_state.chat_history:
        st.info("Connected and ready. What's your query?", icon="👋")
    else:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Your message"):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        # add chat history to the vector storage

        # Get agent response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.agent.run(
                    query=prompt,
                    thread_id=st.session_state.thread_id
                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                store_chat_history_in_vstore(st.session_state.retriever, st.session_state.chat_history)

            except Exception as e:
                logger.error(f"Error getting agent response: {e}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}"
                })

        st.rerun()


def main():
    st.set_page_config(
        page_title="AI Agent with RAG",
        page_icon=":man_astronaut:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar for API keys and document management
    with st.sidebar:
        st.markdown("# :brain: OmniChat")
        st.markdown("---")

        # API Keys section
        st.header("🔑 API Configuration")
        st.markdown("*Enter your API keys to get started*")

        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Required for AI-powered chat and document analysis"
        )
        if not openai_key:
            st.markdown("👆 [Get OpenAI API Key](https://platform.openai.com/api-keys)")
        else:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success("✅ OpenAI key configured")

        # OpenWeather API Key
        weather_key = st.text_input(
            "OpenWeather API Key",
            type="password",
            placeholder="Enter OpenWeather API Key",
            help="Required for weather-related queries"
        )
        if not weather_key:
            st.markdown("👆 [Get OpenWeather API Key](https://openweathermap.org/api)")
        else:
            os.environ["OPENWEATHER_API_KEY"] = weather_key
            st.success("✅ OpenWeather key configured")

        # Check if all keys are provided
        all_keys_provided = bool(openai_key and weather_key)

        if not all_keys_provided:
            st.warning("⚠️ Please provide all API keys to use the app")
            st.markdown("""
            **Why these APIs?**
            - **OpenAI**: AI chat and document embeddings
            - **OpenWeather**: Real-time weather information
            """)

        st.markdown("---")

        # Show document upload only if keys are set
        if all_keys_provided:
            # Upload section
            st.markdown("### Upload Your Documents")
            st.caption("Drag and drop files here")

            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'docx', 'doc'],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="file_uploader"
            )

            # Auto-process uploaded files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Check if file already processed
                    if uploaded_file.name not in st.session_state.documents_metadata:
                        with st.spinner(f"⚙️ Processing {uploaded_file.name}..."):
                            # Save file
                            file_path = save_uploaded_file(uploaded_file)

                            if file_path:
                                # Process and add to vector store
                                uuids = process_and_add_document(
                                    file_path,
                                    uploaded_file.name,
                                    st.session_state.retriever
                                )

                                if uuids:
                                    # Store metadata
                                    st.session_state.documents_metadata[uploaded_file.name] = {
                                        "uuids": uuids,
                                        "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "chunks": len(uuids)
                                    }
                                    st.success(f"✅ {uploaded_file.name}")
                                else:
                                    st.error(f"❌ Failed to process {uploaded_file.name}")

                                # Clean up temp file
                                try:
                                    os.remove(file_path)
                                except:
                                    pass

            st.markdown("---")

            # Display uploaded documents with delete option
            if st.session_state.documents_metadata:
                st.markdown("### 📚 Knowledge Base")

                for filename, metadata in list(st.session_state.documents_metadata.items()):
                    with st.container():
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"📄 **{filename}**")
                            st.caption(f"{metadata['chunks']} chunks • {metadata['upload_time']}")

                        with col2:
                            if st.button("🗑️", key=f"delete_{filename}", help="Delete document"):
                                # Delete from vector store
                                try:
                                    for uuid in metadata["uuids"]:
                                        st.session_state.retriever.delete_document(uuid)

                                    # Remove from metadata
                                    del st.session_state.documents_metadata[filename]
                                    st.success(f"Deleted {filename}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting: {e}")

            else:
                st.info("📭 No documents uploaded yet")

            # Clear chat button
            st.markdown("---")
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("🔒 Enter API keys above to unlock features")

    # Main content area
    if all_keys_provided:
        # Initialize session state after keys are set
        initialize_session_state()
        # Load chat page
        chat_page()
    else:
        # Show welcome page when API keys are missing
        welcome_page()


if __name__ == "__main__":
    main()