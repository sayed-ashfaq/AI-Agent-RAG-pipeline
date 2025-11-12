import streamlit as st
import os
from pathlib import Path
from typing import Optional
import time

from src.tools.rag_agent.document_loader import DocumentService
from src.tools.rag_agent.retriever import Retriever
from custom_logger import GLOBAL_LOGGER as logger

from src.workflow.agent_workflow import AgentRAG


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

        supported_types = ['pdf', 'txt', 'docx', 'doc']
        if file_extension not in supported_types:
            st.warning(f"File type '.{file_extension}' may not be supported.")
            return None

        document_service = DocumentService()
        retrieved_docs = document_service.process_single_file(file_path, file_extension)

        if not retrieved_docs:
            return None

        # Generate UUIDs and add to vector store
        from uuid import uuid4
        uuids = [str(uuid4()) for _ in range(len(retrieved_docs))]
        retriever.vector_store.add_documents(documents=retrieved_docs, ids=uuids)

        logger.info(f"Successfully added {len(retrieved_docs)} chunks to vector store")

        return uuids

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None


def initialize_session_state():
    """Initialize session state variables"""
    if "retriever" not in st.session_state:
        with st.spinner(":rocket: Initializing Vector Store..."):
            st.session_state.retriever = Retriever()

    if "agent" not in st.session_state:
        with st.spinner(":rocket: Initializing AI Agent..."):
            st.session_state.agent = AgentRAG(retriever=st.session_state.retriever)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_session"

    if "documents_metadata" not in st.session_state:
        # Store: {filename: {"uuids": [...], "upload_time": "...", "chunks": int}}
        st.session_state.documents_metadata = {}


def chat_page():
    """Main chat interface page"""
    st.markdown("### :man_astronaut: Engage with the AI")

    # Display chat history
    # chat_container = st.container(height=500)

    # with chat_container:
    if not st.session_state.chat_history:
        st.warning("Connected and ready. What’s your query?", icon="👋")
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

    # Initialize session state
    initialize_session_state()

    # Sidebar for document management
    with st.sidebar:
        st.markdown("# :brain: OmniChat")
        st.markdown("---")
        # api section
        api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
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

    # load chat page
    chat_page()





if __name__ == "__main__":
    main()