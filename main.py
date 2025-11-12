# app.py
"""Streamlit chat interface for the AI pipeline."""

import streamlit as st
import os
from pathlib import Path
# from src.agents.graph import AIPipeline
# from config.settings import LANGCHAIN_TRACING_V2

# Page configuration
st.set_page_config(
    page_title="AI Pipeline Chat",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .metadata-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = []


@st.cache_resource
def load_pipeline():
    """Load the AI pipeline (cached to avoid reloading)."""
    return "AIPipeline()"


def display_message(role, content, metadata=None):
    """Display a chat message with optional metadata."""
    css_class = "user-message" if role == "user" else "assistant-message"

    with st.container():
        st.markdown(f'<div class="chat-message {css_class}">', unsafe_allow_html=True)
        st.markdown(f"**{role.capitalize()}:** {content}")

        if metadata:
            if metadata.get("intent"):
                st.markdown(f"*Intent: {metadata['intent']}*")

            if metadata.get("weather_data"):
                with st.expander("Weather Details"):
                    st.json(metadata["weather_data"])

            if metadata.get("rag_sources"):
                with st.expander(f"Document Sources ({len(metadata['rag_sources'])})"):
                    for i, source in enumerate(metadata["rag_sources"], 1):
                        st.markdown(f"**Source {i}** (Score: {source['score']:.2f})")
                        st.text(source["text"][:200] + "...")

        st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<p class="main-header">🤖 AI Pipeline Chat Interface</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📚 Configuration")

        # Pipeline status
        if st.session_state.pipeline is None:
            with st.spinner("Loading AI Pipeline..."):
                st.session_state.pipeline = load_pipeline()
            st.success("✅ Pipeline Ready!")
        else:
            st.success("✅ Pipeline Ready!")

        st.divider()

        # Document Upload
        st.subheader("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Upload a PDF to index",
            type=["pdf"],
            help="Upload PDF documents for RAG-based Q&A"
        )

        if uploaded_file:
            if uploaded_file.name not in st.session_state.documents_indexed:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    # Save uploaded file temporarily
                    temp_path = Path("data/documents") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        # Index the document
                        st.session_state.pipeline.index_document(str(temp_path))
                        st.session_state.documents_indexed.append(uploaded_file.name)
                        st.success(f"✅ Indexed: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error indexing: {str(e)}")

        # Display indexed documents
        if st.session_state.documents_indexed:
            st.subheader("Indexed Documents")
            for doc in st.session_state.documents_indexed:
                st.text(f"📄 {doc}")

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        var = "true"
        if var == "true":
            st.info("🔍 LangSmith Tracing: Enabled")

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            **Weather Queries:**
            - "What's the weather in London?"
            - "Temperature in Tokyo?"

            **Document Queries:**
            - "What does the document say about X?"
            - "Explain the concept of Y from the PDF"

            **General Queries:**
            - "Tell me about artificial intelligence"
            - "How does machine learning work?"
            """)

    # Main chat area
    st.subheader("💬 Chat")

    # Display chat messages
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("metadata")
        )

    # Chat input
    if prompt := st.chat_input("Ask me about weather or documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        display_message("user", prompt)

        # Get response from pipeline
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.pipeline.run(prompt)

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "metadata": {
                        "intent": result["intent"],
                        "weather_data": result["metadata"].get("weather_data"),
                        "rag_sources": result["metadata"].get("rag_sources")
                    }
                })

                # Display assistant message
                display_message(
                    "assistant",
                    result["response"],
                    {
                        "intent": result["intent"],
                        "weather_data": result["metadata"].get("weather_data"),
                        "rag_sources": result["metadata"].get("rag_sources")
                    }
                )

                # Show error if any
                if result.get("error"):
                    st.warning(f"⚠️ {result['error']}")

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Powered by LangChain, LangGraph, and LangSmith | OpenAI GPT-4 | Qdrant Vector Store
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()