import streamlit as st

st.markdown("### 🚀 About This Application")

st.markdown("""
## AI Agent with RAG & Tools

This intelligent agent combines multiple capabilities to provide comprehensive assistance:

### 🎯 Key Features

#### 1. **Document Analysis (RAG)**
- Upload and process multiple document formats (PDF, TXT, DOCX)
- Intelligent document chunking and embedding
- Semantic search across your knowledge base
- Context-aware responses based on document content

#### 2. **Real-time Weather Information**
- Get current weather data for any location worldwide
- Powered by OpenWeatherMap API
- Temperature, conditions, and forecast information

#### 3. **Conversational AI**
- Natural language understanding
- Context retention across conversations
- Multi-turn dialogue support

### 🏗️ Architecture

**Technology Stack:**
- **Framework**: LangGraph + LangChain
- **Vector Store**: Qdrant (Local)
- **LLM**: OpenAI GPT Models
- **UI**: Streamlit
- **Embeddings**: OpenAI Embeddings

**Workflow:**
""")

st.code("""
flowchart LR
    A[User Query] --> B[Agent (LLM)]
    B --> C[Decision Node]
    C --> D[Tool Node (RAG / Weather)]
    D --> E[Context Retrieved]
    E --> F[Agent (with context)]
    F --> G[Final Response]
    G --> H[User Output]
    C --> I[Direct Response]
    I --> H
""", language="mermaid")

st.markdown("""
### 📊 Use Cases

1. **Research Assistant**: Upload research papers and ask questions
2. **Document Q&A**: Extract information from long documents
3. **Travel Planning**: Check weather before trips
4. **Knowledge Management**: Build a searchable knowledge base

### 🔒 Privacy & Security

- Documents are stored locally in Qdrant vector database
- No data is sent to external servers except for LLM API calls
- Easy document deletion for data management

### 📈 Performance

- **Response Time**: 2-5 seconds average
- **Document Processing**: ~1 second per page
- **Vector Search**: Sub-second retrieval
- **Concurrent Users**: Optimized for single-user sessions

### 🛠️ Future Enhancements

- [ ] Multi-modal document support (images, tables)
- [ ] Advanced filtering and search options
- [ ] Export conversation history
- [ ] Custom embedding models
- [ ] API endpoint for programmatic access

---

**Version**: 1.0.0  
**Last Updated**: November 2024  
**Built with**: ❤️ by Your Team
""")

# Example queries
st.markdown("### 💡 Example Queries")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Document Questions:**
    - "Summarize the main findings in the uploaded paper"
    - "What methodology was used in the research?"
    - "Find all mentions of [specific topic]"
    """)

with col2:
    st.markdown("""
    **Weather Questions:**
    - "What's the weather like in Tokyo?"
    - "Should I bring an umbrella in London today?"
    - "Is it cold in New York right now?"
    """)