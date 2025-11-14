import streamlit as st

st.markdown("## 🤖 About This Application")

st.markdown("""
**AI-Agent-RAG-Pipeline** is an intelligent agent built using **LangGraph** and **LangChain**, designed to demonstrate how AI systems can autonomously choose between multiple tools and respond contextually.  
It combines **real-time weather information** and **document-based question answering (RAG)** into a unified conversational interface.
""")

st.divider()

st.markdown("### 🎯 Key Features")

st.markdown("""
#### 1. **Document Analysis (RAG)**
- Upload and process multiple document formats (PDF, TXT, DOCX)
- Embeds documents using OpenAI embeddings and stores them in Qdrant
- Performs semantic retrieval for context-aware responses
- Supports multi-turn, context-based Q&A

#### 2. **Real-time Weather Data**
- Fetches accurate, up-to-date weather details using OpenWeatherMap API
- Provides temperature, humidity, and conditions for any global location

#### 3. **Conversational AI**
- LLM-based reasoning powered by LangChain
- Handles tool selection autonomously
- Maintains context and supports multi-turn dialogue
""")

st.divider()

st.markdown("### 🏗️ Architecture")

st.markdown("""
**Technology Stack:**
- **Framework**: LangGraph + LangChain  
- **Vector Store**: Qdrant (Local or Cloud)  
- **LLM**: OpenAI GPT Models  
- **UI**: Streamlit  
- **Embeddings**: OpenAI Embeddings  

**Workflow:**
""")

st.code("""
flowchart LR
    A[User Query] --> B[LLM Node]
    B --> C[Decision Node]
    C --> D[Tool Node (RAG / Weather)]
    D --> E[Context Retrieved]
    E --> F[LLM Node (with Context)]
    F --> G[Final Response]
    G --> H[User Output]
    C --> I[Direct Response]
    I --> H
""", language="mermaid")

st.markdown("""
<div align="center">
  <img src="assets/graphs/node_graph.png" alt="LangGraph Node Workflow" width="600"/>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("### 📊 Use Cases")

st.markdown("""
1. **Research Assistant** — Upload academic papers and ask context-based questions  
2. **Document Q&A** — Extract structured information from reports or documents  
3. **Travel Planning** — Query live weather data for destinations  
4. **Knowledge Management** — Build and query your personalized knowledge base  
""")

st.divider()

st.markdown("### 🔒 Privacy & Security")

st.markdown("""
- All uploaded data stays local in Qdrant  
- Only model API calls go to OpenAI (no third-party data sharing)  
- Users can clear data anytime  
""")

st.divider()

st.markdown("### ⚙️ Performance Metrics")

st.markdown("""
| Metric | Average |
|:--------|:---------|
| Response Time | 2–5 seconds |
| Document Processing | ~1 second per page |
| Vector Retrieval | <1 second |
| Concurrent Users | Optimized for single-user sessions |
""")

st.divider()

st.markdown("### 🧭 Future Enhancements")

st.markdown("""
- [ ] Multi-modal document support (images, tables)
- [ ] Advanced filtering and semantic search options
- [ ] Exportable conversation history
- [ ] Support for custom embedding models
- [ ] REST API for programmatic access
""")


st.divider()

st.markdown("""
**Version:** 1.0.0  
**Last Updated:** November 2024  
**Built with ❤️ by Your Team**
""")

st.divider()

st.markdown("### 💡 Example Queries")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Document Questions:**
    - "Summarize the main findings in the uploaded paper"
    - "What methodology was used in this report?"
    - "Find all mentions of deep learning models"
    """)

with col2:
    st.markdown("""
    **Weather Questions:**
    - "What's the weather like in Tokyo?"
    - "Should I bring an umbrella in London today?"
    - "Is it cold in New York right now?"
    """)
