# Multi-Agent RAG Pipeline


## Overview
OmniAgent is an intelligent application designed with 

AI-Resume-Analyzer is an intelligent application designed to help job seekers improve their resumes and get personalized career advice. By combining natural language processing and machine learning techniques, the system analyzes resume content and provides tailored feedback, suggestions, and answers to career-related questions.

## topics should be covered in this
1. Overview
2. Images
3. Nodes
4. Logs
5. How it works
6. vector storage explanation
7. project structure

## Things to remember while developing
1. Langgraph and langchain usage
2. APIs integration
3. Qdrant vector database
4. Evaluation
5. Testcases: 
    - API Handling
    - LLM processing
    - retrieval logic

## Project Structure
```bash
AI-Agent-RAG-pipeline/
├── data/
│   # Sample documents & PDFs for RAG processing and evaluation
│
├── evaluations/
│   # LangSmith evaluation notebooks
│   └── langsmitheval.ipynb
│
├── notebooks/
│   # Notebooks for prototyping, testing, and experiments
│   ├── experiment.ipynb
│   ├── prototype.ipynb
│   ├── vector_service.ipynb
│   └── weather_tool.ipynb
│
├── pages/
│   # Front-end/Streamlit pages (if any)
│   └── 1_About.py
│
├── resources/
│   # Images/screenshots for README or evaluation documentation
│   # (e.g., LangSmith visuals, pytest results)
│
├── logs/
│   # Log files for debugging and workflow runs
│
├── src/
│   ├── config/
│   │   └── config.yaml               # App configuration (API keys, paths)
│   │
│   ├── prompt_library/
│   │   └── system_prompt.py          # System prompts for LLM agents
│   │
│   ├── tools/
│   │   ├── rag_agent/                # Core RAG agent logic
│   │   │   ├── document_loader.py
│   │   │   ├── retriever.py
│   │   │   └── schemas.py
│   │   │
│   │   └── wheather_agent/           # Weather tool logic (rename if needed)
│   │       ├── weather.py
│   │       └── schemas.py
│   │
│   ├── utils/
│   │   ├── config_loader.py          # Load configurations
│   │   └── model_loader.py           # Load models
│   │
│   └── workflow/
│       └── agent_workflow.py         # LangGraph pipeline orchestration
│   
├── tests/
│   # Unit tests for RAG, weather, and agents
│   ├── test_agent.py
│   ├── test_agent_rag.py
│   └── test_weather_api.py
│
├── custom_logger/
│   └── logger.py                      # Centralized logging utility
│
├── exception_handler/
│   └── agent_exceptions.py            # Custom exceptions for agents
│
├── vector_store/
│   ├── meta.json
│   └── collection/
│       # SQLite DBs for different vector collections
│       └── second-collection/
│           └── storage.sqlite
│
├── .env      # for storing environment vairables
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml
│
└── Chat_engine.py                 # Main chat engine orchestrating agents

```