from langsmith import Client

docs = [
    "S:\\Generative AI\\AI-Agent-RAG-pipeline\\data\\test_doc_1.txt",
    "S:\\Generative AI\\AI-Agent-RAG-pipeline\\data\\test_doc_2.txt",
    "S:\\Generative AI\\AI-Agent-RAG-pipeline\\data\\test_doc_3.md"
]

def initiate_client():
    return Client()