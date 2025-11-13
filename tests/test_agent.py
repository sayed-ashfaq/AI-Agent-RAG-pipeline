import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langsmith import traceable, expect, Client

# Import your modules
from src.tools.rag_agent.retriever import Retriever
from src.workflow.agent_workflow import ReActAgent
from src.tools.rag_agent.document_loader import DocumentService


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def langsmith_client():
    """LangSmith client for dataset management"""
    return Client()


@pytest.fixture(scope="module")
def test_retriever():
    """Create a retriever instance for testing"""
    return Retriever()


@pytest.fixture(scope="module")
def test_agent(test_retriever):
    """Create an agent instance with shared retriever"""
    return ReActAgent(retriever=test_retriever)


@pytest.fixture
def sample_documents():
    """Sample documents for testing RAG"""
    return [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "test_doc_1"}
        ),
        Document(
            page_content="LangGraph is built on top of LangChain and provides tools for building stateful agents.",
            metadata={"source": "test_doc_2"}
        ),
        Document(
            page_content="OpenAI provides GPT models including GPT-4 and GPT-3.5-turbo.",
            metadata={"source": "test_doc_3"}
        ),
    ]


@pytest.fixture
def mock_weather_response():
    """Mock weather API response"""
    return "Weather in Hyderabad: Temperature 28°C, Clear sky, Humidity 65%"


# ============================================================================
# RETRIEVER TESTS
# ============================================================================

class TestRetriever:
    """Test suite for Retriever functionality"""

    @pytest.mark.langsmith
    def test_retriever_initialization(self, test_retriever):
        """Test that retriever initializes correctly"""
        assert test_retriever is not None
        assert test_retriever.vector_store is not None
        assert test_retriever.embedding_model is not None
        assert test_retriever.collection_name is not None

    @pytest.mark.langsmith
    def test_add_documents_to_vstore(self, test_retriever, sample_documents):
        """Test adding documents to vector store"""
        collection_name = test_retriever.vector_store.collection_name
        initial_count = test_retriever.vector_store.client.count(collection_name=collection_name).count

        # Add documents
        test_retriever.add_documents_to_vstore(sample_documents)

        # Verify count
        final_count = test_retriever.vector_store.client.count(collection_name=collection_name).count
        assert final_count > initial_count
        assert final_count - initial_count >= len(sample_documents)

    @pytest.mark.langsmith
    @traceable(name="test_retriever_search")
    def test_retriever_search_quality(self, test_retriever, sample_documents):
        """Test retriever returns relevant documents"""
        # Add test documents
        test_retriever.add_documents_to_vstore(sample_documents)

        # Search for relevant content
        query = "What is LangChain?"
        results = test_retriever.call_retriever(query)

        # Assertions
        assert len(results) > 0, "Retriever should return at least one document"

        # Check if most relevant document is returned
        top_result = results[0].page_content
        assert "LangChain" in top_result or "framework" in top_result.lower()

        # Use LangSmith expect for semantic similarity
        expect.embedding_distance(
            prediction=top_result,
            reference="LangChain is a framework for developing applications"
        ).to_be_less_than(0.5)

    @pytest.mark.langsmith
    def test_retriever_handles_empty_query(self, test_retriever):
        """Test retriever handles edge cases"""
        results = test_retriever.call_retriever("")
        assert isinstance(results, list)

    @pytest.mark.langsmith
    def test_delete_document(self, test_retriever, sample_documents):
        """Test document deletion from vector store"""
        from uuid import uuid4

        # Add document with known UUID
        test_uuid = str(uuid4())
        test_retriever.vector_store.add_documents(
            documents=[sample_documents[0]],
            ids=[test_uuid]
        )

        # Delete document
        test_retriever.delete_document(test_uuid)

        # Verify deletion (this may need adjustment based on your Qdrant setup)
        # Note: Actual verification depends on Qdrant's API
        assert True  # Placeholder - implement actual verification


# ============================================================================
# AGENT WORKFLOW TESTS
# ============================================================================

class TestAgentWorkflow:
    """Test suite for AgentRAG workflow"""

    @pytest.mark.langsmith
    def test_agent_initialization(self, test_agent):
        """Test agent initializes with all components"""
        assert test_agent is not None
        assert test_agent.retriever is not None
        assert test_agent.llm is not None
        assert test_agent.tools is not None
        assert len(test_agent.tools) == 2  # RAG and Weather tools

    @pytest.mark.langsmith
    @traceable(name="test_agent_general_query")
    def test_agent_general_query(self, test_agent):
        """Test agent handles general queries without tools"""
        query = "What is your name?"
        response = test_agent.run(query)

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

        # Agent should respond without tool calls for this query
        # LangSmith will trace this automatically

    @pytest.mark.langsmith
    @traceable(name="test_agent_rag_tool_usage")
    def test_agent_uses_rag_tool(self, test_agent, sample_documents):
        """Test agent uses RAG tool for document queries"""
        # Add documents first
        test_agent.retriever.add_documents_to_vstore(sample_documents)

        query = "What is LangGraph according to the documents?"
        response = test_agent.run(query)

        assert response is not None
        assert len(response) > 0

        # Response should mention LangGraph or related content
        # Use expect for semantic checking
        expect.embedding_distance(
            prediction=response,
            reference="LangGraph is a tool for building stateful agents"
        ).to_be_less_than(0.6)

    @pytest.mark.langsmith
    @patch('langchain_community.utilities.OpenWeatherMapAPIWrapper')
    def test_agent_uses_weather_tool(self, mock_weather, test_agent, mock_weather_response):
        """Test agent uses weather tool for weather queries"""
        # Mock weather API
        mock_weather_instance = Mock()
        mock_weather_instance.run.return_value = mock_weather_response
        mock_weather.return_value = mock_weather_instance

        query = "What's the weather in Hyderabad?"
        response = test_agent.run(query)

        assert response is not None
        assert len(response) > 0

        # Check if response contains weather information
        assert any(word in response.lower() for word in ['weather', 'temperature', 'hyderabad'])

    @pytest.mark.langsmith
    def test_agent_tool_error_handling(self, test_agent):
        """Test agent handles tool errors gracefully"""
        # Query that might trigger tool but with edge case
        query = "What's in a document that doesn't exist?"

        try:
            response = test_agent.run(query)
            assert response is not None
        except Exception as e:
            pytest.fail(f"Agent should handle errors gracefully: {e}")

    # @pytest.mark.langsmith
    # def test_agent_prompt_switching(self, test_retriever):
    #     """Test agent with different prompt types"""
    #     agent_default = ReActAgent(retriever=test_retriever, prompt_type="default")
    #     agent_concise = ReActAgent(retriever=test_retriever, prompt_type="concise")
    #
    #     assert agent_default.system_prompt != agent_concise.system_prompt
    #
    #     # Both should work
    #     response1 = agent_default.run("Hello")
    #     response2 = agent_concise.run("Hello")
    #
    #     assert response1 is not None
    #     assert response2 is not None


# ============================================================================
# DOCUMENT LOADER TESTS
# ============================================================================

class TestDocumentLoader:
    """Test suite for DocumentService"""

    @pytest.mark.langsmith
    def test_document_service_initialization(self):
        """Test DocumentService initializes"""
        service = DocumentService()
        assert service is not None

    @pytest.mark.langsmith
    @pytest.mark.skipif(
        not Path("tests/fixtures/sample.pdf").exists(),
        reason="Sample PDF not found"
    )
    def test_process_pdf_file(self):
        """Test PDF processing"""
        service = DocumentService()
        docs = service.process_single_file("tests/fixtures/sample.pdf", "pdf")

        assert docs is not None
        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)

    @pytest.mark.langsmith
    def test_unsupported_file_type(self):
        """Test handling of unsupported file types"""
        service = DocumentService()

        with pytest.raises(Exception):
            service.process_single_file("test.xyz", "xyz")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""

    @pytest.mark.langsmith
    @traceable(name="test_full_rag_pipeline")
    def test_full_rag_pipeline(self, sample_documents):
        """Test complete RAG pipeline from document upload to query"""
        # Create fresh instances
        retriever = Retriever()
        agent = ReActAgent(retriever=retriever)

        # Add documents
        retriever.add_documents_to_vstore(sample_documents)

        # Query
        query = "Tell me about LangChain and LangGraph"
        response = agent.run(query)

        # Assertions
        assert response is not None
        assert len(response) > 10  # Should be a substantial response

        # Semantic similarity check
        expect.embedding_distance(
            prediction=response,
            reference="LangChain is a framework and LangGraph is for agents"
        ).to_be_less_than(0.7)

    @pytest.mark.langsmith
    def test_conversation_memory(self, test_agent):
        """Test agent maintains context across turns"""
        thread_id = "test_conversation_123"

        # First query
        response1 = test_agent.run("My name is Alice", thread_id=thread_id)

        # Second query referencing first
        response2 = test_agent.run("What is my name?", thread_id=thread_id)

        # Response should remember the name
        assert "alice" in response2.lower()


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and latency tests"""

    @pytest.mark.langsmith
    @traceable(name="test_response_time")
    def test_agent_response_time(self, test_agent):
        """Test agent responds within acceptable time"""
        import time

        start_time = time.time()
        response = test_agent.run("What is 2+2?")
        end_time = time.time()

        response_time = end_time - start_time

        assert response is not None
        # Should respond within 10 seconds
        assert response_time < 10.0, f"Response took {response_time:.2f}s"

        # Log metric to LangSmith
        expect.value(response_time).to_be_less_than(10.0)

    @pytest.mark.langsmith
    def test_retriever_performance(self, test_retriever, sample_documents):
        """Test retriever search performance"""
        import time

        # Add documents
        test_retriever.add_documents_to_vstore(sample_documents)

        # Measure retrieval time
        start_time = time.time()
        results = test_retriever.call_retriever("LangChain")
        end_time = time.time()

        retrieval_time = end_time - start_time

        assert len(results) > 0
        # Retrieval should be fast
        assert retrieval_time < 2.0, f"Retrieval took {retrieval_time:.2f}s"


# ============================================================================
# CUSTOM EVALUATORS (for use with LangSmith datasets)
# ============================================================================

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
  Custom evaluator for answer correctness.
  Can be used with LangSmith's client.evaluate()
  """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt = f"""
    Compare the predicted answer with the reference answer.

    Question: {inputs.get('question', inputs.get('query', ''))}
    Predicted: {outputs.get('answer', outputs)}
    Reference: {reference_outputs.get('answer', reference_outputs)}

    Is the predicted answer correct? Answer with just 'yes' or 'no'.
    """

    result = llm.invoke(prompt)
    return 'yes' in result.content.lower()


def retrieval_relevance_evaluator(inputs: dict, outputs: dict) -> bool:
    """
  Evaluator for retrieval relevance.
  Checks if retrieved documents are relevant to the query.
  """
    # This is a simple heuristic - in production, use LLM-based evaluation
    query = inputs.get('question', inputs.get('query', ''))
    documents = outputs.get('documents', [])

    if not documents:
        return False

    # Check if query terms appear in retrieved docs
    query_terms = set(query.lower().split())
    for doc in documents:
        doc_terms = set(doc.page_content.lower().split())
        if len(query_terms & doc_terms) > 0:
            return True

    return False


if __name__ == "__main__":
    # Run tests with LangSmith integration
    pytest.main([
        __file__,
        "-v",
        "--langsmith-output",
        "-s"
    ])