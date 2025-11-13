import os
from langsmith import Client
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(verbose=True)


# ============================================================================
# TEST DATASET CREATION
# ============================================================================

class TestDatasetManager:
    """Manage test datasets in LangSmith"""

    def __init__(self):
        self.client = Client()

    def create_rag_evaluation_dataset(self, dataset_name: str = "AgentRAG-Evaluation"):
        """
        Create a dataset for RAG evaluation with example questions and answers.
        """

        # Sample Q&A pairs for RAG testing
        qa_pairs = [
            {
                "input": {"question": "What is LangChain?"},
                "output": {
                    "answer": "LangChain is a framework for developing applications powered by language models.",
                    "expected_documents": ["langchain_intro"]
                }
            },
            {
                "input": {"question": "How does LangGraph differ from LangChain?"},
                "output": {
                    "answer": "LangGraph is built on top of LangChain and provides tools for building stateful, multi-actor applications with cycles.",
                    "expected_documents": ["langgraph_features"]
                }
            },
            {
                "input": {"question": "What are the main components of a RAG system?"},
                "output": {
                    "answer": "A RAG system consists of a retriever (for fetching relevant documents) and a generator (LLM for creating responses based on retrieved context).",
                    "expected_documents": ["rag_architecture"]
                }
            },
            {
                "input": {"question": "What embedding models are commonly used with LangChain?"},
                "output": {
                    "answer": "Common embedding models include OpenAI embeddings, HuggingFace embeddings, and sentence-transformers.",
                    "expected_documents": ["embedding_models"]
                }
            },
            {
                "input": {"question": "How do you handle conversation memory in agents?"},
                "output": {
                    "answer": "Conversation memory can be handled using MemorySaver in LangGraph or various memory types in LangChain like ConversationBufferMemory.",
                    "expected_documents": ["memory_systems"]
                }
            },
        ]

        # Create dataset
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Evaluation dataset for AgentRAG system - RAG queries",
                # data_type="kv"
            )
            print(f"Created dataset: {dataset_name}")

            # Add examples
            for idx, qa in enumerate(qa_pairs):
                self.client.create_example(
                    inputs=qa["input"],
                    outputs=qa["output"],
                    dataset_id=dataset.id
                )
                print(f"Added example {idx + 1}/{len(qa_pairs)}")

            return dataset

        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Dataset {dataset_name} already exists. Retrieving...")
                return self.client.read_dataset(dataset_name=dataset_name)
            raise

    def create_weather_tool_dataset(self, dataset_name: str = "AgentRAG-Weather-Tool"):
        """
        Create dataset for testing weather tool functionality.
        """

        weather_queries = [
            {
                "input": {"question": "What's the weather in London?"},
                "output": {
                    "should_use_tool": True,
                    "tool_name": "get_weather_data",
                    "expected_location": "London"
                }
            },
            {
                "input": {"question": "Is it raining in Tokyo today?"},
                "output": {
                    "should_use_tool": True,
                    "tool_name": "get_weather_data",
                    "expected_location": "Tokyo"
                }
            },
            {
                "input": {"question": "Should I bring an umbrella in Paris?"},
                "output": {
                    "should_use_tool": True,
                    "tool_name": "get_weather_data",
                    "expected_location": "Paris"
                }
            },
        ]

        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Evaluation dataset for weather tool usage",
                # data_type="kv"
            )

            for idx, query in enumerate(weather_queries):
                self.client.create_example(
                    inputs=query["input"],
                    outputs=query["output"],
                    dataset_id=dataset.id
                )

            print(f"Created weather tool dataset with {len(weather_queries)} examples")
            return dataset

        except Exception as e:
            if "already exists" in str(e).lower():
                return self.client.read_dataset(dataset_name=dataset_name)
            raise

    def create_tool_selection_dataset(self, dataset_name: str = "AgentRAG-Tool-Selection"):
        """
        Create dataset for testing correct tool selection.
        """

        tool_selection_cases = [
            {
                "input": {"question": "What does the document say about transformers?"},
                "output": {
                    "expected_tool": "get_document_context",
                    "should_not_use": ["get_weather_data"]
                }
            },
            {
                "input": {"question": "What's the temperature outside?"},
                "output": {
                    "expected_tool": "get_weather_data",
                    "should_not_use": ["get_document_context"]
                }
            },
            {
                "input": {"question": "What is 2 + 2?"},
                "output": {
                    "expected_tool": None,  # Should answer directly
                    "should_not_use": ["get_document_context", "get_weather_data"]
                }
            },
        ]

        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Dataset for testing tool selection logic",
                # data_type="kv"
            )

            for case in tool_selection_cases:
                self.client.create_example(
                    inputs=case["input"],
                    outputs=case["output"],
                    dataset_id=dataset.id
                )

            print(f"Created tool selection dataset")
            return dataset

        except Exception as e:
            if "already exists" in str(e).lower():
                return self.client.read_dataset(dataset_name=dataset_name)
            raise


# ============================================================================
# ENVIRONMENT SETUP HELPER
# ============================================================================

def setup_langsmith_env():
    """
    Helper function to set up LangSmith environment variables.
    Call this before running tests.
    """

    required_vars = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_TEST_SUITE": "AgentRAG-Tests",
        "LANGSMITH_EXPERIMENT": f"test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    }

    for key, value in required_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"Set {key}={value}")

    # Check for API key
    if "LANGSMITH_API_KEY" not in os.environ:
        print("⚠️  WARNING: LANGSMITH_API_KEY not set!")
        print("Please set it: export LANGSMITH_API_KEY=<your-key>")
        return False

    print("✅ LangSmith environment configured")
    return True


# ============================================================================
# RUN EVALUATION ON DATASET
# ============================================================================

def run_evaluation_on_dataset(dataset_name: str, agent_function):
    """
    Run evaluation on a LangSmith dataset.

    Args:
        dataset_name: Name of the dataset in LangSmith
        agent_function: Function that takes input dict and returns output
    """
    from langsmith import traceable

    client = Client()

    @traceable
    def evaluate_wrapper(inputs: dict) -> dict:
        """Wrapper to make agent compatible with LangSmith evaluation"""
        question = inputs.get("question", "")
        result = agent_function(question)
        return {"answer": result}

    # Run evaluation
    results = client.evaluate(
        evaluate_wrapper,
        data=dataset_name,
        evaluators=[
            # Add your custom evaluators here
        ],
        experiment_prefix=f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        max_concurrency=2,
    )

    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Run this script to set up test datasets in LangSmith.
    """

    print("🚀 Setting up LangSmith test environment...\n")

    # Setup environment
    # if not setup_langsmith_env():
    #     print("❌ Failed to setup environment. Please configure API key.")
    #     exit(1)

    # Create datasets
    manager = TestDatasetManager()

    print("\n📊 Creating test datasets...\n")

    try:
        # Create RAG evaluation dataset
        rag_dataset = manager.create_rag_evaluation_dataset()
        print(f"✅ RAG dataset: {rag_dataset.name} (ID: {rag_dataset.id})\n")

        # Create weather tool dataset
        weather_dataset = manager.create_weather_tool_dataset()
        print(f"✅ Weather dataset: {weather_dataset.name} (ID: {weather_dataset.id})\n")

        # Create tool selection dataset
        tool_dataset = manager.create_tool_selection_dataset()
        print(f"✅ Tool selection dataset: {tool_dataset.name} (ID: {tool_dataset.id})\n")

        print("🎉 All datasets created successfully!")
        print("\nYou can now run tests with:")
        print("  pytest tests/test_agent_rag.py --langsmith-output\n")

    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        raise

