from uuid import uuid4
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import List
from langchain_core.documents import Document

from src.utils.model_loader import ModelLoader
from src.utils.config_loader import load_config
from src.tools.rag_agent.document_loader import DocumentService
from custom_logger import GLOBAL_LOGGER as logger
from exception_handler.agent_exceptions import QdrantServiceError
from pathlib import Path


class Retriever:
    def __init__(self):

        self.embedding_model = ModelLoader().load_embeddings()
        self.config = load_config()

        # using inmemory - can be upgraded to anytype using config settings
        self.client = QdrantClient(":memory:")

        self.collection_name = self.config['qdrant']['collection_name']

        # Create collection only if it doesn't exist
        self._create_collection_if_not_exists()

        self.vector_store = self._create_vector_store()
        self.retriever_instance = None

    def _create_collection_if_not_exists(self):
        """Create collection only if it doesn't already exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists. Reusing existing collection.")
                return

            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config['qdrant']["dim_size"],
                    distance=Distance.COSINE
                ),
            )
            logger.info(f"Created collection successfully: {self.collection_name}")
            return collection

        except Exception as e:
            logger.error(f"Failed to create/check collection: {e}")
            raise QdrantServiceError("Failed to create/check collection", e)

    def _create_vector_store(self):
        try:
            vector_store = QdrantVectorStore(
                client=self.client,
                embedding=self.embedding_model,
                collection_name=self.collection_name,
            )
            logger.info(f"Created vector store successfully: {self.collection_name}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise QdrantServiceError("Failed to create vector store", e)

    def load_retriever(self):
        try:
            if not self.retriever_instance:
                top_k = self.config['qdrant'].get('top_k', 5)
                self.retriever_instance = self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": top_k, "fetch_k": 20}
                )
            return self.retriever_instance
        except Exception as e:
            logger.error(f"Failed to load retriever: {e}")
            raise QdrantServiceError("Failed to load retriever", e)

    def add_documents_to_vstore(self, docs: List[Document]):
        """
        Add multiple documents to Qdrant.
        """
        try:
            uuids = [str(uuid4()) for _ in range(len(docs))]
            self.vector_store.add_documents(documents=docs, ids=uuids)
            logger.info(f"Added {len(docs)} documents to {self.collection_name}")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise QdrantServiceError(str(e))

    def delete_document(self, uuid: str):
        """
        Delete a document from Qdrant by ID.
        """
        try:
            self.vector_store.delete(ids=[uuid])
            logger.info(f"Deleted document with ID: {uuid}")
        except Exception as e:
            logger.error(f"Error deleting document {uuid}: {e}")
            raise QdrantServiceError(str(e))

    def call_retriever(self, query):
        try:
            retriever = self.load_retriever()
            output = retriever.invoke(query)
            logger.info("Retriever invoked successfully")
            return output

        except Exception as e:
            logger.error(f"Failed to call retriever: {e}")
            raise QdrantServiceError("Failed to call retriever", e)

    def delete_collection(self):
        """Delete the entire collection - useful for cleanup/reset"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise QdrantServiceError("Failed to delete collection", e)

    def get_collection_info(self):
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            logger.info(f"Retrieved collection info for: {self.collection_name}")
            return collection_info
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise QdrantServiceError("Failed to get collection info", e)


if __name__ == '__main__':
    print("=" * 25, 'Testing Retriever', "=" * 25)
    path = "S:\\Generative AI\\AI-Agent-RAG-pipeline\\data\\attention_is_all_you_need.pdf"
    retriever = Retriever()
    retrieved_docs = DocumentService().process_single_file(path, 'pdf')
    retriever.add_documents_to_vstore(retrieved_docs)
    print("=" * 25, 'Output', "=" * 25)
    print(retriever.call_retriever("What is meant by langchain?"))