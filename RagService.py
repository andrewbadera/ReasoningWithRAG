import asyncio
import json
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np

# Third-party imports (you'll need to install these)
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, VectorSearch,
    VectorSearchProfile, HnswAlgorithmConfiguration
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError


@dataclass
class AzureOpenAIConfig:
    endpoint: str = ""
    api_key: str = ""
    chat_model_id: str = ""
    embedding_model_id: str = ""


@dataclass
class AzureAISearchConfig:
    endpoint: str = ""
    api_key: str = ""
    index_name: str = ""


@dataclass
class AppConfig:
    azure_openai: AzureOpenAIConfig
    azure_ai_search: AzureAISearchConfig

    @classmethod
    def from_json(cls, config_path: str = "appsettings.json"):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            azure_openai=AzureOpenAIConfig(
                endpoint=data["AzureOpenAI"]["Endpoint"],
                api_key=data["AzureOpenAI"]["ApiKey"],
                chat_model_id=data["AzureOpenAI"]["ChatModelId"],
                embedding_model_id=data["AzureOpenAI"]["EmbeddingModelId"]
            ),
            azure_ai_search=AzureAISearchConfig(
                endpoint=data["AzureAISearch"]["Endpoint"],
                api_key=data["AzureAISearch"]["ApiKey"],
                index_name=data["AzureAISearch"]["IndexName"]
            )
        )


@dataclass
class DocumentRecord:
    title: str = ""
    content: str = ""


@dataclass
class DocumentSearchResult:
    document: DocumentRecord
    score: float = 0.0


class RagService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.index_name = config.azure_ai_search.index_name
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            azure_endpoint=config.azure_openai.endpoint,
            api_key=config.azure_openai.api_key,
            api_version="2024-02-01"
        )
        
        # Initialize Azure Search clients
        credential = AzureKeyCredential(config.azure_ai_search.api_key)
        self.search_index_client = SearchIndexClient(
            endpoint=config.azure_ai_search.endpoint,
            credential=credential
        )
        self.search_client = SearchClient(
            endpoint=config.azure_ai_search.endpoint,
            index_name=self.index_name,
            credential=credential
        )

    async def initialize_async(self):
        """Initialize the RAG service by creating search index if it doesn't exist"""
        await self._create_search_index_if_not_exists_async()

    async def _create_search_index_if_not_exists_async(self):
        """Create search index if it doesn't exist"""
        try:
            self.search_index_client.get_index(self.index_name)
            print(f"Search index '{self.index_name}' already exists.")
        except ResourceNotFoundError:
            print(f"Creating search index '{self.index_name}'...")
            
            fields = [
                SearchField("title", SearchFieldDataType.String, searchable=True, filterable=True),
                SearchField("chunk", SearchFieldDataType.String, searchable=True),
                SearchField(
                    "text_vector",
                    SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="my-vector-profile"
                )
            ]

            vector_search = VectorSearch(
                profiles=[VectorSearchProfile("my-vector-profile", "my-hnsw-config")],
                algorithms=[HnswAlgorithmConfiguration("my-hnsw-config")]
            )

            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )

            self.search_index_client.create_index(index)
            print(f"Search index '{self.index_name}' created successfully.")

    async def _generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding for the given text"""
        response = self.openai_client.embeddings.create(
            model=self.config.azure_openai.embedding_model_id,
            input=text
        )
        return response.data[0].embedding

    async def search_documents_async(self, query: str, limit: int = 3) -> List[DocumentSearchResult]:
        """Search for documents using vector similarity"""
        query_embedding = await self._generate_embedding_async(query)

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=limit,
            fields="text_vector"
        )

        search_results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=limit
        )

        results = []
        for result in search_results:
            document = DocumentRecord(
                title=result.get("title", ""),
                content=result.get("chunk", "")
            )

            results.append(DocumentSearchResult(
                document=document,
                score=result.get("@search.score", 0.0)
            ))

        return results

    async def ask_question_with_rag_async(self, question: str, max_results: int = 3) -> str:
        """Ask a question using RAG (Retrieval-Augmented Generation)"""
        # Search for relevant documents
        search_results = await self.search_documents_async(question, max_results)
        
        # Build context from search results
        context = "\n\n".join([
            f"Document Title: {r.document.title}\nContent: {r.document.content}"
            for r in search_results
        ])

        # Create prompt with context
        prompt = f"""
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please state that clearly.

Answer:"""

        # Get response from the model
        response = self.openai_client.chat.completions.create(
            model=self.config.azure_openai.chat_model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content or "I couldn't generate a response."

    async def ask_question_with_citations_async(self, question: str, max_results: int = 3) -> str:
        """Ask a question with citations included in the response"""
        # Search for relevant documents
        search_results = await self.search_documents_async(question, max_results)
        
        # Build context with citations
        context_with_citations = ""
        citations = []
        
        for index, result in enumerate(search_results, 1):
            context_with_citations += f"[{index}] Title: {result.document.title}\nContent: {result.document.content}\n\n"
            citations.append(f"[{index}] Title: {result.document.title}, Score: {result.score:.3f}")

        # Create prompt with context
        prompt = f"""
You are a helpful assistant that answers questions based on the provided context. Always include citations in your response.

Context:
{context_with_citations}

Question: {question}

Please provide a comprehensive answer based on the context above, including citations [1], [2], etc. where appropriate. If the context doesn't contain enough information to answer the question, please state that clearly.

Answer:"""

        # Get response from the model
        response = self.openai_client.chat.completions.create(
            model=self.config.azure_openai.chat_model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        
        answer = response.choices[0].message.content or "I couldn't generate a response."

        # Add citations at the end
        final_response = f"{answer}\n\nSources:\n" + "\n".join(citations)
        
        return final_response

    async def remove_document_async(self, id: str):
        """Remove a document from the search index"""
        self.search_client.delete_documents([{"id": id}])


async def main():
    """Main function to demonstrate RAG service functionality"""
    try:
        # Load configuration
        config = AppConfig.from_json("settings.json")
        
        # Initialize RAG service
        rag_service = RagService(config)
        await rag_service.initialize_async()
        
        # Example usage
        print("RAG Service initialized successfully!")

        # Ask questions
        question = "What can you tell me about Nereida?"
        print(f"\nQuestion: {question}")
        
        # Basic RAG
        print("\n--- Basic RAG Response ---")
        response = await rag_service.ask_question_with_rag_async(question)
        print(response)
        
        # RAG with citations
        print("\n--- RAG with Citations ---")
        response_with_citations = await rag_service.ask_question_with_citations_async(question)
        print(response_with_citations)
        
    except FileNotFoundError:
        print("Configuration file not found. Please create appsettings.json with your Azure credentials.")
        print("You can copy from ../SemanticKernel/appsettings.sample.json and fill in your values.")
    except Exception as e:
        print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     asyncio.run(main())