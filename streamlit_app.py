import streamlit as st
import asyncio
import json
import os
from typing import List, Dict, Any
import time

from RagService import RagService, AppConfig


class StreamlitRagApp:
    def __init__(self):
        self.rag_service = None
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "rag_service" not in st.session_state:
            st.session_state.rag_service = None
        if "service_initialized" not in st.session_state:
            st.session_state.service_initialized = False

    async def initialize_rag_service(self, config_path: str = "appsettings.json"):
        """Initialize the RAG service"""
        try:
            config = AppConfig.from_json(config_path)
            rag_service = RagService(config)
            await rag_service.initialize_async()
            return rag_service
        except Exception as e:
            st.error(f"Failed to initialize RAG service: {str(e)}")
            return None

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.title("RAG Chat Configuration")
        
        # Configuration file selection
        config_file = st.sidebar.text_input(
            "Configuration File Path", 
            value="appsettings.json",
            help="Path to your appsettings.json file"
        )
        
        # Initialize service button
        if st.sidebar.button("Initialize RAG Service"):
            with st.spinner("Initializing RAG service..."):
                try:
                    # Run async function in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    rag_service = loop.run_until_complete(
                        self.initialize_rag_service(config_file)
                    )
                    loop.close()
                    
                    if rag_service:
                        st.session_state.rag_service = rag_service
                        st.session_state.service_initialized = True
                        st.sidebar.success("RAG service initialized successfully!")
                    else:
                        st.session_state.service_initialized = False
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
                    st.session_state.service_initialized = False

        # Service status
        if st.session_state.service_initialized:
            st.sidebar.success("✅ RAG Service Ready")
        else:
            st.sidebar.warning("⚠️ RAG Service Not Initialized")

        # Chat options
        st.sidebar.subheader("Chat Options")
        max_results = st.sidebar.slider(
            "Max Search Results", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Maximum number of documents to retrieve for context"
        )
        
        include_citations = st.sidebar.checkbox(
            "Include Citations", 
            value=True,
            help="Include source citations in responses"
        )
        
        # Clear chat button
        if st.sidebar.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        return max_results, include_citations

    def render_chat_interface(self, max_results: int, include_citations: bool):
        """Render the main chat interface"""
        st.title("🤖 RAG-Powered Chat Assistant")
        st.markdown("Ask questions and get answers based on your indexed documents!")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Check if service is initialized
            if not st.session_state.service_initialized or not st.session_state.rag_service:
                st.error("Please initialize the RAG service first using the sidebar.")
                return

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    try:
                        # Run async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        if include_citations:
                            response = loop.run_until_complete(
                                st.session_state.rag_service.ask_question_with_citations_async(
                                    prompt, max_results
                                )
                            )
                        else:
                            response = loop.run_until_complete(
                                st.session_state.rag_service.ask_question_with_rag_async(
                                    prompt, max_results
                                )
                            )
                        
                        loop.close()
                        
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    def render_document_search_demo(self):
        """Render a document search demo section"""
        if not st.session_state.service_initialized:
            return
            
        st.sidebar.subheader("Document Search Demo")
        
        if st.sidebar.button("Test Document Search"):
            test_query = st.sidebar.text_input("Test Query", value="What is artificial intelligence?")
            
            if test_query:
                with st.spinner("Searching documents..."):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        search_results = loop.run_until_complete(
                            st.session_state.rag_service.search_documents_async(test_query, 5)
                        )
                        
                        loop.close()
                        
                        st.sidebar.write(f"Found {len(search_results)} results:")
                        for i, result in enumerate(search_results, 1):
                            st.sidebar.write(f"{i}. **{result.document.title}** (Score: {result.score:.3f})")
                            st.sidebar.write(f"Content preview: {result.document.content[:100]}...")
                            
                    except Exception as e:
                        st.sidebar.error(f"Search error: {str(e)}")

    def run(self):
        """Main function to run the Streamlit app"""
        st.set_page_config(
            page_title="RAG Chat Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize session state
        self.initialize_session_state()

        # Render sidebar
        max_results, include_citations = self.render_sidebar()
        
        # Render document search demo
        self.render_document_search_demo()

        # Render main chat interface
        self.render_chat_interface(max_results, include_citations)

        # Add footer with instructions
        st.markdown("---")
        st.markdown("""
        **Instructions:**
        1. Configure your `appsettings.json` file with Azure OpenAI and Azure AI Search credentials
        2. Click "Initialize RAG Service" in the sidebar
        3. Start chatting with your documents!
        
        **Note:** Make sure your search index contains documents before asking questions.
        """)


def main():
    """Entry point for the Streamlit app"""
    app = StreamlitRagApp()
    app.run()


if __name__ == "__main__":
    main()
