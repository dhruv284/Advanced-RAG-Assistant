
import streamlit as st
import tempfile
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from pathlib import Path

from pipeline.rag_pipeline import AdvancedRAGPipeline
from config import Config


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Research Chat",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Advanced RAG: Hybrid + Parent + Rerank + Reorder")

# ---------------------------
# Initialize Pipeline (Cached)
# ---------------------------
@st.cache_resource(show_spinner="Loading RAG Pipeline...")
def load_pipeline():
    config = Config()
    pipeline = AdvancedRAGPipeline(config=config)
    return pipeline


pipeline = load_pipeline()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None

# ---------------------------
# Handle Document Ingestion
# ---------------------------
if uploaded_files:
    if not st.session_state.ingested or st.button("Reprocess Documents"):
        with st.spinner("Saving files and processing documents..."):
            # Create temporary directory if not exists
            if st.session_state.temp_dir is None:
                st.session_state.temp_dir = tempfile.mkdtemp()

            saved_paths = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file to temp directory
                file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                saved_paths.append(file_path)

            # Ingest the saved files
            try:
                # Assuming your pipeline has or you can add this method
                # If not, you can modify AdvancedRAGPipeline to accept list of paths
                # pipeline.ingest_uploaded_files(saved_paths)   # or pipeline.ingest(saved_paths)
                pipeline.ingest(st.session_state.temp_dir)
                
                st.session_state.ingested = True
                st.success(f"✅ {len(uploaded_files)} document(s) processed successfully!")
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
                st.session_state.ingested = False

# ---------------------------
# Main UI
# ---------------------------
st.title("Research Chat")
st.markdown("**Hybrid • Parent-Child • Reranker • Long Context Reorder**")

# Show warning if no documents ingested
if not st.session_state.ingested:
    st.warning("👆 Please upload documents in the sidebar and process them to start chatting.")
    st.stop()  # Optional: prevent further execution until documents are ready

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question about your documents..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = pipeline.run(query)
                
                answer = result.get("answer", "Sorry, I couldn't generate an answer.")
                sources = result.get("sources", [])

                st.markdown(answer)

                # Show sources in expander
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}**")
                            # Show metadata if available
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.caption(f"File: {doc.metadata.get('source', 'Unknown')}")
                            st.write(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                            st.divider()
                else:
                    st.info("No sources found.")

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                answer = "An error occurred while processing your query."

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})