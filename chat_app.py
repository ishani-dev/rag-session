# Import required libraries
import os 
import streamlit as st  # For building the web app
from llama_index.llms.groq import Groq  # For using the LLM from Groq

# Import components from LlamaIndex for RAG (Retrieval-Augmented Generation)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings  

# Define a basic LLM-based response function (no retrieval)
def retrieve_generate(prompt):
    # Create a Groq LLM instance with model and API key
    llm = Groq(model="llama3-8b-8192", api_key="YOUR_API_KEY_HERE")
    # Get completion response for the given prompt
    response = llm.complete(prompt)
    return response

# Define a RAG-based function to give smarter responses using documents
def rag(prompt):
    # Load local documents from the 'data' folder
    documents = SimpleDirectoryReader("./data").load_data()

    # Configure global settings for LLM, embeddings, and parsing
    Settings.llm = Groq(model="llama3-8b-8192", api_key="YOUR_API_KEY_HERE")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.num_output = 512  # Max tokens for output
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)  # Split text into chunks

    # Create vector index using documents and embedding model
    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model, llm=Settings.llm)

    # Save the index to disk
    index.storage_context.persist()

    # Create a query engine from the index
    query_engine = index.as_query_engine()
    # Ask the question using the engine
    response = query_engine.query(prompt) 

    return response

# Streamlit UI starts here
st.title(f"**Chat :green[Assistant]** :sparkles:")  # Add styled title

# Initialize chat history if not already in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Take user input from the chat box
if prompt := st.chat_input("Ask any question here !"):
    # Display user's question
    st.chat_message("user").markdown(prompt)
    # Add it to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get answer using RAG
    response = rag(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
