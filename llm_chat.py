import streamlit as st
from llama_index.llms.groq import Groq

# Set your Groq API Key
GROQ_API_KEY = "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"

# Initialize the LLM
llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

st.title("🔍 Ask LLaMA3 (Groq) – No RAG, Just LLM")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Take user input
if prompt := st.chat_input("Ask me anything!"):
    # Show user prompt
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from LLM
    response = llm.complete(prompt)
    
    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
