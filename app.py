import streamlit as st
import os
import csv
import fitz
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import JinaEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to chunk text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
    )
    return text_splitter.split_text(text)

# Function to process PDF and create CSV
def process_pdf(pdf_file, output_filename, embeddings):
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'embedding'])
        
        for chunk in chunks:
            embedding = embeddings.embed_query(chunk)
            writer.writerow([chunk, embedding])
    
    return output_filename

# Function to load CSV and create FAISS index
def load_csv_and_create_index(csv_filename, embeddings):
    df = pd.read_csv(csv_filename)
    texts = df['text'].tolist()
    embeddings_list = df['embedding'].apply(eval).tolist()
    return FAISS.from_embeddings(zip(texts, embeddings_list), embeddings)

# Function to query Groq
def query_groq(prompt, chat_history, groq_client):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_msg, ai_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": prompt})
    
    completion = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
        temperature=0.7
    )
    return completion.choices[0].message.content

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    
    # Sidebar for API key input and PDF upload and processing
    with st.sidebar:
        st.title("Settings")
        st.subheader("API Keys")
        JINA_API_KEY = st.text_input("Enter Jina API Key:")
        GROQ_API_KEY = st.text_input("Enter Groq API Key:")
        
        st.subheader("PDF Processor")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                if JINA_API_KEY:
                    os.environ["JINA_API_KEY"] = JINA_API_KEY
                    embeddings = JinaEmbeddings()
                    with st.spinner("Processing PDF..."):
                        output_filename = f"{uploaded_file.name.split('.')[0]}.csv"
                        csv_path = process_pdf(uploaded_file, output_filename, embeddings)
                        st.success(f"PDF processed and saved as {output_filename}")
                        st.download_button(
                            label="Download CSV",
                            data=open(csv_path, 'rb').read(),
                            file_name=output_filename,
                            mime='text/csv'
                        )
                else:
                    st.error("Please enter valid Jina API Key.")
        
        st.subheader("Processed Files")
        csv_files = [f for f in os.listdir() if f.endswith('.csv')]
        selected_file = st.selectbox("Select a CSV file", csv_files)

    # Initialize Groq client
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
    else:
        st.error("Please enter valid Groq API Key.")
        groq_client = None

    # Main chat interface
    st.title("Lorem Ipsum")

    # Toggle for RAG vs. LLM-only
    use_rag = st.toggle("Search PDF", value=True)

    if selected_file or not use_rag:
        if use_rag and 'vectorstore' not in st.session_state:
            with st.spinner("Loading data..."):
                if JINA_API_KEY:
                    embeddings = JinaEmbeddings()
                    st.session_state.vectorstore = load_csv_and_create_index(selected_file, embeddings)
                else:
                    st.error("Please upload and process a PDF file, then select it from the sidebar to start chatting with RAG. Or disable RAG to chat with the LLM directly.")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'sources_visibility' not in st.session_state:
            st.session_state.sources_visibility = {}

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for user_msg, ai_msg in st.session_state.chat_history:
                st.markdown(f"**👨:** {user_msg}")
                st.markdown(f"**🤖:** {ai_msg}")
                if 'sources' in st.session_state and user_msg in st.session_state.sources:
                    if st.button(f"Show Source for '{user_msg}'", key=f"btn_{user_msg}"):
                        if st.session_state.sources_visibility.get(user_msg, False):
                            st.session_state.sources_visibility[user_msg] = False
                        else:
                            st.session_state.sources_visibility[user_msg] = True
                        
                        st.experimental_rerun()
                    
                    if st.session_state.sources_visibility.get(user_msg, False):
                        st.info(st.session_state.sources[user_msg])
        
        # User input
        user_input = st.text_input("Ask a question:", key="user_input", on_change=lambda: st.session_state.update({"submit_question": True}))
        
        if (st.session_state.get("submit_question", False) or st.button("Submit")) and user_input:
            with st.spinner("Thinking..."):
                if use_rag and 'vectorstore' in st.session_state and JINA_API_KEY:
                    embeddings = JinaEmbeddings()
                    # Retrieve relevant context
                    relevant_docs = st.session_state.vectorstore.similarity_search(user_input, k=4)
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Prepare prompt for Groq
                    prompt = f"Context: {context}\n\nQuestion: {user_input}\n\nAnswer:"
                    
                    # Store sources
                    if 'sources' not in st.session_state:
                        st.session_state.sources = {}
                    st.session_state.sources[user_input] = context
                    
                    # Query Groq
                    if groq_client:
                        response = query_groq(prompt, st.session_state.chat_history, groq_client)
                        st.session_state.chat_history.append((user_input, response))
                    else:
                        st.error("Please enter valid Groq API Key.")
                else:
                    st.error("Please upload and process a PDF file, then select it from the sidebar to start chatting with RAG. Or disable RAG to chat with the LLM directly.")
            
            # Clear the input box and reset the submit flag
            st.session_state.submit_question = False
            st.experimental_rerun()
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()
