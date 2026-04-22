import os
import streamlit as st
import pickle
import time
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables


def get_groq_api_key():
    secret_key = st.secrets.get("GROQ_API_KEY")
    if secret_key:
        return secret_key
    return os.getenv("GROQ_API_KEY")

st.title("Article Research Tool 📈")
st.sidebar.title("News Article URLs")

# User selects the number of URLs they want to input
num_urls = st.sidebar.number_input("How many URLs?", min_value=1, max_value=10, value=3, step=1)

urls = []
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():  # Only add non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

groq_api_key = get_groq_api_key()
llm = None

if groq_api_key:
    llm = ChatGroq(
        api_key=groq_api_key,
        temperature=0.9,
        model="llama3-70b-8192",
        max_tokens=500
    )
else:
    st.warning("Set GROQ_API_KEY in your local .env file or Streamlit secrets to use the app.")

if process_url_clicked:
    if llm is None:
        st.error("Missing GROQ_API_KEY. Add it before processing URLs.")
    elif not urls:
        st.error("Please enter at least one valid URL")
    else:
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            
            if not data:
                st.error("No content could be extracted from the provided URLs")
            else:
                # Split data
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=500,  # Further reduced to avoid exceeding model token limit
                    # chunk_overlap=60  # Increased overlap to preserve more context
                )
                main_placeholder.text("Text Splitter...Started...✅✅✅")
                docs = text_splitter.split_documents(data)
                
                if not docs:
                    st.error("No text content could be extracted from the URLs")
                else:
                    # Create embeddings and save to FAISS index
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                    time.sleep(2)

                    # Save FAISS index
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore, f)
                    st.success("URLs processed successfully!")
        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")

query = main_placeholder.text_input("Question: ")
if query:
    if llm is None:
        st.error("Missing GROQ_API_KEY. Add it before asking questions.")
    elif os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain.invoke({"question": query}, return_only_outputs=True)
            
            # Display answer
            st.header("Answer")
            st.write(result["answer"])
            
            # Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)
    else:
        st.warning("Please process some URLs first before asking questions")


#streamlit run main.py
