import os
import time  # <-- NEW: We need this to pause the script
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Securely load the API key
load_dotenv()

def build_database():
    print("Starting Phase A: Reading IT Manuals...")

    # 2. Load all Markdown files from the 'docs' folder
    loader = DirectoryLoader('./docs', glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    if not documents:
        print("No files found! Make sure you have .md files in the docs folder.")
        return

    print(f"Success: Loaded {len(documents)} document(s).")

    # 3. Split the manuals into bite-sized paragraphs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Success: Split documents into {len(chunks)} chunks.")

    # 4. Initialize the Google Text Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    # 5. Initialize the database connection
    print("Connecting to local Chroma database...")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 6. The "Throttle" Loop (Bypasses the 429 Error)
    batch_size = 90  # Keep it under the 100 limit
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Translating batch {i//batch_size + 1} of {(len(chunks)//batch_size) + 1}...")
        
        # Add this specific batch to the database
        vectorstore.add_documents(batch)
        
        # If there are still more chunks to process, pause the script
        if i + batch_size < len(chunks):
            print("Sleeping for 60 seconds to respect Google's Free Tier speed limit...")
            time.sleep(60)
            
    print("Phase A Complete! The local ChromaDB vector store is ready.")

if __name__ == "__main__":
    build_database()