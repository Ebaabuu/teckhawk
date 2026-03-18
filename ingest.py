import os
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
    # We use TextLoader here because Markdown is just plain text
    loader = DirectoryLoader('./docs', glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No files found! Make sure you have .md files in the docs folder.")
        return

    print(f"Success: Loaded {len(documents)} document(s).")

    # 3. Split the manuals into bite-sized paragraphs
    # This ensures we only feed relevant chunks to the AI, not the whole book
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Success: Split documents into {len(chunks)} chunks.")

    # 4. Initialize the Google Text Embedding Model (The Concept Translator)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    # 5. Convert text to vectors and save to Local ChromaDB
    print("Translating text into Meaning Fingerprints... Please wait.")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("Phase A Complete! The local ChromaDB vector store is ready.")

if __name__ == "__main__":
    build_database()