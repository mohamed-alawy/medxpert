# chatbot_services.py

import os
import traceback
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
# --- Updated Imports ---
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- Initialization ---
# This checks for the Google API Key. For production, ensure this is set in your server's environment.
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"  # Directory to store the vector database
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.0-flash" # I see you're using gemini-2.0-flash, which is great.

# --- Services ---
# Initialize the core components once to be reused.
try:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, convert_system_message_to_human=True)
except Exception as e:
    print(f"Error initializing Google AI models: {e}")
    # Handle failed initialization gracefully
    embeddings, llm = None, None

def process_document(file_path: str, user_id: int, document_id: int, file_name: str):
    """
    Loads a document from a file path using the appropriate loader, splits it into chunks, 
    generates embeddings, and stores them in a user-specific collection in ChromaDB.
    """
    if not llm or not embeddings:
        print("AI services not initialized. Cannot process document.")
        return False
        
    print(f"Processing document: {file_name} for user {user_id}")
    try:
        # --- Loader Selection Logic ---
        file_extension = os.path.splitext(file_name)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return False

        docs = loader.load()

        # Split the document into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        if not splits:
            print(f"Warning: No text could be extracted from {file_name}.")
            return False

        # Add metadata to each chunk for filtering and citation
        for split in splits:
            split.metadata.update({
                "source": file_name,
                "doc_id": str(document_id)
            })

        # Embed and store the document chunks in a user-specific collection
        collection_name = f"user_{user_id}"
        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=collection_name
        )
        print(f"Successfully stored '{file_name}' in collection '{collection_name}'.")
        return True
    except Exception as e:
        print(f"Error processing document {file_name}: {e}")
        traceback.print_exc()
        return False

def get_rag_response(user_id: int, question: str):
    """
    Takes a user's question, retrieves relevant documents from their personal
    vector store, and generates a response using the Gemini LLM.
    """
    if not llm or not embeddings:
        return "Error: The AI Assistant is not configured correctly."

    try:
        collection_name = f"user_{user_id}"
        
        # Connect to the user's specific collection in the vector store
        user_vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=collection_name
        )

        # Create a retriever to fetch relevant documents
        retriever = user_vector_store.as_retriever(search_kwargs={"k": 4}) # Retrieve top 4 chunks

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" puts all retrieved chunks into the prompt context
            retriever=retriever,
            return_source_documents=True # Optional: to see which chunks were retrieved
        )

        # Run the chain and return the result
        response = qa_chain({"query": question})
        return response.get("result", "I could not find an answer in your documents.")

    except Exception as e:
        # Check for a specific error indicating the collection doesn't exist
        if "does not exist" in str(e):
            return "You haven't uploaded any documents yet. Please upload a document to start chatting."
        print(f"Error getting RAG response for user {user_id}: {e}")
        return "An error occurred while answering your question. Please ensure you have uploaded documents."

def get_rag_response(user_id: int, question: str):
    """
    Takes a user's question, retrieves relevant documents from their personal
    vector store, and generates a response using the Gemini LLM.
    """
    if not llm or not embeddings:
        return "Error: The AI Assistant is not configured correctly."

    try:
        collection_name = f"user_{user_id}"
        
        # Connect to the user's specific collection in the vector store
        user_vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=collection_name
        )

        # Create a retriever to fetch relevant documents
        retriever = user_vector_store.as_retriever(search_kwargs={"k": 4}) # Retrieve top 4 chunks

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" puts all retrieved chunks into the prompt context
            retriever=retriever
        )

        # Run the chain and return the result
        response = qa_chain({"query": question})
        return response.get("result", "I could not find an answer in your documents.")

    except Exception as e:
        print(f"Error getting RAG response for user {user_id}: {e}")
        return "An error occurred while answering your question. Please ensure you have uploaded documents."