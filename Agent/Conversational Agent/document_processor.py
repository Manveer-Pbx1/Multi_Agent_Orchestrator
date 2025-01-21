import os
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from dotenv import load_dotenv

load_dotenv()

def init_pinecone():
    """Initialize Pinecone client with the new API."""
    try:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
            
        pc = Pinecone(api_key=api_key)
        index_name = "agent-rag"
        
        # Verify index exists before trying to connect
        available_indexes = [index.name for index in pc.list_indexes()]
        print(f"Available indexes: {available_indexes}")
        
        if index_name not in available_indexes:
            raise ValueError(f"Index '{index_name}' not found. Available indexes: {available_indexes}")
        
        index = pc.Index(name=index_name)  # Use explicit name parameter
        print(f"Successfully connected to index: {index_name}")
        return index
    except Exception as e:
        print(f"Pinecone initialization error: {str(e)}")
        raise

def process_document(file_path: str) -> List[str]:
    """Process PDF or DOCX files and return chunks."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced chunk size for better granularity
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_index': i,
            'total_chunks': len(chunks),
            'source': os.path.basename(file_path)
        })
    
    return chunks

def store_document(file_path: str, user_id: str):
    """Process and store document in Pinecone."""
    try:
        index = init_pinecone()
        chunks = process_document(file_path)
        embeddings = OpenAIEmbeddings()
        
        print(f"Storing {len(chunks)} chunks in index")
        
        # Initialize vector store with explicit initialization
        vectorstore = LangchainPinecone.from_existing_index(
            index_name="agent-rag",  # Use explicit index name
            embedding=embeddings,
            namespace=user_id
        )
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} of {len(chunks)//batch_size + 1}")
            vectorstore.add_documents(batch)
        
        print(f"Successfully stored all chunks in namespace: {user_id}")
        
        return {
            'chunk_count': len(chunks),
            'filename': os.path.basename(file_path),
            'vectorstore': vectorstore
        }
    except Exception as e:
        print(f"Document storage error: {str(e)}")
        raise

def query_document(query: str, user_id: str, k: int = 5):
    """Query the vector store for relevant chunks."""
    try:
        index = init_pinecone()
        embeddings = OpenAIEmbeddings()
        
        # Initialize vector store with existing index
        vectorstore = LangchainPinecone.from_existing_index(
            index_name="agent-rag",
            embedding=embeddings,
            namespace=user_id
        )
        
        print(f"Querying namespace: {user_id}")
        
        # Use direct similarity search for better results
        results = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            namespace=user_id  # Explicitly specify namespace
        )
        
        if not results:
            print("No results found in vector store")
            return []
            
        print(f"Found {len(results)} relevant chunks")
        
        # Format results with enhanced metadata and relevance filtering
        formatted_chunks = []
        for doc, score in results:
            formatted_chunks.append({
                'content': doc.page_content,
                'score': float(score),
                'metadata': {
                    'chunk_index': doc.metadata.get('chunk_index', 'N/A'),
                    'total_chunks': doc.metadata.get('total_chunks', 'N/A'),
                    'source': doc.metadata.get('source', 'Unknown'),
                }
            })
        
        return formatted_chunks
    except Exception as e:
        print(f"Query error: {str(e)}")
        raise
