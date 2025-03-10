import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import shutil
import logging
import gc
import torch
import os
import signal
import sys
import time
import psutil
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Define the persistent directory and text file path
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")
text_path = os.path.join(current_dir, "context.txt")

# Configuration
CONFIG = {
    "batch_size": 3,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retriever_k": 3,
    "max_history_length": 10,
    "timeout": 30,  # Timeout for operations in seconds
    "embedding_dimension": 768  # Fallback dimension for embeddings
}

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Device selection function
def get_device():
    # Using CPU for consistency with MPI version
    logger.info("Using CPU device")
    return "cpu"

# Set environment variable to control device
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = get_device()

# Function to get current resource usage
def get_resource_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    cpu_percent = process.cpu_percent(interval=0.1)
    return memory_usage_mb, cpu_percent

# Process documents function
def process_documents(documents, batch_size=3):
    logger.info(f"Processing {len(documents)} documents")
    
    # Process documents in batches
    results = {
        "embeddings": [],
        "texts": [],
        "metadatas": []
    }
    
    # Track resource usage
    start_memory, start_cpu = get_resource_usage()
    start_time = time.time()
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_start_time = time.time()
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)+batch_size-1)//batch_size}")
        
        # Extract content and metadata
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [doc.metadata for doc in batch]
        
        # Create embeddings
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device}
        )
        
        try:
            batch_embeddings = model.embed_documents(batch_texts)
            
            # Store results
            results["embeddings"].extend(batch_embeddings)
            results["texts"].extend(batch_texts)
            results["metadatas"].extend(batch_metadatas)
                
            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Log batch processing time
            batch_time = time.time() - batch_start_time
            batch_memory, batch_cpu = get_resource_usage()
            logger.info(f"Batch processed in {batch_time:.2f} seconds. Memory: {batch_memory:.2f} MB, CPU: {batch_cpu:.2f}%")
            
        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
    
    # Calculate and log total metrics
    total_time = time.time() - start_time
    end_memory, end_cpu = get_resource_usage()
    
    logger.info(f"Total embedding generation time: {total_time:.2f} seconds")
    logger.info(f"Memory usage: {end_memory:.2f} MB (change: {end_memory - start_memory:.2f} MB)")
    logger.info(f"Average CPU utilization: {end_cpu:.2f}%")
    
    # Add metrics to results
    results["metrics"] = {
        "total_time": total_time,
        "memory_usage_mb": end_memory,
        "cpu_percent": end_cpu
    }
    
    return results

# Add documents to database
def add_documents_to_db(documents):
    logger.info(f"Adding {len(documents)} documents to the database")
    
    start_time = time.time()
    start_memory, start_cpu = get_resource_usage()
    
    # Process documents
    results = process_documents(documents, batch_size=CONFIG["batch_size"])
    
    # Clear existing DB
    if os.path.exists(persistent_directory):
        shutil.rmtree(persistent_directory)
        logger.info("Cleared existing Chroma DB")
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(persistent_directory), exist_ok=True)
    
    # Initialize new DB
    db_start_time = time.time()
    
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )
    
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=model
    )
    
    # Add documents with pre-computed embeddings
    if results and results["texts"]:
        # Create document objects with pre-computed embeddings
        documents_with_embeddings = []
        
        for i in range(len(results["texts"])):
            # Create a Document object
            doc = Document(
                page_content=results["texts"][i],
                metadata=results["metadatas"][i]
            )
            documents_with_embeddings.append(doc)
        
        # Add documents to the vector store
        add_start_time = time.time()
        db.add_documents(
            documents=documents_with_embeddings,
            embeddings=results["embeddings"]
        )
        add_time = time.time() - add_start_time
        logger.info(f"Added {len(results['texts'])} documents to vector store in {add_time:.2f} seconds")
    
    db_time = time.time() - db_start_time
    logger.info(f"Database creation and population time: {db_time:.2f} seconds")
    logger.info("Database persisted automatically")
    
    # Calculate and log total metrics
    total_time = time.time() - start_time
    end_memory, end_cpu = get_resource_usage()
    
    embedding_time = results.get("metrics", {}).get("total_time", 0)
    
    logger.info(f"PERFORMANCE SUMMARY:")
    logger.info(f"Total text to embedding time: {embedding_time:.2f} seconds")
    logger.info(f"Total database creation time: {db_time:.2f} seconds")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Memory usage: {end_memory:.2f} MB (change: {end_memory - start_memory:.2f} MB)")
    logger.info(f"CPU utilization: {end_cpu:.2f}%")
    
    # Print a formatted summary for easy viewing
    print("\n" + "="*50)
    print("PERFORMANCE METRICS - TEXT TO EMBEDDINGS")
    print("="*50)
    print(f"Embedding generation time: {embedding_time:.2f} seconds")
    print(f"Database creation time: {db_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Memory usage: {end_memory:.2f} MB")
    print(f"CPU utilization: {end_cpu:.2f}%")
    print("="*50 + "\n")
    
    # Store metrics in db metadata for later reference
    db_metrics = {
        "embedding_time": embedding_time,
        "db_creation_time": db_time,
        "total_processing_time": total_time,
        "memory_usage_mb": end_memory,
        "cpu_percent": end_cpu,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return db, db_metrics

# Process text file
def process_text_file():
    if not os.path.exists(text_path):
        logger.error(f"Text file not found at {text_path}")
        return None, None
    
    try:
        start_time = time.time()
        
        # Load and split the text
        loader = TextLoader(text_path)
        text_docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        
        split_docs = splitter.split_documents(text_docs)
        logger.info(f"Split documents into {len(split_docs)} chunks")
        
        # Calculate size of text file
        file_size_mb = os.path.getsize(text_path) / (1024 * 1024)
        logger.info(f"Text file size: {file_size_mb:.2f} MB")
        
        # Add documents to the database
        db, metrics = add_documents_to_db(split_docs)
        
        # Add text file info to metrics
        metrics["file_size_mb"] = file_size_mb
        metrics["chunks"] = len(split_docs)
        
        return db, metrics
    
    except Exception as e:
        logger.error(f"Error processing text file: {str(e)}")
        return None, None

# Initialize chatbot
def initialize_chatbot(db, metrics):
    if db:
        try:
            start_time = time.time()
            start_memory, start_cpu = get_resource_usage()
            
            # Create a retriever
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": CONFIG["retriever_k"]}
            )
            
            # Create LLM
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            
            # Create prompts
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, just "
                "reformulate it if needed and otherwise return it as is."
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Create history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                llm=llm,
                retriever=retriever,
                prompt=contextualize_q_prompt
            )
            
            # Create QA chain
            qa_system_prompt = (
                "You are an assistant for question-answering tasks. Use "
                "the following pieces of retrieved context to answer the "
                "question. If you don't know the answer, just say that you "
                "don't know. Use three sentences maximum and keep the answer "
                "concise."
                "\n\n"
                "{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            
            # Calculate metrics
            init_time = time.time() - start_time
            end_memory, end_cpu = get_resource_usage()
            
            logger.info(f"Chatbot initialization time: {init_time:.2f} seconds")
            logger.info(f"Memory usage during initialization: {end_memory:.2f} MB (change: {end_memory - start_memory:.2f} MB)")
            logger.info(f"CPU utilization during initialization: {end_cpu:.2f}%")
            
            # Update metrics
            metrics["chatbot_init_time"] = init_time
            metrics["chatbot_init_memory_mb"] = end_memory
            metrics["chatbot_init_cpu"] = end_cpu
            
            return {
                "retriever": history_aware_retriever,
                "qa_chain": question_answer_chain,
                "metrics": metrics
            }
        
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            return None
    else:
        return None

# Chat function
def run_chat(components):
    if components:
        retriever = components["retriever"]
        qa_chain = components["qa_chain"]
        metrics = components.get("metrics", {})
        query_metrics = []
        
        logger.info("Starting chat. Type 'exit' to quit.")
        print("\nStart chatting with the AI! Type 'exit' to end the conversation.")
        print("Type 'metrics' to view performance metrics so far.")
        
        chat_history = []
        query_count = 0
        
        while True:
            try:
                query = input("\n> ")
                
                if query.lower() == "exit":
                    break
                    
                if query.lower() == "metrics":
                    print("\n" + "="*50)
                    print("CUMULATIVE PERFORMANCE METRICS")
                    print("="*50)
                    print(f"Initial setup metrics:")
                    print(f"- Text-to-embeddings time: {metrics.get('embedding_time', 0):.2f} seconds")
                    print(f"- Database creation time: {metrics.get('db_creation_time', 0):.2f} seconds")
                    print(f"- Memory usage: {metrics.get('memory_usage_mb', 0):.2f} MB")
                    
                    if query_metrics:
                        print("\nQuery metrics (average of {len(query_metrics)} queries):")
                        avg_retrieval = sum(qm['retrieval_time'] for qm in query_metrics) / len(query_metrics)
                        avg_total = sum(qm['total_time'] for qm in query_metrics) / len(query_metrics)
                        print(f"- Average retrieval time: {avg_retrieval:.2f} seconds")
                        print(f"- Average total query time: {avg_total:.2f} seconds")
                    print("="*50 + "\n")
                    continue
                
                # Echo user's input with proper formatting
                print(f"Human: {query}")
                
                # Track performance for this query
                query_count += 1
                query_start_time = time.time()
                start_memory, start_cpu = get_resource_usage()
                
                # Get relevant documents using the invoke method
                retrieval_start_time = time.time()
                docs = retriever.invoke({
                    "input": query,
                    "chat_history": chat_history
                })
                retrieval_time = time.time() - retrieval_start_time
                
                # Answer the question
                qa_start_time = time.time()
                answer = qa_chain.invoke({
                    "input": query,
                    "chat_history": chat_history,
                    "context": docs
                })
                qa_time = time.time() - qa_start_time
                
                # Calculate total query time
                total_query_time = time.time() - query_start_time
                end_memory, end_cpu = get_resource_usage()
                
                print(f"Assistant: {answer}")
                
                # Print performance metrics for this query
                print("\n" + "-"*50)
                print(f"QUERY #{query_count} PERFORMANCE METRICS:")
                print(f"Retrieval time: {retrieval_time:.2f} seconds")
                print(f"Answer generation time: {qa_time:.2f} seconds")
                print(f"Total processing time: {total_query_time:.2f} seconds")
                print(f"Memory usage: {end_memory:.2f} MB (change: {end_memory - start_memory:.2f} MB)")
                print(f"CPU utilization: {end_cpu:.2f}%")
                print("-"*50)
                
                # Store metrics for this query
                query_metric = {
                    "query_num": query_count,
                    "retrieval_time": retrieval_time,
                    "qa_time": qa_time,
                    "total_time": total_query_time,
                    "memory_mb": end_memory,
                    "cpu_percent": end_cpu
                }
                query_metrics.append(query_metric)
                
                # Update chat history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(SystemMessage(content=answer))
                
                # Limit chat history size
                if len(chat_history) > CONFIG["max_history_length"] * 2:
                    chat_history = chat_history[-CONFIG["max_history_length"]*2:]
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                print("An error occurred. Please try again.")
                
        # Print summary metrics at the end
        if query_count > 0:
            print("\n" + "="*50)
            print("FINAL PERFORMANCE SUMMARY")
            print("="*50)
            print(f"Initial processing:")
            print(f"- Text file size: {metrics.get('file_size_mb', 0):.2f} MB")
            print(f"- Number of chunks: {metrics.get('chunks', 0)}")
            print(f"- Text-to-embeddings time: {metrics.get('embedding_time', 0):.2f} seconds")
            print(f"- Database creation time: {metrics.get('db_creation_time', 0):.2f} seconds")
            print(f"- Total processing time: {metrics.get('total_processing_time', 0):.2f} seconds")
            print(f"- Memory usage: {metrics.get('memory_usage_mb', 0):.2f} MB")
            print(f"- CPU utilization: {metrics.get('cpu_percent', 0):.2f}%")
            
            print("\nQuery performance (average):")
            avg_retrieval = sum(qm['retrieval_time'] for qm in query_metrics) / len(query_metrics)
            avg_qa = sum(qm['qa_time'] for qm in query_metrics) / len(query_metrics)
            avg_total = sum(qm['total_time'] for qm in query_metrics) / len(query_metrics)
            avg_memory = sum(qm['memory_mb'] for qm in query_metrics) / len(query_metrics)
            avg_cpu = sum(qm['cpu_percent'] for qm in query_metrics) / len(query_metrics)
            
            print(f"- Average retrieval time: {avg_retrieval:.2f} seconds")
            print(f"- Average answer generation time: {avg_qa:.2f} seconds")
            print(f"- Average total query time: {avg_total:.2f} seconds")
            print(f"- Average memory usage: {avg_memory:.2f} MB")
            print(f"- Average CPU utilization: {avg_cpu:.2f}%")
            print("="*50)

# Main function
def main():
    try:
        # Start timing the entire application
        total_start_time = time.time()
        start_memory, start_cpu = get_resource_usage()
        
        # Process text file and create database
        logger.info("Starting RAG application (non-MPI version)")
        print("="*50)
        print("STARTING RAG APPLICATION (NON-MPI VERSION)")
        print("="*50)
        
        # Process text file
        db, metrics = process_text_file()
        
        if not db:
            logger.error("Failed to process text file or create database")
            return
        
        # Initialize chatbot components
        components = initialize_chatbot(db, metrics)
        
        # Calculate setup time
        setup_time = time.time() - total_start_time
        current_memory, current_cpu = get_resource_usage()
        
        logger.info(f"Total setup time: {setup_time:.2f} seconds")
        logger.info(f"Memory usage after setup: {current_memory:.2f} MB (change: {current_memory - start_memory:.2f} MB)")
        logger.info(f"CPU utilization during setup: {current_cpu:.2f}%")
        
        # Run the chat interface
        if components:
            run_chat(components)
        else:
            logger.error("Failed to initialize chatbot components")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()