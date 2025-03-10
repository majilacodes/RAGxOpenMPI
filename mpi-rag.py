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
from langchain_core.runnables import Runnable
from mpi4py import MPI
import numpy as np
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

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load environment variables from .env
load_dotenv()

# Define the persistent directory and text file path
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")
text_path = os.path.join(current_dir, "context.txt")

# Configuration
CONFIG = {
    "batch_size": 3,  # Even smaller batch size
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retriever_k": 3,
    "max_history_length": 10,
    "timeout": 30,  # Timeout for operations in seconds
    "embedding_dimension": 768  # Fallback dimension for embeddings
}

# Function to get current resource usage
def get_resource_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    cpu_percent = process.cpu_percent(interval=0.1)
    return memory_usage_mb, cpu_percent

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    logger.info(f"Rank {rank}: Received signal {sig}, shutting down...")
    if rank == 0:
        try:
            comm.bcast("exit", root=0)
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Fix 1: Use CPU for all processes to avoid MPS conflicts
def get_device():
    # Avoid MPS completely - use CPU for all processes
    logger.info(f"Rank {rank}: Using CPU device")
    return "cpu"

# Set environment variable to control device
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = get_device()

# Fix 2: Completely separate document processing by rank
def process_documents_by_rank(documents, batch_size=3):
    # Track resource usage
    start_memory, start_cpu = get_resource_usage()
    start_time = time.time()
    
    # Get total number of documents and broadcast to all ranks
    if rank == 0:
        total_docs = len(documents)
        logger.info(f"Processing {total_docs} documents with {size} processes")
    else:
        total_docs = None
    
    # Broadcast total documents count
    total_docs = comm.bcast(total_docs, root=0)
    
    # Calculate which documents this rank should process
    docs_per_rank = total_docs // size
    extra_docs = total_docs % size
    
    start_idx = rank * docs_per_rank + min(rank, extra_docs)
    extra_count = 1 if rank < extra_docs else 0
    my_doc_count = docs_per_rank + extra_count
    end_idx = start_idx + my_doc_count
    
    # Get documents for this rank
    my_documents = documents[start_idx:end_idx] if rank == 0 and start_idx < total_docs else []
    
    # If not rank 0, receive documents from rank 0
    if rank != 0:
        if my_doc_count > 0:
            my_documents = comm.recv(source=0, tag=rank)
    # If rank 0, send documents to other ranks
    else:
        for r in range(1, size):
            r_start = r * docs_per_rank + min(r, extra_docs)
            r_extra = 1 if r < extra_docs else 0
            r_count = docs_per_rank + r_extra
            r_end = r_start + r_count
            
            if r_start < total_docs:
                r_docs = documents[r_start:r_end]
                comm.send(r_docs, dest=r, tag=r)
    
    # Process documents in batches
    local_results = []
    
    for i in range(0, len(my_documents), batch_size):
        batch = my_documents[i:i+batch_size]
        batch_start_time = time.time()
        logger.info(f"Rank {rank}: Processing batch {i//batch_size + 1}/{(len(my_documents)+batch_size-1)//batch_size}")
        
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
            for j, embedding in enumerate(batch_embeddings):
                doc_idx = start_idx + i + j
                local_results.append((doc_idx, embedding, batch_texts[j], batch_metadatas[j]))
                
            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Log batch processing time
            batch_time = time.time() - batch_start_time
            batch_memory, batch_cpu = get_resource_usage()
            logger.info(f"Rank {rank}: Batch processed in {batch_time:.2f} seconds. Memory: {batch_memory:.2f} MB, CPU: {batch_cpu:.2f}%")
            
        except Exception as e:
            logger.error(f"Rank {rank}: Error embedding batch: {str(e)}")
    
    # Calculate and log metrics for this rank
    process_time = time.time() - start_time
    end_memory, end_cpu = get_resource_usage()
    
    logger.info(f"Rank {rank}: Processing completed in {process_time:.2f} seconds")
    logger.info(f"Rank {rank}: Memory usage: {end_memory:.2f} MB (change: {end_memory - start_memory:.2f} MB)")
    logger.info(f"Rank {rank}: CPU utilization: {end_cpu:.2f}%")
    
    # Gather results from all ranks
    all_results = comm.gather(local_results, root=0)
    
    # Gather metrics from all ranks
    metrics = {
        "rank": rank,
        "doc_count": len(my_documents),
        "processing_time": process_time,
        "memory_usage_mb": end_memory,
        "memory_change_mb": end_memory - start_memory,
        "cpu_percent": end_cpu
    }
    all_metrics = comm.gather(metrics, root=0)
    
    if rank == 0:
        # Combine results
        combined_results = []
        for rank_results in all_results:
            combined_results.extend(rank_results)
        
        # Sort by document index
        combined_results.sort(key=lambda x: x[0])
        
        # Extract embeddings, texts, and metadata
        embeddings = [result[1] for result in combined_results]
        texts = [result[2] for result in combined_results]
        metadatas = [result[3] for result in combined_results]
        
        # Analyze and log all rank metrics
        total_docs_processed = sum(m["doc_count"] for m in all_metrics)
        max_process_time = max(m["processing_time"] for m in all_metrics)
        avg_process_time = sum(m["processing_time"] for m in all_metrics) / len(all_metrics)
        total_memory = sum(m["memory_usage_mb"] for m in all_metrics)
        avg_memory = total_memory / len(all_metrics)
        max_memory = max(m["memory_usage_mb"] for m in all_metrics)
        
        logger.info(f"All ranks processed a total of {total_docs_processed} documents")
        logger.info(f"Max processing time across ranks: {max_process_time:.2f} seconds")
        logger.info(f"Average processing time per rank: {avg_process_time:.2f} seconds")
        logger.info(f"Total memory usage across all ranks: {total_memory:.2f} MB")
        logger.info(f"Average memory usage per rank: {avg_memory:.2f} MB")
        logger.info(f"Max memory usage across ranks: {max_memory:.2f} MB")
        
        return {
            "embeddings": embeddings, 
            "texts": texts, 
            "metadatas": metadatas,
            "rank_metrics": all_metrics,
            "total_time": process_time,
            "max_time": max_process_time,
            "avg_time": avg_process_time,
            "total_memory": total_memory
        }
    else:
        return None

# Fix 3: Fixed document addition with correct Chroma API
def add_documents_to_db(documents):
    if rank == 0:
        logger.info(f"Adding {len(documents)} documents to the database")
        
        start_time = time.time()
        start_memory, start_cpu = get_resource_usage()
        
        # Process documents
        results = process_documents_by_rank(documents, batch_size=CONFIG["batch_size"])
        
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
        
        embedding_time = results.get("total_time", 0)
        max_rank_time = results.get("max_time", 0)
        
        logger.info(f"PERFORMANCE SUMMARY:")
        logger.info(f"Total text to embedding time (rank 0): {embedding_time:.2f} seconds")
        logger.info(f"Maximum embedding time across all ranks: {max_rank_time:.2f} seconds")
        logger.info(f"Total database creation time: {db_time:.2f} seconds")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Rank 0 memory usage: {end_memory:.2f} MB (change: {end_memory - start_memory:.2f} MB)")
        logger.info(f"Rank 0 CPU utilization: {end_cpu:.2f}%")
        
        # Print a formatted summary for easy viewing
        print("\n" + "="*50)
        print(f"PERFORMANCE METRICS - MPI ({size} PROCESSES) TEXT TO EMBEDDINGS")
        print("="*50)
        print(f"Embedding generation time (rank 0): {embedding_time:.2f} seconds")
        print(f"Maximum embedding time across ranks: {max_rank_time:.2f} seconds")
        print(f"Database creation time: {db_time:.2f} seconds")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Rank 0 memory usage: {end_memory:.2f} MB")
        print(f"Total memory across ranks: {results.get('total_memory', 0):.2f} MB")
        print(f"Rank 0 CPU utilization: {end_cpu:.2f}%")
        print("="*50 + "\n")
        
        # Store metrics in db metadata for later reference
        db_metrics = {
            "embedding_time": embedding_time,
            "max_rank_time": max_rank_time,
            "db_creation_time": db_time,
            "total_processing_time": total_time,
            "memory_usage_mb": end_memory,
            "cpu_percent": end_cpu,
            "timestamp": datetime.datetime.now().isoformat(),
            "process_count": size,
            "rank_metrics": results.get("rank_metrics", [])
        }
        
        return db, db_metrics
    else:
        # Non-root ranks contribute to processing but don't interact with DB
        process_documents_by_rank(documents, batch_size=CONFIG["batch_size"])
        return None, None

# Fix 4: Main processing function
def process_text_file():
    # Only rank 0 loads the text file
    if rank == 0:
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
            if metrics:
                metrics["file_size_mb"] = file_size_mb
                metrics["chunks"] = len(split_docs)
            
            return db, metrics
        
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            return None, None
    else:
        # Non-root ranks just wait for document processing
        add_documents_to_db([])
        return None, None

# Fix 5: Add metrics to chatbot initialization
def initialize_chatbot(db, metrics):
    if rank == 0 and db:
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
            if metrics:
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

# Fix 6: Add metrics to chat function
def run_chat(components):
    # Only rank 0 handles the chat interface
    if rank == 0 and components:
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
                    print(f"CUMULATIVE PERFORMANCE METRICS (MPI - {size} PROCESSES)")
                    print("="*50)
                    print(f"Initial setup metrics:")
                    print(f"- Text-to-embeddings time (rank 0): {metrics.get('embedding_time', 0):.2f} seconds")
                    print(f"- Max embedding time across ranks: {metrics.get('max_rank_time', 0):.2f} seconds")
                    print(f"- Database creation time: {metrics.get('db_creation_time', 0):.2f} seconds")
                    print(f"- Rank 0 memory usage: {metrics.get('memory_usage_mb', 0):.2f} MB")
                    print(f"- Process count: {size}")
                    
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
            print(f"FINAL PERFORMANCE SUMMARY (MPI - {size} PROCESSES)")
            print("="*50)
            print(f"Initial processing:")
            print(f"- Text file size: {metrics.get('file_size_mb', 0):.2f} MB")
            print(f"- Number of chunks: {metrics.get('chunks', 0)}")
            print(f"- Text-to-embeddings time (rank 0): {metrics.get('embedding_time', 0):.2f} seconds")
            print(f"- Max embedding time across ranks: {metrics.get('max_rank_time', 0):.2f} seconds")
            print(f"- Database creation time: {metrics.get('db_creation_time', 0):.2f} seconds")
            print(f"- Total processing time: {metrics.get('total_processing_time', 0):.2f} seconds")
            print(f"- Rank 0 memory usage: {metrics.get('memory_usage_mb', 0):.2f} MB")
            print(f"- Process count: {size}")
            
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

    # Make sure other ranks wait during chat
    comm.Barrier()

# Fix 7: Add metrics to main function
def main():
    try:
        # Start timing the entire application
        total_start_time = time.time()
        start_memory, start_cpu = get_resource_usage()
        
        # Process text file and create database
        if rank == 0:
            logger.info(f"Starting RAG application (MPI version with {size} processes)")
            print("="*50)
            print(f"STARTING RAG APPLICATION (MPI VERSION WITH {size} PROCESSES)")
            print("="*50)
        
        # Process text file
        db, metrics = process_text_file()
        
        # Make sure all processes are synchronized
        comm.Barrier()
        
        if rank == 0 and not db:
            logger.error("Failed to process text file or create database")
            return
        
        # Initialize chatbot components
        components = initialize_chatbot(db, metrics)
        
        # Calculate setup time (for rank 0)
        if rank == 0:
            setup_time = time.time() - total_start_time
            current_memory, current_cpu = get_resource_usage()
            
            logger.info(f"Total setup time: {setup_time:.2f} seconds")
            logger.info(f"Memory usage after setup: {current_memory:.2f} MB (change: {current_memory - start_memory:.2f} MB)")
            logger.info(f"CPU utilization during setup: {current_cpu:.2f}%")
        
        # Run the chat interface
        run_chat(components)
            
    except Exception as e:
        logger.error(f"Rank {rank}: Error in main function: {str(e)}")
    finally:
        # Clean exit
        if rank == 0:
            logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()