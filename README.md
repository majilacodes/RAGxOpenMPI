# Distributed Retrieval-Augmented Generation (RAG) System

## Overview

This project implements a **Distributed Retrieval-Augmented Generation (RAG) System** using Open MPI and LangChain. It processes large text datasets, generates embeddings in parallel, stores them in a Chroma vector database, and provides a scalable chatbot interface for question-answering. By leveraging parallel processing with Open MPI, the system significantly reduces embedding generation and retrieval times, making it suitable for real-time applications requiring efficient document processing and natural language understanding.

The system was developed to optimize performance, achieving a **72% reduction in embedding time (from 79 seconds to 22 seconds)** and an **84% reduction in retrieval time (from 0.99 seconds to 0.16 seconds)** through distributed computing techniques.

---

## Features

- **Parallel Processing**: Utilizes Open MPI to distribute document embedding generation across multiple processes, enhancing scalability and speed.
- **Embedding Generation**: Employs Hugging Face's `sentence-transformers/all-mpnet-base-v2` model for high-quality text embeddings.
- **Vector Storage**: Stores embeddings and metadata in a Chroma vector database for efficient similarity search.
- **Chatbot Interface**: Integrates Google’s Gemini-1.5-flash model with a history-aware retriever for context-aware question-answering.
- **Performance Monitoring**: Tracks CPU usage, memory consumption, and processing times for each operation, providing detailed metrics.
- **Graceful Shutdown**: Handles interruptions (e.g., Ctrl+C) with synchronized shutdown across all MPI processes.

---

## Prerequisites

- **Python 3.8+**
- **MPI Implementation**: Open MPI (tested with `mpirun`)
- **Operating System**: Linux or macOS (Windows support may require additional configuration)
- **Hardware**: Multi-core CPU recommended for parallel processing

### Required Libraries
Install dependencies via `pip`:
```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:
```
mpi4py
langchain
langchain-chroma
langchain-google-genai
langchain-huggingface
langchain-community
torch
psutil
python-dotenv
```

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/rag-distributed-system.git
   cd rag-distributed-system
   ```

2. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your Google API key for Gemini:
     ```
     GOOGLE_API_KEY=your-google-api-key
     ```

3. **Install Open MPI**:
   - On Ubuntu:
     ```bash
     sudo apt update
     sudo apt install openmpi-bin openmpi-common libopenmpi-dev
     ```
   - On macOS (via Homebrew):
     ```bash
     brew install openmpi
     ```

4. **Prepare Input Text**:
   - Place your text file (`context.txt`) in the project root directory. This file will be processed and used for question-answering.

5. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Application**:
   - Execute the script with `mpirun` to specify the number of processes (e.g., 4):
     ```bash
     mpirun -np 4 python script.py
     ```
   - Replace `script.py` with the actual filename of your code.

2. **Interact with the Chatbot**:
   - Once the system initializes, you’ll see:
     ```
     Start chatting with the AI! Type 'exit' to end the conversation.
     Type 'metrics' to view performance metrics so far.
     ```
   - Enter a question (e.g., "What is the main topic of the text?") and receive a response.
   - Type `metrics` to view performance statistics or `exit` to quit.

3. **Output**:
   - The system processes `context.txt`, generates embeddings, and stores them in a Chroma database (`db/chroma_db_with_metadata`).
   - Performance metrics are displayed after each query and upon exit.

---

## Performance Metrics

The system was optimized for efficiency using Open MPI’s parallel processing capabilities. Key improvements include:

- **Embedding Generation Time**: Reduced from **79 seconds to 22 seconds** (72% improvement).
- **Retrieval Time**: Reduced from **0.99 seconds to 0.16 seconds** (84% improvement).

These gains were achieved by distributing embedding tasks across multiple processes, minimizing computational bottlenecks, and optimizing retrieval with a similarity-based retriever.

Sample metrics output:
```
==================================================
FINAL PERFORMANCE SUMMARY (MPI - 4 PROCESSES)
==================================================
- Text-to-embeddings time (rank 0): 22.00 seconds
- Max embedding time across ranks: 22.50 seconds
- Average retrieval time: 0.16 seconds
- Total memory usage: 450.20 MB
==================================================
```

---

## Project Structure

```
rag-distributed-system/
├── script.py            # Main Python script
├── context.txt          # Input text file
├── .env                # Environment variables
├── db/                 # Chroma database directory
│   └── chroma_db_with_metadata/
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## How It Works

1. **Text Processing**: Loads `context.txt` and splits it into chunks (500 characters, 50-character overlap).
2. **Distributed Embedding**: Open MPI distributes chunks across processes, where each process generates embeddings using Hugging Face’s model.
3. **Vector Database**: Rank 0 aggregates embeddings and stores them in Chroma.
4. **Chatbot**: A history-aware retriever fetches relevant chunks, and Gemini generates concise answers (max 3 sentences).
5. **Metrics**: Tracks and logs performance at each step for analysis.

---

## Limitations

- **Hardware Dependency**: Performance scales with CPU cores; single-core systems see limited benefits.
- **MPI Overhead**: Small datasets may not justify parallelization due to communication latency.
- **Text File Size**: Extremely large files may require tuning of `batch_size` and `chunk_size` in `CONFIG`.

---

## Future Improvements

- Add support for GPU acceleration with proper device management.
- Implement dynamic load balancing for uneven document sizes.
- Extend to process multiple input files or real-time data streams.
- Integrate a web interface (e.g., Flask) for broader accessibility.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
