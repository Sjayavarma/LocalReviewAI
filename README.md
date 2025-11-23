# LocalReviewAI
Offline RAG system for analyzing restaurant reviews using local LLMs. Uses TinyLlama for reranking and Gemma 2B for answer generation. Runs fully on-device with SBERT embeddings and ChromaDB.
📘 Offline Restaurant Review RAG System (TinyLlama → Gemma)

A fully offline, local Retrieval-Augmented Generation (RAG) system that answers questions about restaurant reviews using lightweight LLMs.
This project uses SBERT embeddings, ChromaDB vector search, TinyLlama for context reranking, and Gemma 2B for final answer generation — all running locally via Ollama.

This system requires no internet connection after installation.

🚀 Features

📂 Loads restaurant reviews from a CSV dataset

🧠 Generates embeddings using SBERT (all-MiniLM-L6-v2)

🔍 Vector search using ChromaDB (local persistent DB)

🦙 TinyLlama (1.1B) for reranking retrieved chunks

🐪 Gemma 2B for generating final answers

📡 Ollama used for running LLMs locally

🔐 100% Offline — works without internet

⚡ Lightweight and fast on CPU

🗂️ Dataset

The system uses the following CSV file:

realistic_restaurant_reviews.csv

Uploaded sample path:
/mnt/data/realistic_restaurant_reviews.csv

You can replace it with your own review dataset as long as columns include:

Title

Review

Rating

Date

🧰 Tech Stack
Component	Technology
Programming	Python
Embeddings	SBERT (all-MiniLM-L6-v2)
Vector DB	ChromaDB
Reranker	TinyLlama (Ollama)
Generator	Gemma 2B (Ollama)
Runtime	Ollama (Local)
Data Format	CSV
📦 Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2️⃣ Create & activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

3️⃣ Install dependencies
pip install chromadb sentence-transformers numpy pandas requests

4️⃣ Install and configure Ollama

Download from: https://ollama.com

Then pull required models:

ollama pull tinyllama
ollama pull gemma:2b

▶️ Run the Project
python rag_restaurant_reviews.py


You will see:

Ready! Ask questions (type 'q' to quit):


Example questions:

Do customers like the pizza?
What do people say about the service?
What are common complaints?
Which dishes get the best reviews?

📘 How It Works (RAG Pipeline)

Loads restaurant reviews from CSV

SBERT generates embeddings

ChromaDB stores and retrieves relevant documents

TinyLlama reranks the retrieved chunks

Gemma 2B generates the final answer

Everything runs locally with Ollama

💾 Project Structure
/AI Agent
│── rag_restaurant_reviews.py
│── realistic_restaurant_reviews.csv
│── chroma_db/            # auto-created
│── venv/                 # virtual environment
│── requirements.txt
│── README.md

📄 Example Query & Output

Question:
“What do customers say about pizza quality?”

Answer (Gemma 2B):
“Most customer reviews praise the pizza for fresh toppings, crisp crust, and balanced flavor. A few mention inconsistency on busy days.”

🔐 Offline Mode

This project is fully offline because:

Ollama runs models locally

SBERT embeddings are local

ChromaDB is local

No external API calls are made

You can disconnect Wi-Fi and the project still works.

📜 License

MIT License — free to use, modify, and distribute.

⭐ Acknowledgements

Ollama for local LLM runtime

Google Gemma team

ChromaDB developers

SBERT / SentenceTransformers team
![Uploading output.png…]()


