# Zero-Leak-AI
🧠 ZeroLeak AI Your confidential documents, your questions — all offline.  is clear, powerful, and privacy-centered.
It instantly communicates:  
💡 AI-powered intelligence  
🔒 Zero data leakage 
🖥️ Fully offline execution  
🧑‍💼 Ideal for professionals handling sensitive documents

This project provides a local, private PDF chatbot powered by advanced NLP models. It allows users to upload a PDF document and then ask questions about its content or request a summary, all within an intuitive web-based interface.

✨ Features
PDF Text Extraction: Extracts text content from uploaded PDF files.

Intelligent Text Chunking: Divides the document into manageable chunks for efficient processing.

Retrieval-Augmented Generation (RAG): Uses a semantic search approach to find the most relevant sections of the PDF based on your query.

Powerful Language Model: Leverages the google/flan-t5-large model from Hugging Face for generating coherent and contextually relevant answers.

Summarization Capability: Automatically summarizes uploaded documents upon request.

Intuitive User Interface: Built with Gradio, featuring a clean, modern chat-like interface.

Real-time Chat Updates: User messages appear instantly in the chat history, followed by the bot's response.

Custom Styling: Enhanced UI with custom CSS for a polished look and feel.

⚙️ How It Works
The chatbot operates in several key steps:

PDF Processing (process_pdf):

When a PDF is uploaded, PyMuPDF (fitz) extracts all text.

langchain.text_splitter.RecursiveCharacterTextSplitter breaks the raw text into smaller, overlapping chunks.

SentenceTransformer (all-MiniLM-L6-v2) creates numerical embeddings for each chunk.

FAISS (Facebook AI Similarity Search) builds an efficient index from these embeddings, enabling fast retrieval of similar chunks.

The google/flan-t5-large model and its tokenizer are loaded using transformers.pipeline for text generation.

Question Answering (add_user_message, get_bot_response):

When a user types a question:

add_user_message immediately displays the user's input in the chat interface.

get_bot_response then takes over:

If the query contains "summarize", the model is prompted to summarize the initial part of the entire document.

Otherwise, the user's question is embedded using SentenceTransformer.

FAISS searches the document's chunk embeddings to find the most semantically similar chunks (context).

The retrieved context and the user's question are combined into a prompt for the flan-t5-large model.

The model generates a response based on this prompt, and the response is then appended to the chat history.

🚀 Setup and Installation
To run this chatbot locally, follow these steps:

Clone the repository (if applicable):

git clone <your-repo-url>
cd <your-repo-directory>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required libraries:

pip install PyMuPDF numpy faiss-cpu gradio transformers sentence-transformers langchain

Note: faiss-cpu is used for CPU-only environments. If you have a GPU and want to use it, you might need faiss-gpu and appropriate CUDA setup.

Run the application:

python your_chatbot_script_name.py

(Replace your_chatbot_script_name.py with the actual name of your Python file containing the code.)

📖 Usage
Upload PDF: Once the application launches in your browser, click the "📄 Upload PDF" button and select your desired PDF file.

Wait for Processing: The "Status" box will update to "✅ PDF uploaded and processed. You can now ask questions." once the document is ready.

Ask Questions: Type your question in the input box at the bottom.

For a general summary, type summarize.

For specific information, ask a direct question (e.g., "What are the main findings?", "Who are the key figures mentioned?").

Clear Chat: Use the "Clear Chat & Reset" button to clear the conversation and reset the application state, allowing you to upload a new PDF.

🛠️ Customization
Model: You can experiment with different SentenceTransformer models (for embeddings) or Hugging Face text generation models by changing model_name in the process_pdf function. Be aware that larger models require more computational resources.

Chunking Strategy: Adjust chunk_size and chunk_overlap in RecursiveCharacterTextSplitter to fine-tune how the document is broken down.

UI Styling: Modify the css string within gr.Blocks() to further customize the look and feel of the Gradio interface.
