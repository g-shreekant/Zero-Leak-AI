import fitz  # PyMuPDF
import numpy as np
import faiss
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ========= GLOBAL STATE =========
# Global variables to store the extracted text, chunks, and model components.
raw_text = ""
chunks = []
encoder = None
index = None
generator = None

# ========= STEP 1: Process Uploaded PDF =========
def process_pdf(file_path):
    """
    Processes the uploaded PDF file: extracts text, chunks it, creates embeddings,
    and initializes the language model.
    """
    global raw_text, chunks, encoder, index, generator

    # Check if a file was actually uploaded.
    if file_path is None:
        # If no file, return an error message and keep input/send disabled.
        return "⚠️ Please upload a PDF file.", gr.update(value="", interactive=False), gr.update(interactive=False)

    try:
        # 1. Extract text from the PDF using PyMuPDF (fitz).
        doc = fitz.open(file_path)
        raw_text = "".join(page.get_text() for page in doc)
        doc.close() # Close the document after extraction

        # 2. Chunk text using RecursiveCharacterTextSplitter for better context management.
        # Chunks are 500 characters long with an overlap of 50 characters.
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(raw_text)

        # 3. Create embeddings for each chunk using SentenceTransformer.
        # These embeddings are used for semantic search (RAG).
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = encoder.encode(chunks)
        dim = embeddings.shape[1] # Dimension of the embeddings

        # Initialize FAISS index for efficient similarity search.
        index = faiss.IndexFlatL2(dim) # L2 distance for similarity
        index.add(np.array(embeddings)) # Add embeddings to the index

        # 4. Load the flan-t5-large model for text generation.
        # This model will answer questions based on the retrieved context.
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

        # On successful processing, return a success message and enable the chat input/send button.
        return "✅ PDF uploaded and processed. You can now ask questions.", gr.update(value="", interactive=True), gr.update(interactive=True)
    except Exception as e:
        # If any error occurs during processing, return an error message and keep chat input/send disabled.
        return f"❌ Error processing PDF: {e}", gr.update(value="", interactive=False), gr.update(interactive=False)

# ========= STEP 2: Handle User Questions =========

def add_user_message(user_input, chat_history):
    """
    Adds the user's message to the chat history immediately and clears the input box.
    The bot's response will be added in a subsequent step.
    """
    if not user_input.strip():
        # If input is empty, do nothing and keep the current state.
        return chat_history, gr.update(value="", interactive=True)

    # Append user message with None for the bot's response.
    # Gradio will display a loading indicator for the bot's part.
    chat_history.append((user_input, None))
    # Return the updated chat history and clear/disable the input box
    return chat_history, gr.update(value="", interactive=False)

def get_bot_response(chat_history):
    """
    Generates the bot's response based on the last user message and updates the chat history.
    This function is called after the user's message has been displayed.
    """
    # Get the last user message from the chat history
    user_input = chat_history[-1][0]

    # Check if the model and PDF content are ready.
    if not generator or not raw_text:
        # If not ready, update the last chat entry with an error message.
        chat_history[-1] = (user_input, "⚠️ Please upload a PDF first.")
        # Return updated chat history and re-enable the input box.
        return chat_history, gr.update(interactive=True)

    try:
        # Determine if the user wants a summary or a specific question answered.
        if "summarize" in user_input.lower():
            # For summarization, use the raw text (truncated to avoid exceeding model limits).
            summary_text = raw_text[:4000]
            prompt = f"Summarize this document:\n\n{summary_text}"
        else:
            # For question answering, retrieve relevant chunks using FAISS.
            q_emb = encoder.encode([user_input]) # Embed the user's question
            D, I = index.search(np.array(q_emb), 3) # Search for the top 3 most similar chunks
            context = "\n\n".join([chunks[i] for i in I[0]]) # Combine retrieved chunks into context

            # Construct the prompt with the retrieved context and the user's question.
            max_context_length = 1500 # Approximate character limit for context
            prompt = f"Context:\n{context[:max_context_length]}\n\nQuestion: {user_input}"

        # Generate the response using the loaded language model pipeline.
        output = generator(prompt, max_new_tokens=200, do_sample=False, temperature=0.7)[0]['generated_text']
        response = output.strip() # Clean up whitespace from the generated text

        # Update the last entry in chat_history with the actual bot response.
        chat_history[-1] = (user_input, response)
    except Exception as e:
        # If an error occurs during response generation, update the last entry with an error message.
        chat_history[-1] = (user_input, f"❌ Error generating response: {e}")

    # Return the updated chat history and re-enable the input box.
    return chat_history, gr.update(interactive=True)

# Function to clear chat and reset the application state.
def clear_all():
    """Resets all global state variables and clears the UI."""
    global raw_text, chunks, encoder, index, generator
    raw_text = ""
    chunks = []
    encoder = None
    index = None
    generator = None
    # Return empty chat history, reset status, and disable input/send button.
    return [], "Upload a PDF to begin...", gr.update(interactive=False), gr.update(interactive=False)

# ========= STEP 3: Launch Gradio Interface =========
# Define the Gradio Blocks interface with custom theming and CSS for enhanced UI.
with gr.Blocks(
    theme=gr.themes.Soft(), # Apply a soft, modern theme
    css="""
    /* General body styling */
    body {
        font-family: 'Inter', sans-serif; /* Use Inter font */
        background-color: #f0f2f5; /* Light grey background */
    }
    /* Main container styling */
    .gradio-container {
        max-width: 1000px; /* Max width for the app */
        margin: auto; /* Center the container */
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Subtle shadow for depth */
        border-radius: 12px; /* Rounded corners */
        background-color: #ffffff; /* White background for the app */
    }
    /* Title styling */
    h2 {
        text-align: center;
        color: #333; /* Dark grey text */
        margin-bottom: 20px;
        font-size: 1.8em; /* Larger font size */
        font-weight: 500; /* Bold font */
    }
    /* Button styling */
    .gr-button {
        border-radius: 8px !important; /* Rounded buttons */
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }
    /* File upload component styling */
    #file-upload-component {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
    }
    /* Textbox styling */
    .gr-textbox {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
    }
    /* Chatbot component styling */
    #main-chatbot {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        min-height: 400px; /* Ensure a decent height for the chat area */
        overflow-y: auto; /* Enable scrolling for chat history */
        background-color: #fdfdfd; /* Slightly off-white background for chat */
    }
    /* Bot message bubble styling */
    .message.bot {
        background-color: #e6f7ff; /* Light blue for bot messages */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    /* User message bubble styling */
    .message.user {
        background-color: #f0f0f0; /* Light gray for user messages */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    /* Message wrapper padding */
    .message-wrap {
        padding: 5px 0;
    }
    /* Column and Row spacing */
    .gr-column, .gr-row {
        gap: 20px; /* Space between columns/rows */
    }
    /* Specific styling for the status box */
    #upload-status-box {
        background-color: #e9ecef; /* Slightly darker background for status */
        color: #495057;
        font-weight: 500;
    }
    /* Styling for the send button */
    #send-button {
        background-color: #007bff; /* Primary blue color */
        color: white;
    }
    #send-button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    /* Styling for the clear button */
    #clear-button {
        background-color: #dc3545; /* Red color for clear */
        color: white;
    }
    #clear-button:hover {
        background-color: #c82333; /* Darker red on hover */
    }
    /* Styling for the message input box */
    #message-input {
        border: 1px solid #007bff !important; /* Add a blue border */
        border-radius: 8px !important; /* Keep rounded corners */
        padding: 10px !important; /* Add some padding */
    }
""") as demo:
    # Main title and subtitle
    gr.Markdown("## 🧠ZeroLeak AI -  Your confidential documents, your questions — all offline\nUpload your PDF and ask anything - 100% private", elem_classes=["title"])

    with gr.Row(): # Use a row to divide the layout into two main columns
        with gr.Column(scale=1): # Left column for controls (smaller)
            # File upload component
            file_upload = gr.File(label="📄 Upload PDF", type="filepath", file_types=[".pdf"], elem_id="file-upload-component")
            # Status textbox to show processing messages
            upload_status = gr.Textbox(label="Status", interactive=False, placeholder="Upload a PDF to begin...", elem_id="upload-status-box")
            # Button to clear chat and reset the application
            clear_button = gr.Button("Clear Chat & Reset", elem_id="clear-button")

        with gr.Column(scale=3): # Right column for the chatbot (larger)
            # Chatbot component to display conversation history
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500, # Fixed height for consistent layout
                show_copy_button=True, # Allow copying messages
                avatar_images=(None, "https://raw.githubusercontent.com/gradio-app/gradio/main/guides/assets/logo.png"), # User avatar (None) and example bot avatar
                elem_id="main-chatbot"
            )
            # Textbox for user input, initially disabled
            msg = gr.Textbox(
                label="Ask a question or type 'summarize'",
                placeholder="e.g. What are the main findings? Or type 'summarize'",
                scale=7, # Takes more space in the row
                container=False, # Remove default container for cleaner look
                interactive=False, # Initially disabled until PDF is processed
                elem_id="message-input"
            )
            # Send button for user input, initially disabled
            send_button = gr.Button("Send", scale=1, interactive=False, elem_id="send-button") # Takes less space, next to msg

    # Event Handlers:

    # When a file is uploaded, call process_pdf.
    # Outputs update the status, and enable/disable the message input and send button.
    file_upload.change(
        fn=process_pdf,
        inputs=file_upload,
        outputs=[upload_status, msg, send_button],
        show_progress=True # Show progress indicator during upload/processing
    )

    # When the clear button is clicked, reset the app state.
    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[chatbot, upload_status, msg, send_button]
    )

    # Chain the message submission:
    # 1. Add user message to chat and clear input.
    # 2. Then, get bot response and update chat.
    msg.submit(
        fn=add_user_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg],
        show_progress=False # No progress for immediate user message display
    ).then(
        fn=get_bot_response,
        inputs=[chatbot],
        outputs=[chatbot, msg],
        show_progress=True # Show progress for bot generation
    )

    send_button.click(
        fn=add_user_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg],
        show_progress=False # No progress for immediate user message display
    ).then(
        fn=get_bot_response,
        inputs=[chatbot],
        outputs=[chatbot, msg],
        show_progress=True # Show progress for bot generation
    )

# Launch the Gradio application.
demo.launch(show_api=False)
