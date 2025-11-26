import os
import google.generativeai as genai
from dotenv import load_dotenv
import gradio as gr

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model Configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model
model = genai.GenerativeModel(model_name="gemini-2.5-flash", generation_config=generation_config)

# Chat session (persistent across user interactions)
chat_session = model.start_chat(history=[])

# Chat Function
def chat(user_input, history):
    """
    Handles chat interaction with Gemini API.
    - user_input: User's message
    - history: List of chat messages
    """
    # Send message to Gemini
    response = chat_session.send_message(user_input)
    model_response = response.text

    # Append to chat history
    history.append((user_input, model_response))

    return "", history

# Create Gradio Chat Interface
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Chat with Gemini 2.5 Flash")
    
    chatbot = gr.Chatbot(label="Gemini AI Chatbot")
    msg = gr.Textbox(label="Your message:", placeholder="Type your message here...")
    submit_btn = gr.Button("Send")

    # Functionality
    submit_btn.click(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])

# Launch UI
demo.launch()

