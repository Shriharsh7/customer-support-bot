import os
import logging
import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# Set up logging with immediate writing
logging.basicConfig(
    filename='support_bot_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    force=True  # Ensures any existing handlers are replaced and logging starts fresh
)
logger = logging.getLogger()

# Load models
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Find the most relevant section in the document
def find_relevant_section(query, sections, section_embeddings):
    stopwords = {"and", "the", "is", "for", "to", "a", "an", "of", "in", "on", "at", "with", "by", "it", "as", "so", "what"}
    
    # Semantic search
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, section_embeddings)[0]
    best_idx = similarities.argmax().item()
    best_section = sections[best_idx]
    similarity_score = similarities[best_idx].item()
    
    SIMILARITY_THRESHOLD = 0.4
    if similarity_score >= SIMILARITY_THRESHOLD:
        logger.info(f"Found relevant section using embeddings for query: {query}")
        return best_section
    
    logger.info(f"Low similarity ({similarity_score}). Falling back to keyword search.")
    
    # Keyword-based fallback search with stopword filtering
    query_words = {word for word in query.lower().split() if word not in stopwords}
    for section in sections:
        section_words = {word for word in section.lower().split() if word not in stopwords}
        common_words = query_words.intersection(section_words)
        if len(common_words) >= 2:
            logger.info(f"Keyword match found for query: {query} with common words: {common_words}")
            return section
    
    logger.info(f"No good keyword match found. Returning default fallback response.")
    return "I don’t have enough information to answer that."

# Process the uploaded file with detailed logging
def process_file(file, state):
    if file is None:
        logger.info("No file uploaded.")
        return [("Bot", "Please upload a file.")], state
    
    file_path = file.name
    if file_path.lower().endswith(".pdf"):
        logger.info(f"Uploaded PDF file: {file_path}")
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        logger.info(f"Uploaded TXT file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        logger.error(f"Unsupported file format: {file_path}")
        return [("Bot", "Unsupported file format. Please upload a PDF or TXT file.")], state
    
    sections = text.split('\n\n')
    section_embeddings = embedder.encode(sections, convert_to_tensor=True)
    state['document_text'] = text
    state['sections'] = sections
    state['section_embeddings'] = section_embeddings
    state['current_query'] = None
    state['feedback_count'] = 0
    state['mode'] = 'waiting_for_query'
    state['chat_history'] = [("Bot", "File processed. You can now ask questions.")]
    logger.info(f"Processed file: {file_path}")
    return state['chat_history'], state

# Handle user input (queries and feedback)
def handle_input(user_input, state):
    if state['mode'] == 'waiting_for_upload':
        state['chat_history'].append(("Bot", "Please upload a file first."))
        logger.info("User attempted to interact without uploading a file.")
    elif state['mode'] == 'waiting_for_query':
        query = user_input
        state['current_query'] = query
        state['feedback_count'] = 0
        context = find_relevant_section(query, state['sections'], state['section_embeddings'])
        if context == "I don’t have enough information to answer that.":
            answer = context
        else:
            result = qa_model(question=query, context=context)
            answer = result["answer"]
        state['last_answer'] = answer
        state['mode'] = 'waiting_for_feedback'
        state['chat_history'].append(("User", query))
        state['chat_history'].append(("Bot", f"Answer: {answer}\nPlease provide feedback: good, too vague, not helpful."))
        logger.info(f"Query: {query}, Answer: {answer}")
    elif state['mode'] == 'waiting_for_feedback':
        feedback = user_input.lower()
        state['chat_history'].append(("User", feedback))
        logger.info(f"Feedback: {feedback}")
        if feedback == "good" or state['feedback_count'] >= 2:
            state['mode'] = 'waiting_for_query'
            if feedback == "good":
                state['chat_history'].append(("Bot", "Thank you for your feedback. You can ask another question."))
                logger.info("Feedback accepted as 'good'. Waiting for next query.")
            else:
                state['chat_history'].append(("Bot", "Maximum feedback iterations reached. You can ask another question."))
                logger.info("Max feedback iterations reached. Waiting for next query.")
        else:
            query = state['current_query']
            context = find_relevant_section(query, state['sections'], state['section_embeddings'])
            if feedback == "too vague":
                adjusted_answer = f"{state['last_answer']}\n\n(More details:\n{context[:500]}...)"
            elif feedback == "not helpful":
                adjusted_answer = qa_model(question=query + " Please provide more detailed information with examples.", context=context)['answer']
            else:
                state['chat_history'].append(("Bot", "Please provide valid feedback: good, too vague, not helpful."))
                logger.info(f"Invalid feedback received: {feedback}")
                return state['chat_history'], state
            state['last_answer'] = adjusted_answer
            state['feedback_count'] += 1
            state['chat_history'].append(("Bot", f"Updated answer: {adjusted_answer}\nPlease provide feedback: good, too vague, not helpful."))
            logger.info(f"Adjusted answer: {adjusted_answer}")
    return state['chat_history'], state

# Function to return the up-to-date log file for download
def get_log_file():
    # Flush all log handlers to ensure log file is current
    for handler in logger.handlers:
        handler.flush()
    # Ensure the log file exists; if not, create an empty one.
    if not os.path.exists("support_bot_log.txt"):
        with open("support_bot_log.txt", "w", encoding="utf-8") as f:
            f.write("")
    logger.info("Log file downloaded by user.")
    return "support_bot_log.txt"

# Initial state
initial_state = {
    'document_text': None,
    'sections': None,
    'section_embeddings': None,
    'current_query': None,
    'feedback_count': 0,
    'mode': 'waiting_for_upload',
    'chat_history': [("Bot", "Please upload a PDF or TXT file to start.")],
    'last_answer': None
}

# Gradio interface
with gr.Blocks() as demo:
    state = gr.State(initial_state)
    
    with gr.Row():
        file_upload = gr.File(label="Upload PDF or TXT file")
        download_btn = gr.Button("Download Log")
        download_file = gr.File(label="Log File", interactive=False)
    
    chat = gr.Chatbot()
    user_input = gr.Textbox(label="Your query or feedback")
    submit_btn = gr.Button("Submit")

    # Process file upload
    file_upload.upload(process_file, inputs=[file_upload, state], outputs=[chat, state])

    # Handle user input and clear the textbox
    submit_btn.click(handle_input, inputs=[user_input, state], outputs=[chat, state]).then(lambda: "", None, user_input)
    
    # Set up download log button
    download_btn.click(fn=get_log_file, inputs=[], outputs=download_file)

demo.launch(share=True)