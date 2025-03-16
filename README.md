# Customer Support Bot with Document Training

## Overview:
Welcome to the **Customer Support Bot project!** This project was developed as an intelligent end-to-end solution to create a smart, agentic workflow in Python that:

- **Trains on a provided document** (a PDF or TXT FAQ file)
- **Answers customer queries** using a pre-trained question-answering model
- **Refines its responses iteratively** based on simulated user feedback ("too vague", "not helpful", or "good")
- **Logs all actions and decisions** for full transparency

The bot is deployed on **Hugging Face Spaces** for easy public access.  
👉 [Customer Support Bot on Hugging Face Spaces](https://huggingface.co/spaces/Shriharsh/Customer_Support_Bot_with_Document_Training)

---

## Features:

### Document Processing:
- **File Support:** Accepts both PDF and TXT files.
- **Text Extraction:**  
  - Uses **PyPDF2** to extract text from PDFs.  
  - Uses Python’s built-in `open()` function for TXT files.
- **Sectioning:** Splits the extracted text into sections (using double newlines as delimiters) for easier retrieval.
- **Embedding Generation:** Generates embeddings for each section using **Sentence-Transformers** (`all-MiniLM-L6-v2`).

### Query Handling:
- **Semantic Search:** Converts user queries into embeddings and retrieves the most relevant section based on cosine similarity.
- **Fallback Mechanism:** If the semantic similarity is too low, a keyword-based search (with stopword filtering) is used.
- **Answer Generation:** Uses a pre-trained QA model (`distilbert-base-uncased-distilled-squad`) to generate concise answers from the selected section.

### Feedback Loop:
- **User Feedback:**  
  - After receiving an answer, the user can mark it as “good”, “too vague”, or “not helpful.”
- **Response Adjustment:**  
  - For “too vague”: The bot appends additional context from the document to the answer.  
  - For “not helpful”: The bot rephrases the query by appending a request for more detailed information and re-queries the document.
- **Iteration Limit:** The feedback loop is limited to **2 iterations per query**.
- **Engineering Note:**  
  We intentionally kept the “not helpful” functionality simple to maintain clarity and meet the project deadline. Future upgrades may incorporate more advanced adjustments.

### Logging:
- **Real-time Logging:** Every key action (file upload, query processing, feedback, adjustments) is logged using Python’s `logging` module.
- **Log File Download:** The log file (`support_bot_log.txt`) can be downloaded directly from the web interface for review.

### Deployment:
- **Local Mode:**  
  Run the bot interactively using `support_bot_agent.py` on your terminal.
- **Hugging Face Spaces:**  
  The Gradio-based web app in `app.py` is deployed on Hugging Face Spaces for public access.

---

## Engineering Decisions

In building this Customer Support Bot, I carefully considered several key engineering decisions to create a solution that is both robust and is centred around the task at hand without overcomplicating:

- **Choosing Simplicity Over Complexity:**  
  I deliberately chose not to incorporate advanced frameworks like LangChain or FAISS. Given that our use case deals with small, FAQ-style documents, I recognized that a straightforward pipeline using Hugging Face Transformers, Sentence-Transformers, and PyPDF2 would be more than sufficient. This decision not only kept the codebase simple and easier to debug, but also ensured faster development and deployment under 1 week timeline.

- **Designing the Feedback Loop:**  
  I designed a feedback mechanism where the bot could adjust its responses based on user input. When a response is flagged as "too vague," the bot appends more context upto a limit, and if marked "not helpful," it reformulates the query for a more detailed answer. Although the “not helpful” function remains relatively simple for now, I discuss it more in future upgrade section. 

- **Implementing Detailed Logging:**  
  Every major action—from file uploads and query processing to feedback collection and response adjustments—is logged using Python’s `logging` module. Achieving a downloadable log file on Hugging Face Spaces was particularly challenging, but I dedicated extra effort to ensure that users can access a complete record of the bot's operations. This transparency is essential for both troubleshooting and continuous improvement. It was the most difficult part of the project.

---

## Development Process & Iterative Testing

  - Throughout development, I encountered several challenges, particularly with implementing an immediately updated and downloadable log file on Hugging Face Spaces. To address this, I built a smaller, focused application (**Upper Case Converter**) that took a user input string, converted it to uppercase, and wrote it to a log file. This mini-project allowed me to isolate and understand the logging mechanism in a simpler context.

  - The lessons learned from the Upper Case Converter were then applied to the main Customer Support Bot project, ensuring robust logging that is critical for transparency and troubleshooting.

  - [Upper Case Converter on Hugging Face Spaces](https://huggingface.co/spaces/Shriharsh/Upper_Case_Converter)
---

## How to Use the Bot

### Using the Hugging Face Spaces Web App
1. **Visit the Deployed App:**  
   [Customer Support Bot on Hugging Face Spaces](https://huggingface.co/spaces/Shriharsh/Customer_Support_Bot_with_Document_Training)
2. **Upload a Document:**  
   Use the file uploader to submit a PDF or TXT file (e.g., your FAQ file).
3. **Ask a Question:**  
   Enter your query in the chat interface. The bot will retrieve the most relevant section of the document and generate an answer.
4. **Provide Feedback:**  
   If the answer is not satisfactory, mark it as “too vague” or “not helpful” to trigger an adjusted response.
5. **Download Logs:**  
   Click the “Download Log” button to obtain the latest log file (`support_bot_log.txt`).
   
---

## Local Interactive Mode
### Running Locally
1. **Run the Local Script:**
   ```bash
   python support_bot_agent.py
   
2. **Follow the Prompts:**
    - The bot loads the document (e.g., faq.txt) and waits for your query.
    - Enter your question in the terminal.
    - After receiving an answer, provide feedback as prompted.
    - 
3. **View Logs:**
    All interactions are logged in **support_bot_log.txt.**

---
   
## Repository Structure

- **customer-support-bot/**
  - **app.py**: Gradio-based web application for Hugging Face Spaces.
  - **support_bot_agent.py**: Local interactive command-line version of the bot.
  - **faq.txt**: FAQ document in Q&A format used for training.
  - **requirements.txt**: Python dependencies required to run the project.
  - **support_bot_log.txt**: Log file generated during execution (dynamically updated).
  - **README.md**: Detailed project documentation (this file).


---
## Future Upgrades

As we look ahead to future iterations of the Customer Support Bot, there are several enhancements we can implement to improve accuracy, scalability, and overall user experience. Below are some potential upgrade directions along with their pros and cons.

### Enhanced "Not Helpful" Functionality
- **Integrate a More Powerful Model for Re-querying:**  
  Instead of simply appending additional context, a more advanced NLP model (e.g., GPT-3.5 or GPT-4) could be used to better understand the nuances of the user's feedback and generate a more refined re-query. e.g It may have synonyms of our embeddings
  - **Pros:** 
    - Produces more nuanced, context-aware, and detailed answers.  
    - Can better capture the subtleties of ambiguous queries and provide richer responses.
  - **Cons:** 
    - Higher computational cost and increased latency.  
    - Requires careful tuning to avoid overly verbose responses or overfitting to specific query patterns.

### Improved Embedding and Retrieval Mechanism
- **Adopt State-of-the-Art Embedding Models:**  
  Consider using more powerful embedding models to enhance the semantic search capabilities.
  - **Pros:** 
    - Improved matching accuracy for complex or nuanced queries.  
    - Better handling of context, leading to more relevant answer retrieval.
  - **Cons:**  
    - May introduce additional processing overhead and increased response time.  
    - Increased resource requirements and potential integration complexity.

### Scalable Architecture with LangChain
- **Leverage LangChain for Modular Processing:**  
  Incorporating LangChain can help streamline document processing and retrieval by providing:
  - **Advanced Document Loaders:**  
    - Automatic parsing of multiple document formats (e.g., PDFs, DOCX, HTML).
  - **Efficient Vector Stores:**  
    - Built-in integration with FAISS or ChromaDB for faster and more scalable similarity search.
  - **Chain-of-Thought Capabilities:**  
    - Support for multi-step reasoning that can improve the handling of complex queries.
  - **Pros:**  
    - Simplifies the integration of various processing components and standardizes the workflow.  
    - Enhances scalability, especially when dealing with larger documents or higher query volumes.
  - **Cons:**  
    - Adds an extra layer of abstraction, which can complicate debugging and fine-tuning.  
    - May increase the initial development and maintenance overhead.

### Time Complexity and Performance Considerations
- **Balancing Accuracy and Speed:**  
  With the potential integration of more advanced models and frameworks, it is important to assess:
  - **Latency vs. Relevance:**  
    - Advanced models may provide better answers but could slow down response times.
  - **Resource Overhead:**  
    - Increased computational requirements and potential bottlenecks with larger documents.
  - **Mitigation Strategies:**  
    - Consider techniques such as caching, asynchronous processing, and optimized vector searches to reduce latency while maintaining high accuracy.

By incorporating these future upgrades, we aim to further enhance the bot's ability to handle a wider range of queries more accurately, scale effectively to larger datasets, and provide a smoother user experience—all while keeping an eye on the trade-offs in terms of system complexity and performance.

---

## Conclusion

I am thrilled to share that the Customer Support Bot is now fully operational and Live and meets all the requirements! After many long hours and overcoming several hurdles especially during the deployment phase where I had a very hard time implementing a fully downloadable log file, I finally achieved a working solution that I can truly be proud of.

This project successfully:
- **Trains on user-provided documents** (PDFs and TXT files),
- **Handles customer queries** using a pre-trained QA model,
- **Iteratively refines responses** based on user feedback,
- **Logs every step** of its decision-making process for complete transparency, and
- **Deploys on Hugging Face Spaces** with a user-friendly interface.

I believe these engineering decisions reflect a thoughtful and pragmatic approach to solving real-world problems. I'm excited about the potential for future upgrades, including more advanced feedback handling and scalability improvements. 
