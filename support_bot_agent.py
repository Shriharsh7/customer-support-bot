import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# Set up logging to record actions
logging.basicConfig(filename='support_bot_log.txt', level=logging.INFO)


class SupportBotAgent:
    def __init__(self, document_path):
        # Load a pre-trained question-answering model
        self.qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        
        # Set up an embedding model for finding relevant sections
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the document text and split it into sections (by paragraphs)
        self.document_text = self.load_document(document_path)
        self.sections = self.document_text.split('\n\n')

        # Generate embeddings for all sections to enable fast similarity search.
        self.section_embeddings = self.embedder.encode(self.sections, convert_to_tensor=True)
        logging.info(f"Loaded document: {document_path}")

    def load_document(self, path):
        """Loads and extracts text from a given document (TXT or PDF) and logs it in the log file."""
            
        if path.lower().endswith(".txt"):
            file_type = "Text File"
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()
        
        elif path.lower().endswith(".pdf"):
            file_type = "PDF File"
            text = ""
            with open(path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        
        else:
            file_type = "Unsupported Format"
            logging.error(f"Unsupported file format: {path}")
            raise ValueError("Unsupported file format. Please provide a TXT or PDF file.")
        
        # Log file type detection
        logging.info(f"Loaded {file_type}: {path}")
        
        return text

    def find_relevant_section(self, query):
        """
        First tries semantic similarity (sentence-transformers).
        If similarity is too low, falls back to keyword search with stricter matching using stopword filtering.
        """

        # Predefined list of common stopwords (you can expand this list)
        stopwords = {"and", "the", "is", "for", "to", "a", "an", "of", "in", "on", "at", "with", "by", "it", "as", "so", "what"}
        
        # Semantic search part
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.section_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_section = self.sections[best_idx]
        similarity_score = similarities[best_idx].item()
        
        # Threshold for semantic search confidence
        SIMILARITY_THRESHOLD = 0.4

        if similarity_score >= SIMILARITY_THRESHOLD:
            logging.info(f"Found relevant section using embeddings for query: {query}")
            return best_section

        logging.info(f"Low similarity ({similarity_score}). Falling back to keyword search.")
        
        # Keyword-based fallback search with stopword filtering
        query_words = {word for word in query.lower().split() if word not in stopwords}
        for section in self.sections:
            section_words = {word for word in section.lower().split() if word not in stopwords}
            common_words = query_words.intersection(section_words)
            
            # Only consider it a match if there are at least 2 significant words overlapping
            if len(common_words) >= 2:
                logging.info(f"Keyword match found for query: {query} with common words: {common_words}")
                return section

        logging.info(f"No good keyword match found. Returning default fallback response.")
        
        return "I don’t have enough information to answer that."


    def answer_query(self, query):
        """
        Answers a user query by:
        - Finding the most relevant section.
        - Using a question-answering model to extract the exact answer.
        """
        context = self.find_relevant_section(query)

        # If no relevant context is found, return a default response.
        if not context:
            answer = "I don’t have enough information to answer that."
        else:
            # Run the QA model to extract the most relevant answer span.
            result = self.qa_model(question=query, context=context, max_answer_len=50)
            answer = result["answer"]
        
        # Log the answer for transparency
        logging.info(f"Answer for query '{query}': {answer}")
        return answer

    def get_feedback(self, response):
        """
        Ask the user for manual feedback on the provided response.
        The user can enter:
        - 'good' (satisfied with the answer)
        - 'too vague' (needs more details)
        - 'not helpful' (needs a better answer)
        """
        
        feedback = input("Enter feedback (good, too vague, not helpful): ").strip().lower()
        logging.info(f"Feedback provided: {feedback}")
        return feedback

    def adjust_response(self, query, response, feedback):
          """
        Modifies the response based on user feedback.
        - If 'too vague', appends more context.
        - If 'not helpful', re-queries with a modified prompt.
        """
        
        if feedback == "too vague":
            context = self.find_relevant_section(query)
            adjusted_response = f"{response}\n\n(More details:\n{context[:500]}...)"
            
        elif feedback == "not helpful":
            adjusted_response = self.answer_query(query + " Please provide more detailed information with examples.")
            
        else:
            adjusted_response = response
        
        logging.info(f"Adjusted answer for query '{query}': {adjusted_response}")
        return adjusted_response

    def run(self):
        """
        Runs the bot in interactive mode.
        - Accepts user input.
        - Processes the query.
        - Asks for feedback.
        - Adjusts responses accordingly.
        """
        
        while True:
            query = input("Enter your query (or type 'exit' to quit): ")
            
            if query.lower() == 'exit':
                break
            logging.info(f"Processing query: {query}")

            # Generate an answer based on document context.
            response = self.answer_query(query)
            print(f"Initial Response: {response}")

            # Allow up to 2 iterations for feedback adjustments
            for _ in range(2):
                feedback = self.get_feedback(response)
                if feedback == "good":
                    break # Exit feedback loop as the user is satisfied.
                    
                # Adjust response and print updated answer.
                response = self.adjust_response(query, response, feedback)
                print(f"Updated Response: {response}")


if __name__ == "__main__":
    bot = SupportBotAgent("faq.txt")
    bot.run()
