# knowledge_assistant_bot

✅ Developed and deployed an intelligent chatbot for TechNova, enabling real-time, context-aware responses to product, service, and company-related queries using advanced NLP and LLM integration.

✅ Built using Python, Flask, LangChain, and Hugging Face LLMs, integrated with a domain-specific FAISS vector database for fast, accurate retrieval of information from TechNova’s official documents, including product specifications, FAQs, and company profile.

✅ Integrated a custom prompt template designed to align with TechNova’s tone and domain, optimizing language understanding and improving answer accuracy through structured context injection.

✅ Preprocessed and embedded company documents using sentence-transformers, ensuring high semantic fidelity and rapid context matching within the knowledge base.

✅ Improved the user experience by providing clear, professional, structured responses while reducing hallucinations through strict context-relevance enforcement.

![Alt text](https://github.com/ananya472/Soybot/blob/main/Screenshot%20(15).png?raw=true)

# Features
Context-aware Q&A based on company documents

Uses FAISS for fast document retrieval

Powered by Hugging Face LLMs for generating responses

Simple Flask web interface

Supports questions in both English and Hindi

# Architecture
Document Ingestion: Upload PDFs (e.g., Company Profile, FAQs, Product Specs) for processing.

Vector Store: Embeds document text into vectors using sentence-transformers, stored in FAISS.

LLM Integration: Uses Hugging Face models for generating answers.

Agentic Workflow: Routes questions based on keywords (e.g., "define", "calculate").

# Setup & Usage
Install Dependencies:
pip install -r requirements.txt

Ingest Documents:
Place your PDF files in the data/ folder and run:
python create_memory_for_llm.py

Run the Flask App:
python app.py
Visit http://localhost:5000 to use the chatbot.


