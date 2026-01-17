# PubMed RAG Researcher

## About This Project
I built this tool to make researching medical papers easier. Instead of manually reading through dozens of abstracts on PubMed, this app fetches the latest papers for you on the topic of your choice, uses AI to summarize the key findings, recommends the best paper to read, and lets you ask specific questions about the data.

It uses a technique called **RAG (Retrieval-Augmented Generation)** to make sure the AI's answers are based on real scientific papers, not just general knowledge.

## Features
* **Real-Time Search:** Connects directly to PubMed to find the most recent papers on your topic
* **Smart Summaries:** Uses Google Gemini to read the abstracts and write a clear summary, highlighting the most relevant paper for you
* **Chat with Data:** After the search, you can ask the AI follow-up questions (like "What were the side effects?") and it answers strictly based on the papers found
* **Chat History:** The app saves your conversation so you can scroll back and see previous answers

## Tech Stack
* **Python**
* **Streamlit** (for the web interface)
* **Google Gemini API** (LLM)
* **Metapub Library** (to access PubMed)

## Demo
https://github.com/user-attachments/assets/b21de8d1-b248-47c9-942c-9e269daee2eb

## How to Run It

1. Clone the repo
2. Install requirements
   
   ```pip install -r requirements.txt```
4. Create a .env file in the folder and add your Google Gemini key:
   
   ```GEMINI_API_KEY=your_key_here```
6. Run the App
   
   ```streamlit run app.py```

## What I Learned
* **Building RAG Applications:** I learned how to feed real-time data into an LLM to get accurate, factual responses
* **Session State:** I figured out how to use Streamlit's session state to keep the chat history and summary from disappearing when the page reloads
* **API Integration:** I learned how to work with the PubMed API to filter searches by date and relevance

## Future Improvements
* **Hybrid Search Implementation:** Currently, the application relies on keyword matching. Future updates will incorporate vector embeddings to enable semantic search, allowing retrieval of conceptually relevant papers that may differ in terminology
* **Full-Text Analysis Integration:** The system is currently limited to abstract analysis. I aim to expand the pipeline to ingest and parse full-text PDFs for open-access articles
* **Report Export Functionality:** Planned development includes a feature to generate and download comprehensive PDF reports containing the AI-synthesized summaries, citation lists, and chatbot messages
