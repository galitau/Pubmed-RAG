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
<img width="1870" height="418" alt="image" src="https://github.com/user-attachments/assets/1f9f8646-20b7-405a-b73e-aad3aa0c0f15" />
<img width="1839" height="828" alt="image" src="https://github.com/user-attachments/assets/f0c09757-d3df-4cf2-89e6-2752c1a98407" />
<img width="1839" height="833" alt="image" src="https://github.com/user-attachments/assets/f1aa6a1a-6466-4631-a34b-2ca433f5055f" />
<img width="1836" height="830" alt="image" src="https://github.com/user-attachments/assets/e9e60eb9-1157-4da8-99bb-7e6cefdf3b92" />
<img width="1832" height="849" alt="image" src="https://github.com/user-attachments/assets/fb72e410-d62a-4771-8e38-33052d9b4f87" />


https://github.com/user-attachments/assets/265c2072-bf4a-4fbf-ae5b-df229cba2fb5




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
