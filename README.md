# PubMed RAG Researcher

## About This Project
I built this tool to make researching medical papers easier. Instead of manually reading through dozens of abstracts on PubMed, this app fetches the latest papers for you on the topic of your choice, uses AI to summarize the key findings, recommends the best paper to read, and lets you ask specific questions about the data.

It uses a technique called **RAG (Retrieval-Augmented Generation)** to make sure the AI's answers are based on real scientific papers, not just general knowledge.

## Features
* **Real-Time Search:** Connects directly to PubMed to find the most recent papers on your topic
* **Smart Summaries:** Uses Google Gemini to read the abstracts and write a clear summary, highlighting the most relevant paper for you
* **Chat with Data:** After the search, you can ask the AI follow-up questions (like "What were the side effects?") and it answers strictly based on the papers found
* **Chat History:** The app saves your conversation so you can scroll back and see previous answers
* **Smart Filtering:** Includes a toggle to filter for Free Full-Text (Open Access) only, ensuring you can read the papers found
* **Export Reports:** Generates and downloads professional PDF reports containing the research summary, references, and your entire chat history

## Tech Stack
* **Python**
* **Streamlit** (for the web interface)
* **Google Gemini API** (LLM)
* **Metapub Library** (to access PubMed)
* **FPDF** (for the PDF report generation)

## Demo
https://github.com/user-attachments/assets/b21de8d1-b248-47c9-942c-9e269daee2eb

## How to Run It

1. Clone the repo
2. Install requirements
   
   ```pip install -r requirements.txt```
4. Create a .env file in the root folder and add your Google Gemini Key. (Optional: Add an NCBI API Key to increase PubMed fetch speed from 3 to 10 requests/second):
   
   ```GEMINI_API_KEY=your_key_here```
   
   ```NCBI_API_KEY=your_ncbi_key_here  # Optional but recommended```
6. Run the App
   
   ```streamlit run app.py```

## What I Learned
* **Building RAG Applications:** I learned how to feed real-time data into an LLM to get accurate, factual responses
* **Session State:** I figured out how to use Streamlit's session state to keep the chat history and summary from disappearing when the page reloads
* **API Integration:** I learned how to work with the PubMed API to filter searches by date and relevance
* **Binary File Generation:** I learned how to convert raw text and chat history into binary streams to generate downloadable PDF files using FPDF

## Future Improvements
* **Semantic Search & Re-ranking:** Currently, the app relies on PubMed's keyword matching. Future updates will fetch a broader pool of papers and use Vector Embeddings (e.g., FAISS, ChromaDB) to locally re-rank them. This allows the user to find papers that match the meaning of their question, even if the specific keywords don't match
* **Data Visualization:** Add interactive charts to visualize publication trends over time
* **Full-Text Parsing:** Expand the pipeline to ingest and analyze full PDF text of open-access articles, rather than just abstract
