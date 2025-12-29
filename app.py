import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from metapub import PubMedFetcher

# load environment variables from .env file
load_dotenv()

# Get key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Page Configuration
st.set_page_config(
    page_title="PUBMED RAG",
    page_icon="🧬",
    layout="wide"
)

# Sidebar
st.sidebar.title("Pubmed RAG with Gemini")
st.sidebar.markdown("### Settings")

# Date Filter
start_year = st.sidebar.slider("Search papers from year:", 2000, 2025, 2023)

# API Key Status indicator
if api_key:
    st.sidebar.success("API Key Loaded Securely")
else:
    st.sidebar.error("No API Key found!")


# Main Interface
st.title("Pubmed Retrieval-Augmented Generation: Automated Literature Review")
st.markdown(
    """
    *Retrieves clinical abstracts, filters by date, and generates consensus summaries using AI.*
    """
)

# Session State for Chat Memory
if 'abstracts' not in st.session_state: # Makes sure abstracts are stored when re-running
    st.session_state['abstracts'] = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "summary" not in st.session_state:
    st.session_state.summary = None
    
# Search interface
col1, col2 = st.columns([2, 1]) # col1 gets 2/3 of the screen, col2 gets 1/3

with col1:
    user_query = st.text_input("Enter Research Topic:", placeholder="e.g., Biocompatibility of 3D-printed bone scaffolds")

with col2:
    st.write("") # Spacer
    st.write("") 
    search_clicked = st.button("Run Literature Search", use_container_width=True)

# Logic for Search
if search_clicked:
    if not api_key:
        st.error("Please configure your API Key in .env file to proceed.")
    else:
        with st.spinner(f"Querying PubMed for '{user_query}' ({start_year}-Present)..."):
            try:
                # Configure AI
                genai.configure(api_key=api_key)
                
                # Fetch Data
                fetch = PubMedFetcher()
                pm_query = f"{user_query} AND {start_year}:3000[dp]" # dp for date of publication
                pmids = fetch.pmids_for_query(pm_query, retmax=10) # Returns a list of 10 PMIDs
                
                # Clears previous abstracts
                st.session_state['abstracts'] = ""
                found_count = 0
                paper_links = []
                
                # Process each PMID
                for pmid in pmids:
                    article = fetch.article_by_pmid(pmid)
                    if article.abstract:
                        st.session_state['abstracts'] += f"TITLE: {article.title}\n"
                        st.session_state['abstracts'] += f"AUTHORS: {str(article.authors)}\n"
                        st.session_state['abstracts'] += f"ABSTRACT: {article.abstract}\n"
                        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        st.session_state['abstracts'] += f"LINK: {link}\n"
                        st.session_state['abstracts'] += "-" * 20 + "\n"
                        found_count += 1
                
                if found_count > 0:
                    # Generate Summary
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"""
                    You are a Senior Biomedical Researcher. Don't fabricate information. If uncertain, state that the information is not available.
                    Make sure you make headings for each task and never mention "provided abstracts" in your answer.
                    Given the following abstracts from recent PubMed papers on a specific research topic, perform the following tasks:
                    1. Synthesize a summary of findings based ONLY on the provided abstracts.
                    2. Identify the single most relevant paper for the user to read. Start with "The most relevant paper to read is", then explain why you chose it in 1 sentence.
                    3. Provide a reference section with links to the papers cited.
                    
                    Here are the abstracts:
                    {st.session_state['abstracts']}
                    """
                    response = model.generate_content(prompt)
                    st.session_state.summary = response.text
                    st.session_state.messages = [] # Clear previous chat messages
                    st.success(f"Analysis Complete. Found {found_count} relevant papers.")
                else:
                    st.warning("No papers with abstracts found. Try a broader term.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Chat interface for follow-up questions
if st.session_state['abstracts']:
    # Displays the Summary (Always visible at the top)
    st.subheader("Summary")
    st.info(st.session_state.summary)

    st.divider()
    st.subheader("Chat with the data:")
    
    # Displays chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat Input
    if chat_q := st.chat_input("Ask a specific question about these papers..."):
        # Displays user message
        with st.chat_message("user"):
            st.markdown(chat_q)
        st.session_state.messages.append({"role": "user", "content": chat_q})

        # Generates chat response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                chat_prompt = f"""
                Answer the user question based strictly on the provided abstracts.
                If the answer is not in the text, state "Not mentioned in these papers."
                
                Abstracts:
                {st.session_state['abstracts']}
                
                Question:
                {chat_q}
                """
                chat_response = model.generate_content(chat_prompt)
                st.markdown(chat_response.text)
        
        # Appends chat messages to session state
        st.session_state.messages.append({"role": "assistant", "content": chat_response.text})
    