import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from metapub import PubMedFetcher
import datetime
import time
from pdf_generator import create_pdf
from database_manager import ResearchDB

# load environment variables from .env file
load_dotenv()

# Get key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Research DB (ChromaDB)
research_db = ResearchDB()

# Session State for memory
if 'abstracts' not in st.session_state: # Makes sure abstracts are stored when re-running
    st.session_state['abstracts'] = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "free_only" not in st.session_state:
    st.session_state.free_only = False

if "last_retrieval_mode" not in st.session_state:
    st.session_state.last_retrieval_mode = None

if "last_retrieval_detail" not in st.session_state:
    st.session_state.last_retrieval_detail = None

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
current_year = datetime.datetime.now().year
year_range = st.sidebar.slider("Search papers from year:", 1871, current_year, (1980,2020))

# Number of papers to look at
paper_count = st.sidebar.number_input("Number of papers to retrieve:", 1, 1000) 

# Free access toggle
st.session_state.free_only = st.sidebar.toggle("Include Non-Free Articles", value=False) 

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
        st.session_state['abstracts'] = ""
        st.session_state.messages = []
        st.session_state.summary = None
        st.session_state.last_retrieval_mode = None
        st.session_state.last_retrieval_detail = None

        try:
            if research_db and getattr(research_db, 'enabled', False):
                research_db.reset_collection()
        except Exception:
            pass

        with st.spinner(f"Querying PubMed for '{user_query}' ({year_range[0]}-{year_range[1]})..."):
            try:
                # Configure AI
                genai.configure(api_key=api_key)
                
                # Initialize PubMed Fetcher
                ncbi_key = os.getenv("NCBI_API_KEY")
                fetch = PubMedFetcher(cachedir="cache") # Uses local cache to speed up re-runs

                # Fetch Data
                fetch = PubMedFetcher()
                pm_query = f"{user_query} AND {year_range[0]}:{year_range[1]}[dp]" # dp for date of publication
                docs_to_add = []
                metadatas = []
                ids = []
                found_count = 0
                paper_links = []
                processed_count = 0
                search_start = 0
                batch_size = max(10, int(paper_count) * 2)
                target_count = int(paper_count)
                    
                # Keep fetching PMID batches until we have the requested number of valid papers
                # Initialize UI progress elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                while found_count < target_count:
                    pmids = fetch.pmids_for_query(pm_query, retstart=search_start, retmax=batch_size)
                    if not pmids:
                        break

                    total_pmids = len(pmids)

                    for idx, pmid in enumerate(pmids):
                        if found_count >= target_count:
                            break

                        try:
                            article = fetch.article_by_pmid(pmid)

                            processed_count += 1

                            # Update progress based on valid papers collected so far.
                            progress_pct = int((found_count / max(1, target_count)) * 100)
                            try:
                                progress_bar.progress(min(progress_pct, 99))
                            except Exception:
                                pass

                            # Short title fragment for status
                            title_fragment = getattr(article, 'title', '') or ''
                            if title_fragment and len(title_fragment) > 120:
                                title_fragment = title_fragment[:120] + "..."
                            try:
                                status_text.text(
                                    f"Processing paper {found_count + 1} of {target_count}: {title_fragment}"
                                )
                            except Exception:
                                pass

                            # Handle Free Only Toggle
                            is_free = article.pmc is not None
                            if not st.session_state.free_only and not is_free:
                                continue
                            
                            # Uses gettattr to avoid errors if 'authors' attribute is missing
                            authors_list = getattr(article, 'authors', [])
                            authors_str = ", ".join(authors_list) if authors_list else "No Authors Listed"

                            # Prepare text for session and DB
                            st.session_state['abstracts'] += f"TITLE: {article.title}\n"
                            st.session_state['abstracts'] += f"AUTHORS: {authors_str}\n"
                            abstract_text = article.abstract if article.abstract else "No abstract available"
                            st.session_state['abstracts'] += f"ABSTRACT: {abstract_text}\n"
                            st.session_state['abstracts'] += f"FREE: {is_free} \n"
                            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            st.session_state['abstracts'] += f"LINK: {link}\n"
                            st.session_state['abstracts'] += "-" * 20 + "\n"

                            # Add to lists to persist in ChromaDB
                            doc_text = f"TITLE: {article.title}\nAUTHORS: {authors_str}\nABSTRACT: {abstract_text}\nLINK: {link}\nFREE: {is_free}"
                            # Try to extract year from article attributes
                            year = getattr(article, 'year', None)
                            if not year:
                                pubdate = getattr(article, 'pubdate', '') or getattr(article, 'publication_date', '')
                                try:
                                    year = str(pubdate).split()[0][:4]
                                except Exception:
                                    year = ""

                            docs_to_add.append(doc_text)
                            metadatas.append({"year": year, "link": link})
                            ids.append(str(pmid))
                            found_count += 1

                            progress_pct = int((found_count / max(1, target_count)) * 100)
                            try:
                                progress_bar.progress(min(progress_pct, 99))
                            except Exception:
                                pass

                            if not ncbi_key:
                                time.sleep(0.35) # To avoid hitting rate limits without an API key

                        except Exception:
                            # Skip any articles that cause errors
                            if not ncbi_key:
                                time.sleep(0.35)
                            continue

                    search_start += total_pmids

                try:
                    progress_bar.progress(100 if found_count >= target_count and target_count > 0 else 0)
                except Exception:
                    pass
                
                # Finalize UI progress elements so summary area is clean
                try:
                    status_text.empty()
                except Exception:
                    pass
                try:
                    progress_bar.empty()
                except Exception:
                    pass

                if found_count > 0:
                    # Persist to ChromaDB (if available)
                    try:
                        if research_db and getattr(research_db, 'enabled', False):
                            research_db.add_abstracts(docs_to_add, metadatas, ids)
                    except Exception:
                        pass

            except Exception as e:
                st.error(f"An error occurred: {e}")

        # The retrieval spinner is now gone; only the summary phase should show its own spinner.
        if found_count > 0:
            try:
                status_text.text("PubMed retrieval complete. Generating summary with Gemini...")
            except Exception:
                pass

            with st.spinner("Generating summary with Gemini..."):
                # Generate Summary
                model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"""
                You are a Senior Biomedical Researcher. Don't fabricate information. If uncertain, state that the information is not available.
                Make sure you make headings for each task and never mention "provided abstracts" in your answer.
                Given the following abstracts from recent PubMed papers on a specific research topic, perform the following tasks:
                1. Synthesize a summary of findings based ONLY on the provided abstracts.
                2. Identify the single most relevant paper for the user to read. Start with "The most relevant paper to read is", then explain why you chose it in 1 sentence.
                3. Provide a reference section in IEEE format. Make sure there are two empty lines between each reference. If the article is free, indicate that in the reference with [Free Access].
                
                Here are the abstracts:
                {st.session_state['abstracts']}
                """
                response = model.generate_content(prompt)
                st.session_state.summary = response.text

            try:
                status_text.empty()
            except Exception:
                pass
            st.success(f"Analysis Complete. Found {found_count} relevant papers.")
        else:
            st.warning("No papers with abstracts found. Try a broader term.")

# Chat interface for follow-up questions
if st.session_state['abstracts']:
    # Displays the Summary (Always visible at the top)
    st.info(st.session_state.summary)

    if st.session_state.last_retrieval_mode:
        st.caption(f"Last chat retrieval mode: {st.session_state.last_retrieval_mode}")
        if st.session_state.last_retrieval_detail:
            st.caption(st.session_state.last_retrieval_detail)

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
            retrieval_status = st.empty()
            with st.spinner("Analyzing..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                # Use semantic search to retrieve top-k relevant context chunks
                context_text = None
                used_vector_db = False
                try:
                    if research_db and getattr(research_db, 'enabled', False):
                        used_vector_db = True
                        res = research_db.query_db(chat_q, n_results=5) # finds 5 most relevant docs
                        docs = res.get('documents', []) # list of paper abstracts
                        metas = res.get('metadatas', []) # list of dicts with metadata (e.g., year, link)
                        if docs:
                            parts = []
                            for i, d in enumerate(docs): # for each retrieved doc, add metadata if available, i is index of the doc and d is the doc text
                                meta = metas[i] if i < len(metas) else {}
                                link = meta.get('link', '')
                                year = meta.get('year', '')
                                parts.append(f"TITLE/YEAR: {year}\n{d}\nLINK: {link}\n" + "-"*20)
                            context_text = "\n".join(parts)
                except Exception:
                    context_text = None
                    used_vector_db = False

                # Fallback to full abstracts if semantic DB unavailable
                if not context_text:
                    context_text = st.session_state['abstracts']
                    used_vector_db = False

                if used_vector_db:
                    st.session_state.last_retrieval_mode = "Vector embeddings via ChromaDB"
                    st.session_state.last_retrieval_detail = "This answer used semantic retrieval from the vector store."
                    retrieval_status.success("Retrieval mode: vector embeddings via ChromaDB")
                else:
                    st.session_state.last_retrieval_mode = "Fallback to stored abstracts"
                    st.session_state.last_retrieval_detail = "This answer did not use vector retrieval; it fell back to the saved abstracts." 
                    retrieval_status.warning("Retrieval mode: fallback to stored abstracts")

                chat_prompt = f"""
                Answer the user question based strictly on the provided abstracts.
                If the answer is not in the text, state "Not mentioned in these papers."

                Abstracts:
                {context_text}

                Question:
                {chat_q}

                For context, here are the previous messages in this conversation:
                {st.session_state.messages}
                """
                chat_response = model.generate_content(chat_prompt)
                st.markdown(chat_response.text)
        
        # Appends chat messages to session state
        st.session_state.messages.append({"role": "assistant", "content": chat_response.text})

    # Export Section
    if st.session_state.summary:
        st.divider()
        st.subheader("📥 Export Your Research")

        # Combines the query, summary, and source abstracts into one string
        report_text = f"--- PUBMED LITERATURE REVIEW ---\n"
        report_text += f"DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report_text += f"TOPIC: {user_query}\n"
        report_text += f"SEARCH RANGE: {year_range[0]}-{year_range[1]}\n"
        report_text += "-" * 40 + "\n\n"
        report_text += "AI SUMMARY\n"
        report_text += st.session_state.summary + "\n\n"
        report_text += "-" * 40 + "\n"

        # Adds chat log to a separate string
        chat_log_text = report_text + "\n" + "-" * 40 + "\n"
        chat_log_text += "FOLLOW-UP CHAT LOG \n\n"
        
        for msg in st.session_state.messages:
            role = msg["role"].upper()
            content = msg["content"]
            chat_log_text += f"[{role}]: {content}\n\n"

        # Download Buttons
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            pdf_bytes_summary = create_pdf(report_text)
            st.download_button(
                label="📄 Download Summary (PDF)",
                data=pdf_bytes_summary,
                file_name=f"Research_Summary_{datetime.date.today()}.pdf",
                mime="application/pdf"
            )

        with d_col2:
            pdf_bytes_summary2 = create_pdf(chat_log_text)
            st.download_button(
                label="📄 Download Summary (PDF) With Chat History",
                data=pdf_bytes_summary2,
                file_name=f"Research_Summary_{datetime.date.today()}.pdf",
                mime="application/pdf"
            )
    