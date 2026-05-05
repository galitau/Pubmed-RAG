import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from metapub import PubMedFetcher
import datetime
import time
import json
from pdf_generator import create_pdf
from database_manager import ResearchDB
from fulltext_fetcher import (
    fetch_pmc_fulltext,
    chunk_text,
    resolve_article,
    get_pmc_id,
)

# load environment variables from .env file
load_dotenv()

# Get key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
ncbi_key = os.getenv("NCBI_API_KEY")

# Initialize Research DB (ChromaDB)
research_db = ResearchDB()

# Session State for memory
if 'abstracts' not in st.session_state:
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
year_range = st.sidebar.slider("Search papers from year:", 1871, current_year, (1980, 2020))

# Number of papers to look at
paper_count = st.sidebar.number_input("Number of papers to retrieve:", 1, 1000)

# Free access toggle
st.session_state.free_only = st.sidebar.toggle("Include Non-Free Articles", value=False)

# API Key Status indicator
if api_key:
    st.sidebar.success("API Key Loaded Securely")
else:
    st.sidebar.error("No API Key found!")


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_article_reference(user_message: str, abstracts_text: str) -> dict:
    """
    A Gemini call detects if the user is asking about
    a specific paper (e.g., "What does paper 2 say", "Tell me about the Johnson study")
    vs. a general question ("What are the key findings?").
    
    If specific article detected → fetch full-text
    If general question → use vector search on all abstracts
    
    Returns:
      { "referenced": False }
      or
      { "referenced": True, "identifier": "<what they referenced>" }
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Extract all paper titles from the stored abstracts to give Gemini context
    # This allows it to understand references like "the first study" or "paper 3"
    import re
    titles = re.findall(r"TITLE:\s*(.+)", abstracts_text)
    title_list = "\n".join(f"{i+1}. {t.strip()}" for i, t in enumerate(titles[:30]))

    prompt = f"""You are a citation detector. The user has a set of research papers numbered 1 to {len(titles)}.

Paper list:
{title_list}

User message: "{user_message}"

Does the user reference ONE specific paper from the list? This includes:
- "source 2", "reference 3", "paper 1", "#4", etc.
- Mentioning a title or partial title
- Mentioning author names (e.g. "the Johnson study")
- Mentioning a year + author combination

Respond ONLY with valid JSON, no markdown, no explanation:
If yes: {{"referenced": true, "identifier": "<the title, author, or label they used>"}}
If no:  {{"referenced": false}}"""

    try:
        # Use temperature=0 for deterministic JSON output (no randomness/creativity needed)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 80},
        )
        # Clean up response in case Gemini wrapped it in markdown code blocks
        raw = resp.text.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception:
        # On any parsing error, assume no specific article was referenced
        return {"referenced": False}


def get_fulltext_context(article: dict, user_query: str) -> tuple[str, str, str]:
    """
    This function implements a 3-stage fallback chain:
      1. Check ChromaDB cache (if already fetched this article before)
      2. Fetch from PMC (NCBI PubMed Central - open-access articles only)
      3. Fall back to abstract-only if PMC unavailable
    
    For each stage, it returns:
      - context_text: The actual content to pass to LLM (cached chunks, full-text chunks, or abstract)
      - status_level: "success", "warning", or "error" (for UI display)
      - status_message: User-friendly message explaining what was used
    
    The retrieved context is chunked (800 words, 100-word overlap) to fit in token limits
    while maintaining context continuity.
    """
    pmid = article.get("pmid", "")
    title = article.get("title", "Unknown")
    is_free = article.get("is_free", False)  # True = PMC open-access, False = paywalled

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: Check ChromaDB for cached full-text chunks
    # ═══════════════════════════════════════════════════════════════════════════
    # If we've already fetched & chunked this article, reuse it instead of re-fetching
    # Each PMID gets its own collection named "fulltext_{pmid}" in ChromaDB
    if pmid and research_db.enabled and research_db.article_collection_exists(pmid):
        # Query the article's full-text collection for semantic matches to user's question
        res = research_db.query_article_fulltext(pmid, user_query, n_results=6)
        if res["documents"]:
            context = "\n\n".join(res["documents"])
            return (
                context,
                "success",
                f"📄 **Full text loaded from cache** — *{title}*",
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2a: Check if article is open-access (is_free = True)
    # ═══════════════════════════════════════════════════════════════════════════
    # Only articles in PMC (PubMed Central) have publicly available full text
    # Paywalled articles fall back to abstract
    if not is_free:
        context = _abstract_context_for(article)
        return (
            context,
            "warning",
            f"⚠️ **Full text unavailable** (not open-access) — using abstract for *{title}*",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2b: Convert PMID to PMC ID
    # ═══════════════════════════════════════════════════════════════════════════
    # PMID and PMC ID are different identifiers; PMC ID is needed for fetching
    # This calls NCBI's ID converter API
    pmc_id = get_pmc_id(pmid, ncbi_key) if pmid else None

    if not pmc_id:
        # PMC ID not found in NCBI database (article may not be in PMC despite is_free=True)
        context = _abstract_context_for(article)
        return (
            context,
            "warning",
            f"⚠️ **PMC ID not found** — using abstract only for *{title}*",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2c: Fetch full-text XML from PMC and clean it
    # ═══════════════════════════════════════════════════════════════════════════
    # This calls NCBI's eFetch API to retrieve the article's full text
    fulltext = fetch_pmc_fulltext(pmc_id, ncbi_key)

    if not fulltext or len(fulltext) < 500:
        # Fetch failed or returned inadequate content
        context = _abstract_context_for(article)
        return (
            context,
            "warning",
            f"⚠️ **Full text fetch failed** — using abstract only for *{title}*",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 3: Split into chunks, store, and retrieve semantically
    # ═══════════════════════════════════════════════════════════════════════════
    # Break full-text into ~800-word overlapping chunks so LLM doesn't exceed token limits
    # while maintaining context continuity between chunks (100-word overlap)
    chunks = chunk_text(fulltext, chunk_size=800, overlap=100)

    if research_db.enabled and pmid:
        # Store chunks in per-article ChromaDB collection for future reuse
        # Uses upsert() so calling multiple times is safe (idempotent)
        research_db.store_article_chunks(pmid, chunks, title=title)
        
        # Query stored chunks semantically for content matching user's question
        # This scores chunks by relevance rather than just using the first N
        res = research_db.query_article_fulltext(pmid, user_query, n_results=6)
        if res["documents"]:
            # Found semantically relevant chunks in storage
            context = "\n\n".join(res["documents"])
        else:
            # Storage succeeded but query returned nothing; use first N chunks as fallback
            context = "\n\n".join(chunks[:6])
    else:
        # ChromaDB unavailable or disabled — just concatenate first N chunks
        # Skip semantic ranking; provide raw chunks
        context = "\n\n".join(chunks[:6])

    return (
        context,
        "success",
        f"✅ **Full text retrieved** ({len(chunks)} chunks) — *{title}*",
    )


def _abstract_context_for(article: dict) -> str:
    """
    Extract just one article's block from session abstracts.
    
    When full-text is unavailable (paywalled, PMC lookup failed, etc.),
    use the abstract instead. This searches through the stored abstracts
    (which are formatted as title/author/abstract blocks separated by dashes)
    to find the matching article's block.
    """
    import re
    title = article.get("title", "")
    if not title:
        # No title provided; return all abstracts as fallback
        return st.session_state["abstracts"]

    # Session abstracts are divided into blocks by a 20-dash separator (────────────────────────)
    # Each block has TITLE: ..., AUTHORS: ..., ABSTRACT: ..., etc.
    blocks = st.session_state["abstracts"].split("-" * 20)
    
    # Search for this article's block by matching title (first 40 chars for fuzziness)
    for block in blocks:
        if title.lower()[:40] in block.lower():
            return block.strip()

    # Title not found in blocks; return all abstracts as last resort
    return st.session_state["abstracts"]


# ── Main Interface ────────────────────────────────────────────────────────────

st.title("Pubmed Retrieval-Augmented Generation: Automated Literature Review")
st.markdown(
    """
    *Retrieves clinical abstracts, filters by date, and generates consensus summaries using AI.*
    """
)

# Search interface
col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_input("Enter Research Topic:", placeholder="e.g., Biocompatibility of 3D-printed bone scaffolds")

with col2:
    st.write("")
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
                genai.configure(api_key=api_key)

                fetch = PubMedFetcher(cachedir="cache")
                fetch = PubMedFetcher()
                pm_query = f"{user_query} AND {year_range[0]}:{year_range[1]}[dp]"
                docs_to_add = []
                metadatas = []
                ids = []
                found_count = 0
                processed_count = 0
                search_start = 0
                batch_size = max(10, int(paper_count) * 2)
                target_count = int(paper_count)

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

                            progress_pct = int((found_count / max(1, target_count)) * 100)
                            try:
                                progress_bar.progress(min(progress_pct, 99))
                            except Exception:
                                pass

                            title_fragment = getattr(article, 'title', '') or ''
                            if title_fragment and len(title_fragment) > 120:
                                title_fragment = title_fragment[:120] + "..."
                            try:
                                status_text.text(
                                    f"Processing paper {found_count + 1} of {target_count}: {title_fragment}"
                                )
                            except Exception:
                                pass

                            is_free = article.pmc is not None
                            if not st.session_state.free_only and not is_free:
                                continue

                            authors_list = getattr(article, 'authors', [])
                            authors_str = ", ".join(authors_list) if authors_list else "No Authors Listed"

                            st.session_state['abstracts'] += f"TITLE: {article.title}\n"
                            st.session_state['abstracts'] += f"AUTHORS: {authors_str}\n"
                            abstract_text = article.abstract if article.abstract else "No abstract available"
                            st.session_state['abstracts'] += f"ABSTRACT: {abstract_text}\n"
                            st.session_state['abstracts'] += f"FREE: {is_free} \n"
                            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            st.session_state['abstracts'] += f"LINK: {link}\n"
                            st.session_state['abstracts'] += "-" * 20 + "\n"

                            doc_text = f"TITLE: {article.title}\nAUTHORS: {authors_str}\nABSTRACT: {abstract_text}\nLINK: {link}\nFREE: {is_free}"
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
                                time.sleep(0.35)

                        except Exception:
                            if not ncbi_key:
                                time.sleep(0.35)
                            continue

                    search_start += total_pmids

                try:
                    progress_bar.progress(100 if found_count >= target_count and target_count > 0 else 0)
                except Exception:
                    pass
                try:
                    status_text.empty()
                except Exception:
                    pass
                try:
                    progress_bar.empty()
                except Exception:
                    pass

                if found_count > 0:
                    try:
                        if research_db and getattr(research_db, 'enabled', False):
                            research_db.add_abstracts(docs_to_add, metadatas, ids)
                    except Exception:
                        pass

            except Exception as e:
                st.error(f"An error occurred: {e}")

        if found_count > 0:
            with st.spinner("Generating summary with Gemini..."):
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

            st.success(f"Analysis Complete. Found {found_count} relevant papers.")
        else:
            st.warning("No papers with abstracts found. Try a broader term.")

# ── Chat Interface ────────────────────────────────────────────────────────────

if st.session_state['abstracts']:
    st.info(st.session_state.summary)

    if st.session_state.last_retrieval_mode:
        st.caption(f"Last chat retrieval mode: {st.session_state.last_retrieval_mode}")
        if st.session_state.last_retrieval_detail:
            st.caption(st.session_state.last_retrieval_detail)

    st.divider()
    st.subheader("Chat with the data:")

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Re-render any stored status banners
            if message["role"] == "assistant":
                if message.get("retrieval_status"):
                    lvl = message.get("retrieval_level", "info")
                    msg = message["retrieval_status"]
                    if lvl == "success":
                        st.success(msg)
                    elif lvl == "warning":
                        st.warning(msg)
                    elif lvl == "error":
                        st.error(msg)
                    else:
                        st.info(msg)
                if message.get("retrieval_mode_label"):
                    st.caption(message["retrieval_mode_label"])
            st.markdown(message["content"])

    # Chat input
    if chat_q := st.chat_input("Ask a specific question about these papers..."):
        with st.chat_message("user"):
            st.markdown(chat_q)
        st.session_state.messages.append({"role": "user", "content": chat_q})

        with st.chat_message("assistant"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            # ═══════════════════════════════════════════════════════════════════════════
            # HIERARCHICAL RAG CHAT FLOW
            # ═══════════════════════════════════════════════════════════════════════════
            # This implements a 5-step chain:
            #   1. Detect if user references a specific article
            #   2. If yes → resolve it to an article dict
            #   3. If resolved → fetch full-text (with fallbacks)
            #   4. If no specific article → semantic search all abstracts (vector DB)
            #   5. Generate answer with best available context
            
            # Placeholders for live status indicators (updated as we progress through steps)
            detect_status = st.empty()
            retrieval_status = st.empty()
            mode_caption = st.empty()

            # Context retrieval metadata (for tracking which method was used)
            context_text = None
            retrieval_level = "info"
            retrieval_msg = None
            mode_label = None
            used_fulltext = False
            used_vector_db = False

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 1: DETECT ARTICLE REFERENCE
            # ═══════════════════════════════════════════════════════════════════════════
            detect_status.info("🔍 Checking if a specific article is referenced...")

            # Ask Gemini: does the user mention a specific paper, or ask a general question?
            detection = detect_article_reference(chat_q, st.session_state["abstracts"])

            if detection.get("referenced"):
                identifier = detection.get("identifier", "")
                detect_status.info(f"🔍 Specific article detected: *\"{identifier}\"* — resolving...")

                # ═══════════════════════════════════════════════════════════════════════════
                # STEP 2: RESOLVE ARTICLE
                # ═══════════════════════════════════════════════════════════════════════════
                # User said something like "source 2", "Johnson study", or "the first paper"
                # Now match that to an actual article dict with PMID, title, etc.
                article = resolve_article(identifier, st.session_state["abstracts"])

                if article:
                    title = article.get("title", "Unknown")

                    # ═══════════════════════════════════════════════════════════════════════════
                    # STEP 3: FETCH FULL-TEXT FOR SPECIFIC ARTICLE
                    # ═══════════════════════════════════════════════════════════════════════════
                    # Now that we know which article the user is asking about,
                    # retrieve its full-text (if available) instead of just abstract
                    with st.spinner(f"📥 Fetching full text for: *{title[:80]}*..."):
                        context_text, retrieval_level, retrieval_msg = get_fulltext_context(
                            article, chat_q
                        )
                        used_fulltext = True

                    detect_status.empty()
                    if retrieval_level == "success":
                        retrieval_status.success(retrieval_msg)
                        mode_label = "Retrieval mode: full-text semantic search (PMC)"
                        st.session_state.last_retrieval_mode = "Full-text via PMC"
                        st.session_state.last_retrieval_detail = f"Full article text used: {title}"
                    elif retrieval_level == "warning":
                        retrieval_status.warning(retrieval_msg)
                        mode_label = "Retrieval mode: abstract only (full text unavailable)"
                        st.session_state.last_retrieval_mode = "Abstract fallback (PMC unavailable)"
                        st.session_state.last_retrieval_detail = retrieval_msg
                    else:
                        retrieval_status.error(retrieval_msg)
                        mode_label = "Retrieval mode: fallback to all abstracts"
                        st.session_state.last_retrieval_mode = "Fallback (article unresolved)"
                        st.session_state.last_retrieval_detail = retrieval_msg

                    mode_caption.caption(mode_label)

                else:
                    # Could not resolve the article from the session
                    detect_status.empty()
                    retrieval_status.warning(
                        f"⚠️ **Could not identify article** matching *\"{identifier}\"* — falling back to semantic search"
                    )
                    mode_label = "Retrieval mode: vector search (article not resolved)"
                    mode_caption.caption(mode_label)
                    st.session_state.last_retrieval_mode = "Vector search (article unresolved)"
                    st.session_state.last_retrieval_detail = None

            else:
                # User didn't reference a specific article; continue to vector search
                detect_status.empty()

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 4: VECTOR SEARCH FALLBACK (if no specific article identified)
            # ═══════════════════════════════════════════════════════════════════════════
            # If context_text is still None, use ChromaDB semantic search on all abstracts
            # This finds the most relevant papers based on query embedding similarity
            if not context_text:
                try:
                    if research_db and getattr(research_db, 'enabled', False):
                        used_vector_db = True
                        res = research_db.query_db(chat_q, n_results=5)
                        docs = res.get('documents', [])
                        metas = res.get('metadatas', [])
                        if docs:
                            parts = []
                            for i, d in enumerate(docs):
                                meta = metas[i] if i < len(metas) else {}
                                link = meta.get('link', '')
                                year = meta.get('year', '')
                                parts.append(f"TITLE/YEAR: {year}\n{d}\nLINK: {link}\n" + "-" * 20)
                            context_text = "\n".join(parts)
                except Exception:
                    context_text = None
                    used_vector_db = False

                if not context_text:
                    # Final fallback: raw session abstracts
                    context_text = st.session_state['abstracts']
                    used_vector_db = False
                    retrieval_status.warning("⚠️ **Retrieval mode: fallback to stored abstracts**")
                    mode_label = "Retrieval mode: fallback to stored abstracts"
                    mode_caption.caption(mode_label)
                    st.session_state.last_retrieval_mode = "Fallback to stored abstracts"
                    st.session_state.last_retrieval_detail = (
                        "This answer did not use vector retrieval; it fell back to the saved abstracts."
                    )
                elif used_vector_db and not used_fulltext:
                    retrieval_status.success("✅ **Retrieval mode: vector embeddings via ChromaDB**")
                    mode_label = "Retrieval mode: vector embeddings via ChromaDB"
                    mode_caption.caption(mode_label)
                    st.session_state.last_retrieval_mode = "Vector embeddings via ChromaDB"
                    st.session_state.last_retrieval_detail = (
                        "This answer used semantic retrieval from the vector store."
                    )

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 5: GENERATE ANSWER WITH BEST AVAILABLE CONTEXT
            # ═══════════════════════════════════════════════════════════════════════════
            # Now generate LLM response using the context we found (full-text, vector search, or abstract)
            with st.spinner("Generating answer..."):
                chat_prompt = f"""
Answer the user question based strictly on the provided context.
If the answer is not in the text, state "Not mentioned in these papers."

Context:
{context_text}

Question:
{chat_q}

For context, here are the previous messages in this conversation:
{st.session_state.messages}
"""
                chat_response = model.generate_content(chat_prompt)
                response_text = chat_response.text

            st.markdown(response_text)

        # Persist message with retrieval metadata for re-render
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "retrieval_status": retrieval_msg,
            "retrieval_level": retrieval_level,
            "retrieval_mode_label": mode_label,
        })

    # ── Export Section ────────────────────────────────────────────────────────
    if st.session_state.summary:
        st.divider()
        st.subheader("📥 Export Your Research")

        report_text = f"--- PUBMED LITERATURE REVIEW ---\n"
        report_text += f"DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report_text += f"TOPIC: {user_query}\n"
        report_text += f"SEARCH RANGE: {year_range[0]}-{year_range[1]}\n"
        report_text += "-" * 40 + "\n\n"
        report_text += "AI SUMMARY\n"
        report_text += st.session_state.summary + "\n\n"
        report_text += "-" * 40 + "\n"

        chat_log_text = report_text + "\n" + "-" * 40 + "\n"
        chat_log_text += "FOLLOW-UP CHAT LOG \n\n"
        for msg in st.session_state.messages:
            role = msg["role"].upper()
            content = msg["content"]
            chat_log_text += f"[{role}]: {content}\n\n"

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