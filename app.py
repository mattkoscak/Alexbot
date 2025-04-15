import hmac
import streamlit as st
import os
import json
import time
import re
from typing import List, Dict, Optional, Any, NamedTuple
import cohere
from cohere.compass.clients.compass import CompassClient

# Define Citation class compatible with v2 API
class Citation(NamedTuple):
    text: str
    start: int
    end: int
    sources: List[Any]  # Changed from document_ids to sources in v2
    source: Optional[str] = None  # For display purposes
    date: Optional[str] = None    # For display purposes
    chunk_id: Optional[str] = None  # Added to track originating chunk

# --- Global Password Protection ---
def check_password():
    """
    Returns True if the user enters the correct global password.
    The expected password is stored in st.secrets["password"].
    """
    def password_entered():
        # Compare user-entered password with the stored secret
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            # Remove password from session state for security
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True
    
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:first-child {
        background-color: #f8fafc;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        max-width: 400px;
        margin: 100px auto;
    }
    div[data-testid="stVerticalBlock"] > div:first-child > div {
        background-color: white;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Rev Research")
    st.write("Please enter the password to access the application.")
    
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")
    return st.session_state.get("password_correct", False)

if not check_password():
    st.stop()

# -------------------- Enhanced Transcript Analyzer Class --------------------
class EnhancedTranscriptAnalyzer:
    """
    Enhanced transcript analyzer that uses query decomposition and reasoning synthesis
    to generate more comprehensive answers from transcript chunks.
    """
    def __init__(self, compass_url, compass_token, cohere_api_key, index_name="rev-test"):
        self.index_name = index_name
        self.name = "enhanced_transcript_analyzer"
        
        # Initialize Cohere client with v2 API - REDUCED TIMEOUT TO 45 SECONDS
        self.cohere_client = cohere.ClientV2(api_key=cohere_api_key, timeout=45)
        self.model_name = "command-a-03-2025"  # Default model
        
        # Use the same Compass client from the original app
        self.compass_client = CompassClient(
            index_url=compass_url,
            bearer_token=compass_token
        )
        
        # Settings
        self.per_query_limit = 5  # Reduced from 7 to 5 to minimize timeout risk
        self.temperature = 0.1
        self.max_tokens = 8000
        
        # System message for enhanced reasoning
        self.system_message = """
You are Rev Insights, an expert transcript analyst. Answer questions about business meeting transcripts directly and naturally. Today is March 23, 2025.

Focus on:
- Providing clear, direct answers based only on transcript evidence
- Including relevant people, projects, dates, and technical details when mentioned
- Using bullet points for lists when appropriate
- Organizing information chronologically when dates are available

Write in a natural, conversational style. Avoid creating arbitrary section headers or empty template sections in your answer.
"""
        
        # Reasoning template for synthesizing answers
        self.reasoning_template = """
You are an expert transcript analyst working with rev.com meeting transcripts, podcast interviews, and political speeches.

ORIGINAL QUESTION: {question}

SEARCH QUERIES USED:
{query_list}

RETRIEVED TRANSCRIPT CHUNKS:
{chunks}

ANALYSIS CONTEXT:
- Information is often scattered across multiple meetings on different dates
- Speakers may reference projects or ideas inconsistently
- The most recent transcripts generally contain the most up-to-date information
- Discussions may be cut off mid-sentence or continue in later chunks
- Project statuses and plans evolve over time

RESPONSE GUIDELINES:
- Focus ONLY on directly answering the original question
- Include ONLY information that directly addresses what was asked
- Avoid including tangential information about key people, timelines, or projects unless specifically requested
- Prioritize relevant evidence over comprehensive coverage
- Do not create sections or categories that weren't explicitly asked for
- Do NOT list sources at the end of your response

FORMATTING REQUIREMENTS:
- Use bullet points with the '•' character (not dashes or numbers) for lists
- Put each bullet point on its own line
- Include a blank line before and after each list
- Use **bold text** ONLY for section headings or when directly quoting a term from the transcript that represents a branded feature, product name, or official program
- Do NOT bold common terms, descriptive phrases, or general concepts
- Include blank lines between paragraphs
- Keep formatting simple and consistent

Using the transcript chunks above, analyze the information and synthesize a focused answer that directly addresses the original question. Keep your response tightly scoped to exactly what was asked, not what might be interesting or related.

Final Answer:
"""

    def decompose_query(self, query, status_text=None):
        """
        Decompose a complex query into multiple simpler queries for more effective searching.
        """
        # No status text updates
        
        # Create prompt for query decomposition
        decomposition_prompt = """
You are a specialized assistant that prepares queries for searching through a large corpus of transcripts.
The transcripts include business meetings from the transcription company rev.com.

Sometimes, user queries are complex and need to be broken down for better search results.
Your job is to decompose complex queries into multiple simpler queries that can be answered
independently, then combined for a complete answer.

IMPORTANT CONTEXT:
1. The search system contains transcripts from business meetings from the transcription and automatic speech recognition company, Rev.com
2. Each searchable chunk is approximately 300 words from a single speaker with the date prepended.
3. Information about a single topic is often fragmented across multiple chunks and multiple meetings.
4. People's names may appear in different forms (e.g., "Alex", "Alexander", "Alex Smith", etc.).
5. Projects and topics may be referred to by different terms or descriptions across meetings.
6. Discussions about a person's work might not explicitly mention the person in every relevant chunk.

Your job is to decompose complex queries into multiple specific queries that will effectively:
- Cast a wide enough net to capture relevant information
- Account for variations in terminology and references
- Capture content across different time periods when appropriate
- Consider both explicit and implicit references to topics and people
- You should never exceed 5 total decompose queries

DECOMPOSITION STRATEGIES:

FOR PERSON-FOCUSED QUERIES:
- Include name variations (formal/informal)
- Create separate queries for different aspects (projects, roles, opinions)
- For timeline questions, create date-specific variations
- Include queries that capture discussions where the person is mentioned indirectly

FOR PROJECT-FOCUSED QUERIES:
- Include different terms used to describe the project
- Create separate queries for project phases, challenges, and outcomes
- Include queries about people associated with the project
- Consider timeline-specific variations if appropriate

FOR TIME-BOUNDED QUERIES:
- Create month-specific variations when appropriate
- Include queries that capture references to specific time periods
- Consider seasonal references (Q1, Q2, summer launch, etc.)

EXAMPLES:

Simple Factual Query:
User: "When was the product launch meeting?"
Decomposed Queries: [
  "product launch meeting date",
  "when product launch occurred"
]

Person-Focused Query:
User: "What projects has Alex been working on for the past 6 months?"
Decomposed Queries: [
  "Alex projects last 6 months",
  "Alexander projects recent",
  "Alex working on since October",
  "Alex responsibilities recent months",
  "Alex mentioned project work"
]

Project-Focused Query:
User: "What are the main challenges with the ASR technology implementation?"
Decomposed Queries: [
  "ASR technology challenges",
  "automatic speech recognition problems",
  "ASR implementation issues",
  "speech recognition accuracy concerns",
  "ASR technical limitations"
]

IMPORTANT OUTPUT INSTRUCTIONS:
1. Your response must be ONLY a valid JSON array of strings.
2. Each string should be a decomposed search query.
3. DO NOT include any explanations or additional text - ONLY the JSON array.
4. Create an appropriate number of queries based on the complexity of the question.
5. Format: ["query 1", "query 2", "query 3", ...]

USER QUERY: {query}

DECOMPOSED QUERIES (JSON ARRAY ONLY):
"""
        
        try:
            # Call Cohere v2 API for query decomposition
            response = self.cohere_client.chat(
                model=self.model_name,
                temperature=0.1,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": decomposition_prompt.format(query=query)}
                ]
            )
            
            # Extract and parse the response (changed for v2)
            decomposition_text = response.message.content[0].text.strip()
            
            # Try to find and parse JSON in the response
            json_pattern = r'\[.*\]'
            json_match = re.search(json_pattern, decomposition_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    decomposed_queries = json.loads(json_str)
                    if isinstance(decomposed_queries, list) and decomposed_queries:
                        # Ensure we have at least a few queries for complex questions
                        if len(query.split()) > 8 and len(decomposed_queries) < 3:
                            decomposed_queries.append(query)  # Add original query as fallback
                        
                        # Limit to maximum 5 queries to reduce processing time
                        decomposed_queries = decomposed_queries[:5]
                        return decomposed_queries
                except:
                    pass
            
            # Fall back to using just the original query if parsing fails
            return [query]
            
        except Exception as e:
            print(f"Error decomposing query: {e}")
            return [query]  # Fall back to the original query

    def get_relevant_chunks(self, query, limit=5):
        """
        Retrieve transcript chunks from Compass based on a query.
        """
        try:
            search_results = self.compass_client.search_chunks(
                index_name=self.index_name,
                query=query,
                top_k=limit
            )
            docs = []
            if search_results.hits:
                for idx, hit in enumerate(search_results.hits):
                    text = hit.content.get("text", "")
                    source_filename = f"document_{idx}"
                    if hasattr(hit, 'document_id'):
                        source_filename = hit.document_id
                    
                    # Create a unique ID for citation tracking
                    chunk_id = f"chunk_{idx}_{hash(text) % 10000}"
                    
                    docs.append({
                        "title": f"doc_{idx}",
                        "source": source_filename,
                        "snippet": text,
                        "score": getattr(hit, 'score', 1.0),
                        "date": self.extract_date(text),  # Extract date from text
                        "id": chunk_id  # Add an ID for citation tracking
                    })
            # Sort by score to ensure most relevant chunks come first
            docs = sorted(docs, key=lambda x: x.get('score', 0), reverse=True)
            return docs
        except Exception as e:
            st.error(f"Error retrieving documents from Compass: {e}")
            return []
    
    def extract_date(self, text):
        """
        Extract date from transcript text if available.
        """
        date_match = re.search(r'Transcript Date:\s*(\d{4}-\d{2}-\d{2})', text)
        if date_match:
            return date_match.group(1)
        return "Unknown Date"

    def synthesize_answer(self, query, decomposed_queries, all_chunks, status_text=None):
        """
        Synthesize a comprehensive answer using the reasoning template and system message.
        Updated for Cohere API v2 with citation support and improved chunk tracking.
        """
        # No status text updates
        
        # Check if we have any relevant chunks first
        if not all_chunks:
            no_info_message = f"I couldn't find any information about '{query}' in the available transcripts."
            return no_info_message, []
        
        # Format query list for the reasoning template
        query_list = "\n".join([f"- {q}" for q in decomposed_queries])
        
        # Create a mapping of chunk IDs to their full information for easier lookup
        chunk_map = {chunk.get('id', f'chunk_{i}'): chunk for i, chunk in enumerate(all_chunks)}
        
        # Format chunks for the reasoning template
        chunks_text = ""
        # Prepare documents for RAG with citations
        documents = []
        
        for i, chunk in enumerate(all_chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            
            # Format for prompt
            chunks_text += f"[CHUNK {i+1}]\n"
            chunks_text += f"ID: {chunk_id}\n"
            chunks_text += f"Source: {chunk.get('source', 'Unknown')}\n"
            chunks_text += f"Transcript Date: {chunk.get('date', 'Unknown Date')}\n"
            chunks_text += f"{chunk.get('snippet', '')}\n\n"
            
            # Prepare document for RAG
            documents.append({
                "id": chunk_id,
                "data": {
                    "text": chunk.get('snippet', ''),
                    "source": chunk.get('source', 'Unknown'),
                    "date": chunk.get('date', 'Unknown Date')
                }
            })
        
        # Create the reasoning prompt
        reasoning_prompt = self.reasoning_template.format(
            question=query,
            query_list=query_list,
            chunks=chunks_text
        )
        
        # Determine if this is a complex query
        is_complex = len(query.split()) > 8 or "what is" in query.lower() or "tell me about" in query.lower()
        max_output_tokens = 2500 if is_complex else self.max_tokens
        
        try:
            # Call Cohere for synthesis with v2 API and citation support
            response = self.cohere_client.chat(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=max_output_tokens,
                citation_options={"mode": "FAST"},  # Use FAST mode for citations
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": reasoning_prompt}
                ],
                documents=documents  # Pass documents for RAG with citations
            )
            
            # Extract the synthesized answer with safety checks (v2 format)
            if (hasattr(response, 'message') and 
                hasattr(response.message, 'content') and 
                response.message.content and 
                len(response.message.content) > 0 and
                hasattr(response.message.content[0], 'text')):
                answer = response.message.content[0].text.strip()
            else:
                # Handle empty response
                answer = f"I couldn't find specific information about '{query}' in the available transcripts, although I did search some potentially relevant sections."
                return answer, []
            
            # Get citations from the response with safety checks (v2 format)
            citations = []
            if hasattr(response.message, 'citations') and response.message.citations is not None:
                citations = response.message.citations
            
            # Add source info to citations with improved chunk tracking
            enhanced_citations = []
            for i, citation in enumerate(citations):
                # Extract source information
                sources = []
                if hasattr(citation, 'sources') and citation.sources is not None:
                    sources = citation.sources
                
                # Extract source IDs from the sources
                source_ids = []
                source_text = ""
                
                for source in sources:
                    # Try to extract ID in different formats
                    if hasattr(source, 'id'):
                        source_ids.append(source.id)
                    elif hasattr(source, 'document') and hasattr(source.document, 'id'):
                        source_ids.append(source.document.id)
                    
                    # Try to extract source text
                    if hasattr(source, 'document') and hasattr(source.document, 'data') and hasattr(source.document.data, 'text'):
                        source_text = source.document.data.text
                
                # Find the corresponding chunks
                cited_chunks = []
                for source_id in source_ids:
                    if source_id in chunk_map:
                        cited_chunks.append(chunk_map[source_id])
                
                # If no exact matches, try content-based matching
                if not cited_chunks and citation.text:
                    for chunk in all_chunks:
                        if citation.text.lower() in chunk.get('snippet', '').lower():
                            cited_chunks.append(chunk)
                        # If we still can't match, try the other way around
                        elif chunk.get('snippet', '').lower() in citation.text.lower():
                            cited_chunks.append(chunk)
                
                # Use the first matched chunk for source/date info
                source_name = "Unknown"
                source_date = "Unknown Date" 
                chunk_id = None
                
                if cited_chunks:
                    source_name = cited_chunks[0].get('source', 'Unknown')
                    source_date = cited_chunks[0].get('date', 'Unknown Date')
                    chunk_id = cited_chunks[0].get('id')
                
                # Create enhanced citation with source info
                enhanced_citation = Citation(
                    text=citation.text,
                    start=citation.start,
                    end=citation.end,
                    sources=sources,
                    source=source_name,
                    date=source_date,
                    chunk_id=chunk_id  # Track the originating chunk
                )
                enhanced_citations.append(enhanced_citation)
            
            # Remove any "Sources:" section from the answer
            answer = re.sub(r'\n+---\n+\*\*Sources:\*\*\n(•.+\n)+$', '', answer)
            
            return answer, enhanced_citations
            
        except Exception as e:
            # Improved error handling with user-friendly messages
            if "NoneType" in str(e) or "iterable" in str(e):
                error_message = f"I couldn't find specific information about '{query}' in the available transcripts."
            elif "timeout" in str(e).lower():
                error_message = "The operation timed out. This complex question requires too much processing time. Try asking a more specific or shorter question."
            else:
                error_message = f"I encountered an issue while analyzing the transcripts: {str(e)}. Please try a different question."
                
            return error_message, []

    def generate_response_with_citations(self, answer_text, citations, text_color="#5464f7", font_family="'Inter', sans-serif", line_height="1.7"):
        """
        Format response with inline citations.
        """
        if not citations:
            return answer_text, f'<div style="font-family: {font_family}; line-height: {line_height};">{answer_text}</div>'
        
        # Start with the original answer
        markdown_response = answer_text
        offset = 0
        citation_counter = 0
        
        # Sort citations by position to process from start to end
        sorted_citations = sorted(citations, key=lambda x: x.start)
        
        for citation in sorted_citations:
            # Adjust positions based on previously added HTML
            start = citation.start + offset
            end = citation.end + offset
            text = citation.text
            
            # Safety check for citation positions
            if start >= len(markdown_response) or end > len(markdown_response) or start < 0 or end < 0 or start >= end:
                continue
            
            # Create citation marker (starting from 1 instead of 0)
            doc_refs = f'<sup>[{citation_counter + 1}]</sup>'
            
            # Create the HTML replacement with colored text and citation marker
            replacement = f'<span style="color: {text_color};">{text}</span>{doc_refs}'
            
            # Insert the HTML into the response
            markdown_response = markdown_response[:start] + replacement + markdown_response[end:]
            
            # Update offset for next citation
            offset += len(replacement) - len(text)
            citation_counter += 1
        
        # Create final HTML response
        html_response = f"""
        <div style="font-family: {font_family}; line-height: {line_height};">
        {markdown_response}
        </div>
        """
        
        return markdown_response, html_response

    def format_response(self, raw_answer, citations=None):
        """
        Convert the AI response to properly structured HTML with citations
        """
        if not raw_answer or len(raw_answer) < 10:
            return "<div class='answer-section'>Unable to generate a complete answer from the evidence.</div>"
            
        # Remove common prefixes that mention transcripts
        raw_answer = re.sub(r'^(Based on the (transcripts|evidence|information))[,:]?\s*', '', raw_answer)
        
        # First apply citation formatting if available
        if citations:
            _, html_with_citations = self.generate_response_with_citations(raw_answer, citations)
            raw_answer = html_with_citations
            
            # If we have citations, return directly as HTML
            return f'<div class="answer-section">{raw_answer}</div>'
        
        # For responses without citations, process the raw text line by line
        html_lines = []
        in_list = False
        
        for line in raw_answer.split('\n'):
            line = line.strip()
            
            # Empty line (preserve paragraph breaks)
            if not line:
                if not in_list:  # Don't add empty lines inside lists
                    html_lines.append("")
                continue
            
            # Skip "Sources:" section entirely
            if line.startswith('**Sources:**') or line.startswith('---'):
                break
                
            # Bullet points - standardize first
            bullet_match = re.match(r'^\s*[•\-\*]\s+(.*)', line)
            if bullet_match:
                content = bullet_match.group(1)
                # Handle bold formatting
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
                
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                
                html_lines.append(f"<li>{content}</li>")
            # Regular paragraph text
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                
                # Handle bold formatting
                line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                html_lines.append(f"<p>{line}</p>")
        
        # Close any open list
        if in_list:
            html_lines.append("</ul>")
        
        # Clean up consecutive empty lines
        cleaned_lines = []
        for i, line in enumerate(html_lines):
            if line == "" and i > 0 and html_lines[i-1] == "":
                continue
            cleaned_lines.append(line)
        
        # Join and wrap in styled div
        html_content = "\n".join([line for line in cleaned_lines if line is not None])
        
        return f'<div class="answer-section">{html_content}</div>'

    def answer_question(self, query, status_placeholder, answer_placeholder):
        """
        End-to-end method to answer a question.
        """
        start_time = time.time()
        
        status_text = status_placeholder
        # Clear any status text immediately
        status_text.empty()
        
        # Step 1: Decompose the query
        decomposed_queries = self.decompose_query(query)
        
        # Track research steps for UI
        research_steps = []
        for idx, sq in enumerate(decomposed_queries):
            research_steps.append({
                "step": idx + 1,
                "action": "search",
                "query": sq
            })
        
        # Step 2: Retrieve chunks for each decomposed query
        all_chunks = []
        unique_snippets = set()
        
        for idx, sq in enumerate(decomposed_queries):
            # Retrieve chunks
            chunks = self.get_relevant_chunks(sq, limit=self.per_query_limit)
            
            # Only add unique chunks to avoid duplication
            for chunk in chunks:
                if chunk["snippet"] not in unique_snippets:
                    unique_snippets.add(chunk["snippet"])
                    all_chunks.append(chunk)
        
        # Special handling for no results case - create a "no results" response
        if not all_chunks:
            no_results_response = f"I couldn't find any information about '{query}' in the available transcripts."
            
            # Format as HTML
            html_answer = f'<div class="answer-section"><p>{no_results_response}</p></div>'
            
            # Calculate processing time
            total_time = time.time() - start_time
            
            # Prepare result for UI
            result = {
                "query": query,
                "steps": research_steps,
                "evidence": [],
                "answer": no_results_response,
                "html_answer": html_answer,
                "citations": [],
                "metrics": {
                    "total_time": total_time,
                    "evidence_count": 0,
                    "decomposed_queries": decomposed_queries
                }
            }
            
            return result
        
        # Step 3: Synthesize the answer
        synthesized_answer, citations = self.synthesize_answer(
            query, decomposed_queries, all_chunks
        )
        
        # Step 4: Format the answer as HTML with citations
        html_formatted_answer = self.format_response(synthesized_answer, citations)
        
        total_time = time.time() - start_time
        
        # Prepare result in the format expected by the UI
        result = {
            "query": query,
            "steps": research_steps,
            "evidence": all_chunks,
            "answer": synthesized_answer,
            "html_answer": html_formatted_answer,
            "citations": citations,
            "metrics": {
                "total_time": total_time,
                "evidence_count": len(all_chunks),
                "decomposed_queries": decomposed_queries
            }
        }
        
        return result

# -------------------- Streamlit UI Code --------------------
st.set_page_config(
    page_title="Rev Research",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for a more professional UI
st.markdown("""
<style>
/* Base styles and resets */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
}

/* Main container styles */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Header styling */
h1, h2, h3 {
    font-weight: 600 !important;
    color: #1E293B;
}

.app-header {
    background: linear-gradient(90deg, #5464f7 0%, #7B61FF 100%);
    padding: 1rem 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(84, 100, 247, 0.15);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.app-header h1 {
    color: white !important;
    font-size: 2rem !important;
    margin: 0;
    padding: 0;
}

.app-subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1rem;
    margin-top: 0.5rem;
}

/* Sidebar customization */
section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.sidebar-header {
    margin-bottom: 2rem;
    text-align: center;
}

.logo-container {
    background-color: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}

.sidebar-divider {
    height: 1px;
    background-color: #e2e8f0;
    margin: 2rem 0;
    border: none;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    margin-bottom: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    height: 3rem;
    border-radius: 8px;
    padding: 0 1.5rem;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background-color: #EEF2FF !important;
    color: #5464f7 !important;
    border-bottom: none !important;
}

/* Query input area */
.query-container {
    background-color: white;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    margin-bottom: 1rem;
}

textarea[data-testid="stTextArea"] {
    border-radius: 8px;
    border-color: #e2e8f0;
    padding: 0.75rem;
    min-height: 80px;
    transition: all 0.2s ease;
}

textarea[data-testid="stTextArea"]:focus {
    border-color: #5464f7;
    box-shadow: 0 0 0 2px rgba(84, 100, 247, 0.2);
}

/* Button styling */
button[data-testid="baseButton-secondary"] {
    background-color: #5464f7 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
    border: none !important;
    height: 2.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(84, 100, 247, 0.25) !important;
}

button[data-testid="baseButton-secondary"]:hover {
    background-color: #4553DD !important;
    box-shadow: 0 4px 8px rgba(84, 100, 247, 0.35) !important;
}

/* Example question buttons */
.example-question {
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    height: auto !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.4 !important;
}

/* Answer section styling */
.answer-container {
    background-color: white;
    border-radius: 12px;
    padding: 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    margin-bottom: 1rem;
    overflow: hidden;
}

.answer-header {
    background-color: #f8fafc;
    padding: 0.75rem 1.5rem;
    border-bottom: 1px solid #e2e8f0;
}

.answer-header h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 500;
    color: #1E293B;
}

.answer-section {
    font-family: 'Inter', sans-serif !important;
    padding: 1.5rem 2rem !important;
    margin: 0 !important;
    border-radius: 0 !important;
    background-color: white !important;
    box-shadow: none !important;
    line-height: 1.7 !important;
    color: #334155 !important;
}

.answer-section p {
    margin-bottom: 1.25rem !important;
    line-height: 1.7 !important;
}

.answer-section ul {
    margin-left: 1.5rem !important;
    margin-top: 1rem !important;
    margin-bottom: 1.5rem !important;
}

.answer-section li {
    margin-bottom: 0.8rem !important;
    line-height: 1.6 !important;
}

.answer-section strong, .answer-section b {
    font-weight: 600 !important;
    color: #1E293B !important;
}

/* Citation styling */
.answer-section span {
    border-radius: 2px;
    padding: 0 1px;
}

.answer-section sup {
    color: #5464f7 !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    background-color: #EEF2FF;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    margin: 0 0.2rem;
    cursor: pointer;
    vertical-align: baseline !important;
    position: relative;
    top: -0.4em;
}

/* Process metrics styling */
.metrics-container {
    display: flex !important;
    gap: 1.5rem !important;
    margin-bottom: 1rem !important;
}

.metric-card {
    background-color: white !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    flex: 1 !important;
    text-align: center !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    border: 1px solid #e2e8f0 !important;
}

.metric-value {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #5464f7 !important;
    margin-bottom: 0.25rem !important;
}

.metric-label {
    font-size: 0.9rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
}

/* Citation details styling */
.stExpander {
    border: none !important;
    box-shadow: none !important;
    margin-bottom: 1.5rem !important;
}

[data-testid="stExpander"] > div:first-child {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    background-color: white !important;
}

.citation-details {
    background-color: #f8fafc !important;
    border-left: 4px solid #5464f7 !important;
    padding: 1rem 1.5rem !important;
    margin: 1rem 0 !important;
    border-radius: 0 8px 8px 0 !important;
}

.citation-text {
    font-style: italic !important;
    color: #334155 !important;
    background-color: white !important;
    padding: 1rem !important;
    border-radius: 6px !important;
    border: 1px solid #e2e8f0 !important;
    margin: 0.5rem 0 !important;
}

.citation-meta {
    font-size: 0.9rem !important;
    color: #64748b !important;
    margin-top: 0.75rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}

.citation-preview {
    background-color: white !important;
    padding: 1rem !important;
    border-radius: 6px !important;
    margin-top: 1rem !important;
    font-size: 0.9rem !important;
    color: #334155 !important;
    max-height: 350px !important;
    overflow-y: auto !important;
    border: 1px solid #e2e8f0 !important;
}

.footer {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
}

/* Initial message styling */
.stInfo {
    background-color: #EEF2FF !important;
    color: #334155 !important;
    border: none !important;
    padding: 1.25rem !important;
    border-radius: 10px !important;
}

.stInfo > div:first-child {
    background-color: #5464f7 !important;
}

/* Loading spinner customization */
.stSpinner > div {
    border-top-color: #5464f7 !important;
}

/* Helper for hiding Streamlit elements */
.hide-streamlit-elements div[data-testid="stVerticalBlock"] > div:nth-child(1) {
    display: none;
}

/* Helper class to add to divs when you want to center content */
.center-content {
    display: flex;
    justify-content: center;
    align-items: center;
}

/* Reset button styling */
.reset-button {
    background-color: #f1f5f9 !important;
    color: #334155 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 500 !important;
    border: 1px solid #e2e8f0 !important;
    transition: all 0.2s ease !important;
}

.reset-button:hover {
    background-color: #e2e8f0 !important;
    border-color: #cbd5e1 !important;
}

/* For custom expander button styles */
.custom-expander button p {
    font-weight: 500 !important;
    color: #5464f7 !important;
}

/* Fix for the tab content to take less vertical space */
.stTabs > div:nth-child(2) {
    padding-top: 0 !important;
}

/* Make the input textarea take less space */
[data-testid="stText"] {
    margin-bottom: 0.5rem !important;
}

/* Streamlit spacing adjustments */
div.stButton > button {
    margin-top: 0 !important;
}

@media (max-width: 768px) {
    .app-header {
        padding: 0.75rem;
    }
    
    .app-header h1 {
        font-size: 1.5rem !important;
    }
    
    .metrics-container {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

if 'agent' not in st.session_state:
    required_vars = ["COHERE_API_KEY", "COMPASS_TOKEN", "COMPASS_URL"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.stop()
    # Use the environment variable for index_name if available
    index_name = os.environ.get("COMPASS_INDEX_NAME", "rev-test")
    st.session_state.agent = EnhancedTranscriptAnalyzer(
        compass_url=os.environ["COMPASS_URL"],
        compass_token=os.environ["COMPASS_TOKEN"],
        cohere_api_key=os.environ["COHERE_API_KEY"],
        index_name=index_name
    )

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("revlogo.jpeg", width=100, caption="", use_container_width=True)
    with col2:
        st.image("coherelogo.png", width=100, caption="", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-bottom: 20px; color: #64748b; font-weight: 500;'>Powered by Rev and Cohere</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    st.subheader("Actions")
    if st.button("Reset Agent", key="reset_agent", use_container_width=True):
        index_name = os.environ.get("COMPASS_INDEX_NAME", "rev-test")
        st.session_state.agent = EnhancedTranscriptAnalyzer(
            compass_url=os.environ["COMPASS_URL"],
            compass_token=os.environ["COMPASS_TOKEN"],
            cohere_api_key=os.environ["COHERE_API_KEY"],
            index_name=index_name
        )
        st.success("Agent reset successfully!")

# --- MAIN CONTENT ---
# Header section with gradient background
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.markdown('<h1>Rev Research</h1>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Search meeting transcripts with AI-powered insights</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if 'results' not in st.session_state:
    st.session_state.results = []

example_questions = [
    "Who secured the Udemy renewal?",
    "What is Ryan Schumacher's favorite color?",
    "What is the subscription management platform used by Rev?",
    "How does Rev's customer-centric approach influence its product roadmap?",
    "How does Rev gather and utilize customer feedback to shape its products and services?",
    "What are the benefits of Rev's autonomous pod structure for team dynamics and innovation?",
    "What security measures has Rev implemented to address the concerns of law firms, and who is leading these efforts?"
]

tab1, tab2 = st.tabs(["Ask a Question", "Example Questions"])

with tab1:
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    if 'rerun_query' in st.session_state:
        query = st.session_state.rerun_query
        del st.session_state.rerun_query
    else:
        query = st.text_area("Enter your question about the transcripts:", height=80)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit", use_container_width=True)
    with col2:
        pass
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.write("Click on an example question to use it:")
    
    for i, example in enumerate(example_questions):
        if st.button(example, key=f"example_{i}", use_container_width=True, 
                    help=f"Click to ask: {example}", 
                    type="secondary",
                    on_click=None):
            query = example
            submit_button = True
    st.markdown('</div>', unsafe_allow_html=True)

# Immediately render the answer container to minimize scrolling
answer_container = st.container()
answer_container.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

if submit_button and query.strip():
    # Create placeholder elements
    status_placeholder = st.empty()
    
    # Create a styled answer container
    answer_container.markdown('<div class="answer-container">', unsafe_allow_html=True)
    answer_container.markdown('<div class="answer-header"><h2>Answer</h2></div>', unsafe_allow_html=True)
    answer_placeholder = answer_container.empty()
    answer_container.markdown('</div>', unsafe_allow_html=True)
    
    # Process the question
    with st.spinner("Analyzing transcripts..."):
        result = st.session_state.agent.answer_question(
            query, 
            status_placeholder, 
            answer_placeholder
        )
    
    # Store the result
    st.session_state.results.append(result)
    
    # Display the final answer
    answer_placeholder.markdown(result["html_answer"], unsafe_allow_html=True)
    
    # Display metrics with improved styling
    answer_container.markdown(f"""
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-value">{result['metrics']['total_time']:.2f}s</div>
            <div class="metric-label">Processing Time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{result['metrics']['evidence_count']}</div>
            <div class="metric-label">Evidence Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(result.get('citations', []))}</div>
            <div class="metric-label">Citations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show citation details if available with improved styling
    if result.get("citations"):
        with answer_container.expander("Citation Details", expanded=False):
            st.markdown("<div style='background-color: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
            st.subheader("Citations")
            for idx, citation in enumerate(result["citations"]):
                st.markdown(f'<div class="citation-details">', unsafe_allow_html=True)
                st.markdown(f"**Citation [{idx + 1}]**")
                st.markdown(f'<div class="citation-text">"{citation.text}"</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="citation-meta">Source: {citation.source} | Date: {citation.date}</div>', unsafe_allow_html=True)
                
                # Find the matching chunk for this citation
                matching_chunks = []
                if citation.chunk_id:
                    # First try exact ID match
                    for chunk in result["evidence"]:
                        if chunk.get('id') == citation.chunk_id:
                            matching_chunks.append(chunk)
                
                # If no direct match, try text matching
                if not matching_chunks:
                    for chunk in result["evidence"]:
                        if citation.text.lower() in chunk.get('snippet', '').lower():
                            matching_chunks.append(chunk)
                
                # Display the entire transcript chunk
                if matching_chunks:
                    chunk = matching_chunks[0]  # Use the first match
                    full_text = chunk.get('snippet', '')
                    
                    # Simply show the full text with basic formatting
                    st.markdown(f'<div class="citation-preview">Transcript Date: {chunk.get("date", "Unknown Date")}<br><br>{full_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("*Source text not found*")
                
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
elif not st.session_state.results:
    # Add subtle info message
    st.info("👆 Enter a question about the meeting transcripts to get started!")

# Footer section
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2025 - Powered by Rev and Cohere", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)