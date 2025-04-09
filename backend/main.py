import os
import shutil
from pathlib import Path
import pdfplumber
import logging
from typing import List, Dict
from uuid import uuid4
from dotenv import load_dotenv
import json
import re


from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate

# ================================
# Agent Class
# ================================

# In-memory store to track agent instances
agent_memory_store = {}  # key = doc_id (filename), value = ComplianceAgent instance
class ComplianceAgent:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.session_id = str(uuid4())
        self.memory: List[Dict] = []
        logger.debug(f"Initialized ComplianceAgent with doc_id: {self.doc_id}, session_id: {self.session_id}")

    def build_markdown_prompt(self, document_text: str, retrieved_rules: List[Document]) -> str:
        previous = self.memory[-1]["markdown"] if self.memory else None
        comparison_block = (
            f"This is a revised version of a previously analyzed 10-K.\n\n"
            f"---\nPrior Analysis:\n{previous}\n---\n\n"
            "Compare the new version to the prior one. Identify:\n"
            "- Which items improved or were added\n"
            "- Which still remain deficient\n"
            "- Summarize new compliance risks or fixes\n\n"
            if previous else ""
        )

        rules_text = "\n".join([doc.page_content for doc in retrieved_rules])
        return (
            f"{comparison_block}\n"
            "## 📄 Updated 10-K Document:\n"
            f"{document_text}\n\n"
            "## 📜 Relevant SEC Compliance Rules:\n"
            f"{rules_text}\n\n"
            "Return a markdown-formatted compliance analysis with sections for:\n"
            "1. Section Summary\n2. Potential Concern\n3. Best Practices\n4. Referenced SEC Rule\n"
        )

    def analyze(self, document_text: str, retrieved_rules: List[Document], markdown_result: str, structured_result: List[Dict]):
        self.memory.append({
            "document_snippet": document_text[:300],
            "rules_used": [doc.metadata for doc in retrieved_rules],
            "markdown": markdown_result,
            "structured": structured_result,
        })
        logger.debug(f"Analysis saved to memory for doc_id: {self.doc_id}")

    def diff_latest_vs_previous(self) -> List[Dict]:
        if len(self.memory) < 2:
            return [
                {
                    "section": item["section"],
                    "previous_status": "N/A",
                    "current_status": item["status"],
                    "change_summary": f"🆕 New: {item['status']} — {item['summary']}"
                }
                for item in self.memory[-1]["structured"]
            ]

        prev_structured = self.memory[-2]["structured"]
        curr_structured = self.memory[-1]["structured"]

        prev_map = {item["section"]: item for item in prev_structured}
        curr_map = {item["section"]: item for item in curr_structured}

        timeline_rows = []

        for section, curr_item in curr_map.items():
            prev_item = prev_map.get(section)

            if prev_item:
                if curr_item["status"] != prev_item["status"]:
                    change = f"{prev_item['status']} → {curr_item['status']}: {curr_item['summary']}"
                else:
                    change = "No change"
                timeline_rows.append({
                    "section": section,
                    "previous_status": prev_item["status"],
                    "current_status": curr_item["status"],
                    "change_summary": change,
                })
            else:
                timeline_rows.append({
                    "section": section,
                    "previous_status": "N/A",
                    "current_status": curr_item["status"],
                    "change_summary": f"🆕 New: {curr_item['status']} — {curr_item['summary']}"
                })

        return timeline_rows

    def get_memory(self):
        return self.memory






# ================================
# FastAPI & Logging Setup
# ================================
app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
langsmiith_tracing = os.getenv("LANGSMITH_TRACING")
langsmiith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmith_project = os.getenv("LANGSMITH_PROJECT")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("INDEX_NAME")
google_api_key = os.environ["GOOGLE_API_KEY"] 
google_cloud_project = os.environ["GOOGLE_CLOUD_PROJECT"]


# ================================
# PDF Extraction
# ================================
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF via pdfplumber.
    Returns the extracted text as a string.
    """
    logger.debug(f"Extracting text from PDF: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    logger.debug(f"Extracted text length: {len(text)}")
    return text

# ================================
# LangChain Components Setup
# ================================
# Initialize text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
vertexai.init(project='gen-lang-client-0348162842', location='us-central1')
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    project_id="gen-lang-client-0348162842",
    location="us-central1"
)


# Define the prompt template
prompt_template = """
You are a highly skilled financial compliance expert with deep knowledge of SEC regulations for Form 10-K filings. Your task is to review the provided financial document and compare it to relevant SEC compliance rules. 

## 📄 Financial Document Excerpt:
{document}

## 📜 Relevant SEC Compliance Rules:
{rules}

### 🔍 **Analysis Tasks:**
1. **Identify Potential Compliance Concerns**  
   - Highlight sections that may require further review.  
   - Explain why these areas could be relevant to SEC guidelines.  
   - Provide references to similar SEC rules for context.

2. **Provide Best Practices and Industry Standards**  
   - Share examples of how similar filings structure these sections.  
   - Recommend general best practices for SEC-compliant disclosures.  

3. **Risk Indicators (Not a Legal Assessment)**  
   - Indicate whether any sections might attract regulatory scrutiny.  
   - Provide insights on improving clarity, transparency, or disclosure quality.  

### ⚠️ **Response Format:**
For each identified area of concern, respond with:
- **Section Summary:** [Summarized key point from the document]
- **Potential Concern:** [Explain why this area might require attention]
- **Best Practices:** [General industry best practice for improving compliance]
- **Referenced SEC Rule:** [Cite SEC guidelines, but do not interpret them as legal advice]

**Do NOT provide legal conclusions. Instead, focus on comparison, best practices, and informational guidance.**  
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
llm = VertexAI(model_name="gemini-1.5-flash-002", temperature=0.0)

# ================================
# FastAPI Routes
# ================================
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Saves the PDF to disk.
    """
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "message": "File uploaded successfully"}

# Helper to clean up markdown-wrapped JSON
def strip_markdown_json_fencing(text: str) -> str:
    return re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

def strip_markdown_json_fencing(text: str) -> str:
    return re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

@app.post("/analyze/")
async def analyze_document(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pdf_text = extract_text_from_pdf(file_path)
        chunks = text_splitter.split_text(pdf_text)
        documents = [Document(page_content=text) for text in chunks]
        docsearch = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
        retriever = docsearch.as_retriever()
        retrieved_rules = retriever.get_relevant_documents(pdf_text)

        doc_id = file.filename
        if doc_id not in agent_memory_store:
            logger.debug(f"Creating new ComplianceAgent for document: {doc_id}")
            agent_memory_store[doc_id] = ComplianceAgent(doc_id)
        else:
            logger.debug(f"Using existing ComplianceAgent for document: {doc_id}")

        agent = agent_memory_store[doc_id]

        # === 1. Markdown-style analysis ===
        markdown_prompt_text = agent.build_markdown_prompt(pdf_text, retrieved_rules)

        markdown_result = llm.invoke(markdown_prompt_text)

        # === 2. Structured JSON analysis from markdown ===
        structured_prompt = (
            "You previously wrote the following compliance analysis of a 10-K document:\n\n"
            "---\n"
            f"{markdown_result}\n"
            "---\n\n"
            "Based on the report above, output a structured JSON list of compliance sections in the following format:\n"
            "[\n"
            "  {\n"
            "    \"section\": \"Item 1A - Risk Factors\",\n"
            "    \"status\": \"partial\",  // 'pass', 'fail', or 'partial'\n"
            "    \"summary\": \"Risks are now listed but still vague.\",\n"
            "    \"rule\": \"Regulation S-K Item 105\"\n"
            "  },\n"
            "  ...\n"
            "]\n"
            "Respond ONLY with valid JSON. Do not include markdown, code blocks, or any text outside the list."
        )

        structured_result_raw = llm.invoke(structured_prompt)
        logger.debug(f"Structured analysis result: {structured_result_raw}")
        cleaned_result = strip_markdown_json_fencing(structured_result_raw)

        try:
            structured_result = json.loads(cleaned_result)
        except Exception as e:
            logger.warning(f"Failed to parse structured analysis JSON: {e}")
            structured_result = []

        # Store in memory
        agent.analyze(pdf_text, retrieved_rules, markdown_result, structured_result)
        timeline = agent.diff_latest_vs_previous()

        return {
            "filename": file.filename,
            "file_url": file_path,
            "analysis": markdown_result,
            "structured_analysis": structured_result,
            "timeline": timeline,
        }

    except Exception as e:
        logger.exception("Analysis failed.")
        return {"error": str(e)}
