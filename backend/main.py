import os
import shutil
from pathlib import Path
import pdfplumber
import logging
from typing import List, Dict
from uuid import uuid4
from dotenv import load_dotenv

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
        self.session_id = str(uuid4())  # Unique session ID for each agent instance
        self.memory: List[Dict] = []
        logger.debug(f"Initialized ComplianceAgent with doc_id: {self.doc_id}, session_id: {self.session_id}")

    def analyze(self, document_text: str, retrieved_rules: List[Document]):
        logger.debug(f"Starting analysis for doc_id: {self.doc_id}, session_id: {self.session_id}")

        # If previous memory exists, include it in prompt
        previous_analysis = self.memory[-1]["response"] if self.memory else None

        comparison_instruction = ""
        if previous_analysis:
            comparison_instruction = (
                "This is a revised version of a previously analyzed 10-K filing. \n"
                "Below is the earlier compliance analysis:\n\n"
                "--- START OF PRIOR ANALYSIS ---\n"
                f"{previous_analysis}\n"
                "--- END OF PRIOR ANALYSIS ---\n\n"
                "Please compare this new version to the prior one. "
                "Only highlight issues that are still unresolved, or note improvements.\n"
            )

        # Format rule context separately to avoid f-string expression with backslashes
        rules_text = "\n".join([doc.page_content for doc in retrieved_rules])

        # Build final prompt text
        prompt_text = (
            f"{comparison_instruction}\n"
            "Now analyze this updated 10-K document for SEC compliance.\n\n"
            "## 📄 Document:\n"
            f"{document_text}\n\n"
            "## 📜 Relevant SEC Compliance Rules:\n"
            f"{rules_text}\n\n"
            "Respond with a summary of:\n"
            "- Remaining compliance issues\n"
            "- Resolved items\n"
            "- Improvements since the prior version (if any)\n"
        )

        # Invoke Gemini LLM
        result = llm.invoke(prompt_text)
        logger.debug(f"LLM analysis completed for doc_id: {self.doc_id}, session_id: {self.session_id}")

        # Save result to memory
        self.memory.append({
            "document_snippet": document_text[:300],
            "rules_used": [doc.metadata for doc in retrieved_rules],
            "response": result,
        })
        logger.debug(
            f"Analysis result stored in memory for doc_id: {self.doc_id}, "
            f"session_id: {self.session_id}, memory length: {len(self.memory)}"
        )

        return result

    def get_memory(self):
        logger.debug(f"Retrieving memory for doc_id: {self.doc_id}, session_id: {self.session_id}")
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
llm = VertexAI(model_name="gemini-1.5-flash-002")

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

@app.post("/analyze/")
async def analyze_document(file: UploadFile = File(...)):
    """
    Endpoint to:
      1. Save the uploaded PDF
      2. Extract its text
      3. Create document chunks and build a Pinecone vector store
      4. Retrieve relevant SEC rules (via similarity search)
      5. Invoke the LLM with the prompt to perform compliance analysis
    """
    # Save the uploaded PDF
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(file_path)
        
        # Split the text into chunks
        chunks = text_splitter.split_text(pdf_text)
        documents = [Document(page_content=text) for text in chunks]
        
        # Build the Pinecone vector store and retriever
        docsearch = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
        retriever = docsearch.as_retriever()
        
        # Retrieve relevant compliance rules (using the same PDF text here for demonstration)
        retrieved_rules = retriever.get_relevant_documents(pdf_text)
        
        # Get or create agent for this document
        # TODO: Change doc_id to userId
        doc_id = file.filename
        if doc_id not in agent_memory_store:
            logger.debug(f"Creating new ComplianceAgent for document: {doc_id}")
            agent_memory_store[doc_id] = ComplianceAgent(doc_id)
        else:
            logger.debug(f"Using existing ComplianceAgent for document: {doc_id}")

        agent = agent_memory_store[doc_id]
        logger.debug(f"Starting analysis for document: {doc_id}")
        analysis = agent.analyze(pdf_text, retrieved_rules)
        logger.debug(f"Analysis completed for document: {doc_id}")

        return {
            "analysis": analysis,
            "filename": file.filename,
            "file_url": file_path,
        }
    except Exception as e:
        logger.exception("Analysis failed.")
        return {"error": str(e)}