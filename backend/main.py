import os
import shutil
from pathlib import Path
import pdfplumber
import logging
import fitz
import re
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pdfplumber import open as open_pdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate



# ================================
# FastAPI & Logging Setup
# ================================
app = FastAPI()

UPLOAD_DIR = Path("../uploads")
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
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")


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
- **Direct Quote** [One direct quote from the document that is an area of concern - wrapped in ""]
- **Potential Concern:** [Explain why this area might require attention]
- **Best Practices:** [General industry best practice for improving compliance]
- **Referenced SEC Rule:** [Cite SEC guidelines, but do not interpret them as legal advice]

**Do NOT provide legal conclusions. Instead, focus on comparison, best practices, and informational guidance.**  

**Make sure there are appropriate headers for each task.**
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
        
        # Format the prompt with the document and retrieved rules
        prompt_text = prompt.format(document=pdf_text, rules=retrieved_rules)
        
        # Invoke the LLM to get the compliance analysis
        analysis_response = llm.invoke(prompt_text)

        print(analysis_response)
        
        direct_quotes = extract_quotes(analysis_response)
        
        highlighted_path = highlight_sentences_in_pdf(file_path, direct_quotes)
        
        return {
            "analysis": analysis_response,
            "highlighted_pdf_path": highlighted_path,
            "filename": file.filename
        }
    except Exception as e:
        logger.exception("Analysis failed.")
        return {"error": str(e)}
    
def highlight_sentences_in_pdf(file_path, direct_quotes):
    # Open the PDF
    doc = fitz.open(file_path)
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Search for each quote and highlight it
        for quote in direct_quotes:

            if isinstance(quote, tuple):
                # Use the first element of the tuple
                quote_text = quote[0]
            else:
                quote_text = quote

            # Clean up quote - remove extra whitespace that might affect matching
            clean_quote = ' '.join(quote_text.split())
            instances = page.search_for(clean_quote)
            
            # Add highlight for each instance found
            for inst in instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()
    
    # Save the highlighted PDF to the uploads directory
    highlighted_filename = f"{Path(file_path).stem}_highlighted.pdf"
    highlighted_path = UPLOAD_DIR / highlighted_filename
    doc.save(str(highlighted_path))
    doc.close()
    
    return highlighted_path

# Assuming analysis_response.content contains the LLM output text
def extract_quotes(text):
    # This regex finds text between double quotes, handling escaped quotes
    pattern = r'"([^"\\]*(\\.[^"\\]*)*)"'
    return re.findall(pattern, text)