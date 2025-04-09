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

from fastapi import Request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from supabase import create_client, Client



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
google_api_key = os.environ["GOOGLE_API_KEY"] 
google_cloud_project = os.environ["GOOGLE_CLOUD_PROJECT"]
supabase_url = os.getenv("REACT_APP_SUPABASE_URL")
supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY") 
supabase: Client = create_client(supabase_url, supabase_service_key)



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
async def analyze_document(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to:
      1. Save the uploaded PDF
      2. Extract its text
      3. Create document chunks and build a Pinecone vector store
      4. Retrieve relevant SEC rules (via similarity search)
      5. Invoke the LLM with the prompt to perform compliance analysis
    """
    # get user_id from request
    user_id = request.headers.get("x-user-id")
    if not user_id:
        return {"error": "User ID missing in headers"}
    storage_path = f"{user_id}/{file.filename}"
    try:
        file_bytes = await file.read()
        upload_response = supabase.storage.from_("legal-docs").upload(
            storage_path,
            file_bytes,
            {"content-type": file.content_type,
             "x-upsert" : "true"}
        )
        if hasattr(upload_response, "error") and upload_response.error:
            return {"error": f"Supabase upload failed: {upload_response.error['message']}"}
        signed_url_response = supabase.storage.from_("legal-docs").create_signed_url(
            storage_path,
            60 * 60  # 1 hour
        )
        if signed_url_response.get("error"):
            return {"error": "Failed to create signed URL"}
        signed_url = signed_url_response["signedURL"]
        with open(f"{UPLOAD_DIR}/{file.filename}", "wb") as temp_file:
            temp_file.write(file_bytes)

        pdf_text = extract_text_from_pdf(Path(f"{UPLOAD_DIR}/{file.filename}"))
        chunks = text_splitter.split_text(pdf_text)
        documents = [Document(page_content=text) for text in chunks]

        docsearch = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
        retriever = docsearch.as_retriever()
        retrieved_rules = retriever.get_relevant_documents(pdf_text)

        # 8. Format prompt and run analysis
        prompt_text = prompt.format(document=pdf_text, rules=retrieved_rules)
        analysis_response = llm.invoke(prompt_text)

        print(analysis_response)

        direct_quotes = extract_quotes(analysis_response)
        
        highlighted_path = highlight_sentences_in_pdf(f"{UPLOAD_DIR}/{file.filename}", direct_quotes)
        
        analysis_filename = file.filename.rsplit(".", 1)[0] + "-analysis.txt"
        analysis_path = f"{user_id}/{analysis_filename}"

        # Convert analysis to bytes
        analysis_bytes = analysis_response.encode("utf-8")
        analysis_upload_response = supabase.storage.from_("legal-docs").upload(
            analysis_path,
            analysis_bytes,
            {
                "content-type": "text/plain",
                "x-upsert": "true"
            }
        )

        if hasattr(analysis_upload_response, "error") and analysis_upload_response.error:
            return {"error": f"Failed to upload analysis: {analysis_upload_response.error['message']}"}

        signed_analysis_url_response = supabase.storage.from_("legal-docs").create_signed_url(
            analysis_path,
            60 * 60
        )

        if hasattr(signed_analysis_url_response, "error") and signed_analysis_url_response.error:
            return {"error": "Failed to create signed URL for analysis"}

        analysis_txt_url = signed_analysis_url_response["signedURL"]

        return {
            "analysis": analysis_response,
            "filename": file.filename,
            "highlighted_pdf_path": highlighted_path,
            "file_url": signed_url,
        "analysis_file_url": analysis_txt_url
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
