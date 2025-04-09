#!/usr/bin/env python3
import os
import json
import time
import re
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import uvicorn
from fastapi.middleware.cors import CORSMiddleware  # Add this for CORS support

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM

# frontend files
from fastapi.staticfiles import StaticFiles



from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



CONFIG = {
    "upload_dir": "/app/uploads",
    "temp_dir": "/app/temp_results",
    "chunk_size": int(os.getenv("CHUNK_SIZE", "1500")),  # Default to 1500 and convert to int
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),  # Fixed typo and default to 150
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),
    "timeout": int(os.getenv("TIMEOUT", "60")),
    "max_requests_per_min": int(os.getenv("MAX_REQUESTS_PER_MIN", "10")),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama2"),
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "min_quality_score": float(os.getenv("MIN_QUALITY_SCORE", "0.7")),
    "qa_per_chunk": int(os.getenv("QA_PER_CHUNK", "3")),
    "result_ttl": 3 * 24 * 60 * 60,  # 3 days in seconds
    "max_pairs_per_file": 10000,  # Split output into files of 10k QA pairs
    "preview_size": 50,  # Number of rows to show in preview
}

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Synthetic Data Generation Service")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory task storage
tasks: Dict[str, Dict] = {}

# Ensure directories exist
Path(CONFIG["upload_dir"]).mkdir(exist_ok=True)
Path(CONFIG["temp_dir"]).mkdir(exist_ok=True)

# Pydantic Models
class QAGeneration(BaseModel):
    question: str = Field(..., description="Generated question")
    answer: str = Field(..., description="Accurate answer")

class QAPair(BaseModel):
    question: str
    answer: str
    context: str
    source: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskResult(BaseModel):
    task_id: str
    status: str
    result_files: Optional[List[str]] = None
    download_url: Optional[str] = None
    preview: Optional[List[Dict]] = None
    progress: Optional[float] = None
    total_pairs: Optional[int] = None

# QA Generator Class
class QAGenerator:
    def __init__(self):
        self.llm = OllamaLLM(base_url=CONFIG['ollama_base_url'], model=CONFIG["ollama_model"], temperature=0.3, timeout=CONFIG["timeout"])
        self.request_count = 0
        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(CONFIG["max_retries"]),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.RequestException, ValidationError))
    )
    def generate_qa(self, context: str, source: str) -> Optional[QAPair]:
        self._rate_limit()
        prompt = f"""Generate a medical factoid question and answer pair from the following context.
Output a JSON object with "question" and "answer" keys only.
Context: {context}"""
        try:
            response = self.llm.invoke(prompt)
            logger.info(f"Raw LLM response: {response}")
            output_json = json.loads(response[response.find("{"):response.rfind("}") + 1])
            validated = QAGeneration(**output_json)
            return QAPair(question=validated.question, answer=validated.answer, context=context, source=source)
        except Exception as e:
            logger.error(f"QA generation failed: {str(e)}")
            return None

    def critique_qa(self, qa_pair: QAPair) -> float:
        self._rate_limit()
        critique_prompt = f"""Rate this QA pair (1-5 average score) based on groundedness, relevance, and clarity:
Question: {qa_pair.question}
Answer: {qa_pair.answer}
Context: {qa_pair.context}
Output only a number."""
        try:
            response = self.llm.invoke(critique_prompt)
            logger.info(f"Critique response: {response}")
            match = re.search(r"(\d+(?:\.\d+)?)", response)
            return float(match.group(1)) if match else 0.0
        except Exception as e:
            logger.error(f"Critique failed: {str(e)}")
            return 0.0

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        self.request_count += 1
        if self.request_count >= CONFIG["max_requests_per_min"]:
            time.sleep(max(60 - elapsed, 0))
            self.request_count = 0
            self.last_request_time = time.time()

# Dataset Builder Class
class DatasetBuilder:
    def __init__(self, file_paths: List[str], task_id: str):
        self.generator = QAGenerator()
        self.qa_pairs = []
        self.errors = []
        self.file_paths = file_paths
        self.task_id = task_id

    def process_documents(self):
        all_docs = []
        for file_path in self.file_paths:
            docs = self._load_and_split_documents(file_path)
            all_docs.extend(docs)
        
        total_chunks = len(all_docs)
        logger.info(f"Processing {total_chunks} chunks from {len(self.file_paths)} files for task {self.task_id}")
        
        tasks[self.task_id]["progress"] = 0.0
        tasks[self.task_id]["status"] = "processing"

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._process_chunk, doc.page_content, os.path.basename(doc.metadata["source"])): doc
                for doc in all_docs
            }
            for i, future in enumerate(tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks")):
                try:
                    results = future.result()
                    if results:
                        self.qa_pairs.extend(results)
                except Exception as e:
                    self.errors.append(str(e))
                tasks[self.task_id]["progress"] = (i + 1) / total_chunks * 100

    def _load_and_split_documents(self, file_path: str) -> List:
        file_extension = Path(file_path).suffix.lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        # elif file_extension in [".doc", ".docx"]:
        #     loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["chunk_overlap"])
        return loader.load_and_split(text_splitter=splitter)

    def _process_chunk(self, text: str, source: str) -> List[Dict]:
        qa_entries = []
        for _ in range(CONFIG["qa_per_chunk"]):
            qa_pair = self.generator.generate_qa(text, source)
            if not qa_pair:
                continue
            score = self.generator.critique_qa(qa_pair)
            logger.info(f"QA pair: {qa_pair}, Score: {score}")
            if score >= CONFIG["min_quality_score"]:
                qa_entries.append(qa_pair.model_dump())
        return qa_entries

    def save_result(self):
        result_dir = Path(CONFIG["temp_dir"]) / self.task_id
        result_dir.mkdir(exist_ok=True)
        result_files = []
        
        # Split QA pairs into chunks
        for i in range(0, len(self.qa_pairs), CONFIG["max_pairs_per_file"]):
            chunk = self.qa_pairs[i:i + CONFIG["max_pairs_per_file"]]
            chunk_file = result_dir / f"part_{i // CONFIG['max_pairs_per_file']}.json"
            with open(chunk_file, "w") as f:
                json.dump({"qa_pairs": chunk, "errors": self.errors if i == 0 else []}, f)
            result_files.append(str(chunk_file))
        
        # Create a zip file containing all parts
        zip_path = Path(CONFIG["temp_dir"]) / f"{self.task_id}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in result_files:
                zipf.write(file, os.path.basename(file))
        
        logger.info(f"Saved {len(self.qa_pairs)} QA pairs across {len(result_files)} files, zipped to {zip_path}")
        return str(zip_path), result_files, self.qa_pairs[:CONFIG["preview_size"]]

def process_task(file_paths: List[str], task_id: str, background_tasks: BackgroundTasks):
    try:
        builder = DatasetBuilder(file_paths, task_id)
        builder.process_documents()
        zip_path, result_files, preview = builder.save_result()
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result_files"] = result_files
        tasks[task_id]["zip_path"] = zip_path
        tasks[task_id]["total_pairs"] = len(builder.qa_pairs)
        tasks[task_id]["preview"] = preview
        logger.info(f"Task {task_id} completed")
        background_tasks.add_task(cleanup_old_results)
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

def cleanup_old_results():
    now = datetime.now()
    for task_dir in Path(CONFIG["temp_dir"]).glob("task_*"):
        if task_dir.is_dir():
            creation_time = datetime.fromtimestamp(task_dir.stat().st_ctime)
            if now - creation_time > timedelta(seconds=CONFIG["result_ttl"]):
                shutil.rmtree(task_dir)
                logger.info(f"Deleted expired result directory: {task_dir}")
        elif task_dir.suffix == ".zip":
            creation_time = datetime.fromtimestamp(task_dir.stat().st_ctime)
            if now - creation_time > timedelta(seconds=CONFIG["result_ttl"]):
                task_dir.unlink()
                logger.info(f"Deleted expired result zip: {task_dir}")


# Add a root endpoint to serve the index.html
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

    
# API Endpoints
@app.post("/generate", response_model=TaskResponse)
async def generate_synthetic_data(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    # supported_types = [".pdf", ".doc", ".docx", ".txt"]
    supported_types = [".pdf",".txt"]
    task_id = f"task_{int(time.time() * 1000)}"
    file_paths = []

    for file in files:
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in supported_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        file_path = Path(CONFIG["upload_dir"]) / f"{task_id}_{len(file_paths)}{file_extension}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(file_path)

    tasks[task_id] = {"status": "queued", "progress": 0.0}
    background_tasks.add_task(process_task, file_paths, task_id, background_tasks)
    
    return TaskResponse(
        task_id=task_id,
        status="queued",
        message=f"Task queued for processing {len(file_paths)} files"
    )

@app.get("/task/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        logger.error(f"Task {task_id} not found")
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or expired")

    task = tasks[task_id]
    status = task.get("status", "queued")
    progress = task.get("progress", 0.0)
    base_url = "http://localhost:8000"  # Adjust this based on your deployment

    if status == "completed":
        return TaskResult(
            task_id=task_id,
            status="completed",
            result_files=task.get("result_files"),
            download_url=f"{base_url}/task/{task_id}/download",
            preview=task.get("preview"),
            progress=100.0,
            total_pairs=task.get("total_pairs")
        )
    elif status == "failed":
        error = task.get("error", "Unknown error")
        raise HTTPException(status_code=500, detail=f"Task failed: {error}")
    else:
        return TaskResult(task_id=task_id, status=status, progress=progress)

@app.get("/task/{task_id}/download")
async def download_task_result(task_id: str):
    if task_id not in tasks or tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or not completed")
    
    zip_path = tasks[task_id].get("zip_path")
    if not Path(zip_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{task_id}_results.zip"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)