from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

app = FastAPI()
VECTORSTORE_DIR = "vectorstore"

embedding = OpenAIEmbeddings()
llm = ChatOpenAI()

class RegenerateRequest(BaseModel):
    question: str
    count: int = 3

class DoubtRequest(BaseModel):
    doubt: str

class TopicRequest(BaseModel):
    topic: str
    count: int = 5

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[List[str]]] = []

# Extract text from PDF
def extract_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Upload PDF
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    path = f"data/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    text = extract_text(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    db = FAISS.from_texts(chunks, embedding)
    db.save_local(VECTORSTORE_DIR)

    return {"message": "Uploaded and processed!"}

# Load vectorstore from disk
def load_vectorstore():
    return FAISS.load_local(VECTORSTORE_DIR, embedding, allow_dangerous_deserialization=True)

# Get a retriever from the saved FAISS index
def get_retriever(k=4):
    db = load_vectorstore()
    return db.as_retriever(search_kwargs={"k": k})

# Ask question
@app.get("/ask/")
def ask(q: str):
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa_chain.run(q)
    return {"answer": answer}

# Solve a doubt using the uploaded document
@app.post("/solve/")
def solve_doubt(request: DoubtRequest):
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(request.doubt)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        f"You are an expert tutor. Use the retrieved document content below to answer the student's doubt clearly and concisely.\n\n"
        f"{context}\n\nDoubt: {request.doubt}"
    )
    answer = llm.predict(prompt)
    return {"answer": answer}

# Regenerate or rephrase a question based on document content
@app.post("/regenerate/")
def regenerate_question(request: RegenerateRequest):
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(request.question)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        f"You are an exam question generator. Based on the retrieved document content below, create {request.count} alternate questions that test the same concept. "
        f"Use clear, exam-style wording and keep the meaning aligned with the original question.\n\n"
        f"Document context:\n{context}\n\nOriginal question: {request.question}"
    )
    result = llm.predict(prompt)
    return {"regenerated_questions": result}

# Generate MCQ
@app.get("/mcq/")
def mcq(topic: str):
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(topic)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        f"Create 5 multiple-choice questions from the retrieved content below. Include the correct answer for each question.\n\n"
        f"{context}\n\nTopic: {topic}"
    )

    result = llm.predict(prompt)

    return {"mcqs": result}

# Conversational QA using RAG
@app.post("/chat/")
def chat(request: ChatRequest):
    retriever = get_retriever()
    conv_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    response = conv_chain({"question": request.question, "chat_history": request.chat_history})
    return {
        "answer": response.get("answer"),
        "chat_history": response.get("chat_history", request.chat_history)
    }