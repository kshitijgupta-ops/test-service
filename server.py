from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# Load embeddings using your TF deployed model
embeddings = OpenAIEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL", "openai-main/text-embedding-3-small"),
    base_url=os.environ["BASE_URL"],   # full Gateway Base URL (e.g. https://<host>/api/llm)
    api_key=os.environ["EMBEDDING_API_KEY"],      # store as env var
    default_headers={
        "X-TFY-METADATA": "{}",
        "X-TFY-LOGGING-CONFIG": '{"enabled": true}',
    },
    # Gateway handles tokenization; don't let langchain pre-tokenize into token IDs
    check_embedding_ctx_length=False,
)

if os.environ.get("ENV", "staging") == "dev":
    # local paths for development, resolved relative to this script
    vector_store_path = os.path.join(os.path.dirname(__file__), "vector-store")
else:
    vector_store_path = "/mnt/rag/vector-store"

# Load vector store from mounted volume
vectorstore = FAISS.load_local(
    vector_store_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# LLM using your exact code
llm = ChatOpenAI(
    model="openai-main/gpt-5",
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["BASE_URL"],
    default_headers={
        "X-TFY-METADATA": "{}",
        "X-TFY-LOGGING-CONFIG": '{"enabled": true}',
    },
)

class Query(BaseModel):
    question: str


class Citation(BaseModel):
    source: str = Field(description="Name of the source document the information came from")
    page: Optional[int] = Field(default=None, description="Page number in the source document, if known")
    snippet: Optional[str] = Field(default=None, description="Short verbatim quote from the source supporting the answer")


class RagAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the question, based only on the provided context")
    citations: List[Citation] = Field(default_factory=list, description="Sources from the context that support the answer")
    additional_info: Optional[str] = Field(default=None, description="Any extra relevant details found in the context; null if none")


# Ask the model to return data matching the RagAnswer schema (reliable JSON)
structured_llm = llm.with_structured_output(RagAnswer)


@app.post("/rag")
def rag_query(query: Query):
    # Retrieve relevant chunks from vector store
    docs = vectorstore.similarity_search(query.question, k=6)

    # Build context with labelled sources so the model can cite precisely
    context_blocks = []
    for i, d in enumerate(docs, start=1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        context_blocks.append(f"[Source {i}: {src}{page_label}]\n{d.page_content}")
    context = "\n\n".join(context_blocks)

    messages = [
        SystemMessage(content=(
            "You are an AI assistant. Answer the question using ONLY the provided context. "
            "Always cite the source(s) you used in the citations field. "
            "If the answer is not present in the context, say so in the answer field and return an empty citations list.\n\n"
            f"Context:\n{context}"
        )),
        HumanMessage(content=query.question),
    ]

    try:
        result = structured_llm.invoke(messages)
        return result.model_dump()
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}