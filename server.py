from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

app = FastAPI()

# ─────────────────────────────────────
# EMBEDDINGS (unchanged)
# ─────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL", "openai-main/text-embedding-3-small"),
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["EMBEDDING_API_KEY"],
    default_headers={
        "X-TFY-METADATA": "{}",
        "X-TFY-LOGGING-CONFIG": '{"enabled": true}',
    },
    check_embedding_ctx_length=False,
)

# ─────────────────────────────────────
# QDRANT — replaces FAISS.load_local
# ─────────────────────────────────────
QDRANT_HOST = os.environ.get("QDRANT_HOST", "ml.tfy-eo.truefoundry.cloud")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant-vectordb-kshitij-test")
QDRANT_URL        = "https://"+QDRANT_HOST+"/"+QDRANT_PATH
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "ipl-docs")

client = QdrantClient(
    host=QDRANT_HOST,
    port=443,
    https=True,
    prefix=QDRANT_PATH,
    prefer_grpc=False,
    check_compatibility=False,
    timeout=30,
)

# No load_local needed — Qdrant is always live
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

# ─────────────────────────────────────
# LLM (unchanged)
# ─────────────────────────────────────
llm = ChatOpenAI(
    model="openai-main/gpt-5",
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["BASE_URL"],
    default_headers={
        "X-TFY-METADATA": "{}",
        "X-TFY-LOGGING-CONFIG": '{"enabled": true}',
    },
)

# ─────────────────────────────────────
# SCHEMAS (unchanged)
# ─────────────────────────────────────
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


structured_llm = llm.with_structured_output(RagAnswer)


# ─────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────
@app.post("/rag")
def rag_query(query: Query):
    # Retrieve relevant chunks from Qdrant
    docs = vectorstore.similarity_search(query.question, k=6)

    # Build context with labelled sources (unchanged)
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
    # Also verify Qdrant connection is alive
    try:
        collections = [c.name for c in client.get_collections().collections]
        return {
            "status": "ok",
            "qdrant": "connected",
            "collections": collections
        }
    except Exception as e:
        return {
            "status": "ok",
            "qdrant": "unreachable",
            "error": str(e)
        }