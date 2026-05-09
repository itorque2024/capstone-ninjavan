from pathlib import Path
import chromadb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RAG_DIR = _PROJECT_ROOT / "data" / "rag_documents"
_DEFAULT_CHROMA_PATH = str(_PROJECT_ROOT / "chroma_db")

# Map filename keywords → collection name
_COLLECTION_MAP = {
    "track":         "nv_tracking",
    "delivery":      "nv_delivery",
    "reschedul":     "nv_delivery",
    "notification":  "nv_delivery",
    "sla":           "nv_delivery",
    "claims":        "nv_claims",
    "refund":        "nv_claims",
    "proof":         "nv_claims",
    "compensation":  "nv_claims",
    "returns":       "nv_policy",
    "prohibited":    "nv_policy",
    "size":          "nv_policy",
    "packaging":     "nv_policy",
    "cod":           "nv_policy",
    "international": "nv_policy",
    "premium":       "nv_policy",
}
ALL_COLLECTIONS = ["nv_tracking", "nv_delivery", "nv_claims", "nv_policy", "nv_general"]


def _classify(filename: str) -> str:
    fname = filename.lower()
    for keyword, collection in _COLLECTION_MAP.items():
        if keyword in fname:
            return collection
    return "nv_general"


def build_chroma_from_files(rag_dir=None, chroma_path=None):
    """Rebuild all 5 topic collections from .txt files. Called at app startup when empty."""
    docs_dir = Path(rag_dir) if rag_dir else _DEFAULT_RAG_DIR
    chroma_path = chroma_path or _DEFAULT_CHROMA_PATH

    doc_files = sorted(docs_dir.glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")

    client = chromadb.PersistentClient(path=chroma_path)

    # Delete and recreate all collections
    for col_name in ALL_COLLECTIONS:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    collections = {
        name: client.create_collection(name, metadata={"hnsw:space": "cosine"})
        for name in ALL_COLLECTIONS
    }

    # Also keep a unified collection for backward compatibility
    try:
        client.delete_collection("ninjavan_kb")
    except Exception:
        pass
    unified = client.create_collection("ninjavan_kb", metadata={"hnsw:space": "cosine"})

    buckets = {name: ([], [], []) for name in ALL_COLLECTIONS}

    for f in doc_files:
        text = f.read_text()
        topic = text.split("\n")[0].replace("Topic: ", "").strip()
        col_name = _classify(f.name)
        ids, texts, metas = buckets[col_name]
        ids.append(f.stem)
        texts.append(text)
        metas.append({"topic": topic, "source": f.name, "collection": col_name})

    total = 0
    for col_name, (ids, texts, metas) in buckets.items():
        if ids:
            collections[col_name].add(ids=ids, documents=texts, metadatas=metas)
            unified.add(ids=ids, documents=texts, metadatas=metas)
            total += len(ids)
            print(f"  {col_name}: {len(ids)} documents")

    print(f"ChromaDB ready: {total} documents across {len(ALL_COLLECTIONS)} collections at {chroma_path}")
    return collections


def query_collection(collection_name: str, query: str, n_results: int = 3,
                     chroma_path: str = None) -> list[str]:
    """Query a specific collection and return document texts."""
    path = chroma_path or _DEFAULT_CHROMA_PATH
    client = chromadb.PersistentClient(path=path)
    try:
        col = client.get_collection(collection_name)
        results = col.query(query_texts=[query], n_results=min(n_results, col.count()))
        return results["documents"][0] if results["documents"] else []
    except Exception:
        # Fallback to unified collection
        try:
            col = client.get_collection("ninjavan_kb")
            results = col.query(query_texts=[query], n_results=n_results)
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []


def is_chroma_ready(chroma_path: str = None) -> bool:
    path = chroma_path or _DEFAULT_CHROMA_PATH
    try:
        client = chromadb.PersistentClient(path=path)
        col = client.get_collection("ninjavan_kb")
        return col.count() > 0
    except Exception:
        return False
