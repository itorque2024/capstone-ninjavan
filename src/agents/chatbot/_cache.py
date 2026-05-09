from joblib import Memory
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".chatbot_cache"
memory = Memory(_CACHE_DIR, verbose=0)
