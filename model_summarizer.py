import os
import sqlite3
import urllib.request
import urllib.parse
import json
import re
import ollama

DB_PATH = os.path.expanduser("~/.config/superboof/model_info.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_uses (
            model_name TEXT PRIMARY KEY,
            best_use TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_cached_use(model_name: str) -> str:
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT best_use FROM model_uses WHERE model_name = ?", (model_name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return ""

def set_cached_use(model_name: str, best_use: str):
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO model_uses (model_name, best_use) VALUES (?, ?)", (model_name, best_use))
        conn.commit()
        conn.close()
    except Exception:
        pass

def search_huggingface(query: str) -> str:
    """Query HuggingFace models API to find tags/metadata for a model."""
    try:
        clean_query = re.sub(r'[:_-]', ' ', query).strip()
        url = f"https://huggingface.co/api/models?search={urllib.parse.quote(clean_query)}&limit=3"
        req = urllib.request.Request(url, headers={"User-Agent": "Superboof/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data:
                # Compile tags and pipeline info across first few matches
                combined_info = []
                for item in data:
                    item_id = item.get("id", "")
                    pipeline = item.get("pipeline_tag", "")
                    tags = ", ".join(item.get("tags", []))
                    combined_info.append(f"Model ID: {item_id}. Pipeline: {pipeline}. Tags: {tags}.")
                return "\n".join(combined_info)
    except Exception:
        pass
    return ""

def get_best_use_from_llm(model_name: str, model_info_text: str) -> str:
    """Use phi4-mini (or another tiny model) to summarize the model in 5 words or less."""
    prompt = (
        f"You are a concise model metadata assistant. "
        f"Summarize the best use case for the model '{model_name}' in exactly 5 words or less (lowercase only). "
        f"Use the following metadata info to guide you:\n\n{model_info_text}\n\n"
        f"If the metadata is empty or you don't know, provide a generic best use based on the name (e.g. 'general purpose text generator' or 'fast coding assistant').\n"
        f"Return ONLY the 5-word summary, nothing else. No punctuation, no quotes, no conversational filler."
    )
    try:
        # Check if phi4-mini is pulled, otherwise fallback to whatever is available or the model itself
        available = []
        try:
            res = ollama.list()
            models = res.models if hasattr(res, 'models') else res.get('models', [])
            available = [m.model if hasattr(m, 'model') else m.get('name', '') for m in models]
        except Exception:
            pass
        
        # Prefer phi4-mini, then qwen2.5-coder:3b, then whatever is available
        llm_model = "phi4-mini"
        if "phi4-mini:latest" in available or "phi4-mini" in available:
            llm_model = "phi4-mini"
        elif "qwen2.5-coder:3b" in available or "qwen2.5-coder:3b:latest" in available:
            llm_model = "qwen2.5-coder:3b"
        elif available:
            # use any small one if present, otherwise just pick first one
            llm_model = available[0]
        else:
            return "unknown model purpose"

        response = ollama.generate(
            model=llm_model,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 15}
        )
        summary = response.get("response", "").strip().lower()
        # Clean up punctuation and quotes if the model hallucinated them
        summary = re.sub(r'[^\w\s-]', '', summary)
        # Enforce 5 words limit just in case
        words = summary.split()
        if len(words) > 5:
            summary = " ".join(words[:5])
        return summary
    except Exception:
        return "general purpose text model"

def get_or_create_model_summary(model_name: str) -> str:
    cached = get_cached_use(model_name)
    if cached:
        return cached

    # 1. Search HuggingFace for metadata
    info_text = search_huggingface(model_name)
    
    # 2. Ask local tiny model to summarize
    best_use = get_best_use_from_llm(model_name, info_text)
    
    # 3. Cache it
    if best_use:
        set_cached_use(model_name, best_use)
        
    return best_use
