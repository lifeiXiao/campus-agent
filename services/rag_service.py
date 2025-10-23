"""
rag_service.py — 本地轻量级 RAG 模块（国内源+缓存）
Author: Code GPT 🧠
"""

import os
import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === 1️⃣ 统一项目路径 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

for d in [MODEL_DIR, DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "campus_knowledge.txt")
EMBED_PATH = os.path.join(CACHE_DIR, "embeddings.npy")

# === 2️⃣ 模型加载（使用 ModelScope 镜像） ===
print("⚙️ 正在加载轻量语义模型 (MiniLM-L12-v2, 来自魔塔)...")

model_path = snapshot_download(
    model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    cache_dir=MODEL_DIR
)

embedder = SentenceTransformer(model_path)

# === 3️⃣ 加载知识库 ===
def load_knowledge():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ 知识文件不存在: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks

knowledge_chunks = load_knowledge()

# === 4️⃣ Embedding 缓存 ===
if os.path.exists(EMBED_PATH):
    print("📂 检测到缓存 embeddings，正在加载 ...")
    embeddings = np.load(EMBED_PATH)
else:
    print("⚙️ 正在生成知识向量（本地CPU计算）...")
    embeddings = embedder.encode(knowledge_chunks)
    np.save(EMBED_PATH, embeddings)
    print(f"💾 向量已缓存至: {EMBED_PATH}")

print(f"✅ 已加载 {len(embeddings)} 条知识片段。")

# === 5️⃣ 相似度检索 ===
def search_relevant_knowledge(query: str, top_k: int = 2):
    query_vec = embedder.encode([query])
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [knowledge_chunks[i] for i in top_indices]
