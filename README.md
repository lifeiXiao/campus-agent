# 🎓 Campus Agent — 校园智能助手 MVP

> 基于 RAG + 大语言模型 的校园生活 AI 助手  
> 一键问答 · 智能检索 · 云端推理 · 本地轻量运行

> 访问地址： http://39.104.79.111:8000/
---

## 🚀 功能特性
- 🧠 智能问答：快速回答校园事务问题  
- 🔍 RAG 检索：结合本地知识库增强回答准确性  
- ☁️ 云端推理：调用 ModelScope Qwen 模型生成回复  
- ⚡ 本地嵌入：轻量 CPU 模型生成向量，无需 GPU  
- 💬 极简前端：纯 HTML 聊天界面  

---

## 🧩 项目结构
```
campus-agent/
├── backend/
│ ├── main.py # FastAPI 入口
│ ├── routers/chat.py # 聊天接口
│ ├── services/
│ │ ├── llm_service.py # 云端推理服务
│ │ └── rag_service.py # 知识检索服务
│ ├── data/campus_knowledge.txt # 知识库文件
│ ├── cache/embeddings.npy # 本地缓存向量
│ └── .env # 环境变量
└── frontend-simple/
└── index.html # 前端页面
```

---

## ⚙️ 环境配置

### 1️⃣ 创建环境
```bash
conda create -n agent python=3.9
conda activate agent

pip install -r requirements.txt
MODELSCOPE_API_TOKEN=你的魔搭令牌
REMOTE_MODEL=Qwen/Qwen2.5-7B-Instruct
cd backend
``` 
## ⚙️ 启动服务
uvicorn main:app --reload --port 8000

## ⚙️ 本地访问地址
127.0.0.1：8000
