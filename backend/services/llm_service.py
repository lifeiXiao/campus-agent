"""
llm_service.py — 使用 ModelScope API-Inference (OpenAI兼容接口) + RAG 检索增强
完全云端推理，不占用本地算力。
Author: Code GPT 🧠
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from services.rag_service import search_relevant_knowledge  # 🧩 引入 RAG 检索模块

# 读取环境变量
load_dotenv()

MODEL_ID = os.getenv("REMOTE_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
MODELSCOPE_TOKEN = os.getenv("MODELSCOPE_API_TOKEN")

if not MODELSCOPE_TOKEN:
    raise ValueError("❌ 未检测到 MODELSCOPE_API_TOKEN，请前往 modelscope.cn/my/myaccesstoken 获取。")

# 初始化 OpenAI 客户端（ModelScope 兼容接口）
client = OpenAI(
    api_key=MODELSCOPE_TOKEN,
    base_url="https://api-inference.modelscope.cn/v1/"
)

async def get_llm_response(question: str) -> str:
    """
    使用 ModelScope API + 本地知识库 RAG 进行智能回答
    """
    try:
        # 🧠 Step 1: 从知识库检索最相关内容
        relevant_docs = search_relevant_knowledge(question)
        context_text = "\n".join(relevant_docs) if relevant_docs else "（暂无相关资料）"

        # 🧩 Step 2: 构造增强提示词
        system_prompt = (
            "你是一个校园智能助手，请结合以下校园知识库中的内容回答问题。\n"
            "如果知识库中没有提及，请用礼貌的方式说明无法确定。\n\n"
            f"【校园知识库】\n{context_text}\n\n"
            "【请根据以上资料回答问题】"
        )

        # 🧠 Step 3: 调用 ModelScope 云端模型生成回答
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
        )

        # 🧾 Step 4: 返回模型输出
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ModelScope API Error]: {e}")
        return "抱歉，校园助手暂时无法回答这个问题。"
