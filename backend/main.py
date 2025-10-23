from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers import chat
import os

app = FastAPI()

# CORS 设置：允许前端访问后端接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册聊天路由
app.include_router(chat.router, prefix="/api")

# 🚀 前端页面目录 (frontend-simple)
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend-simple")

# 检查前端是否存在
if not os.path.exists(frontend_path):
    raise RuntimeError(f"❌ 前端目录不存在: {frontend_path}")

# 绑定静态文件目录到根路径
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

@app.get("/health")
def health():
    return {"message": "Campus Agent API is running"}
