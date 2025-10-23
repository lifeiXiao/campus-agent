"""
数据库模块（占位版）
"""

async def save_message(user_id: str, question: str, answer: str):
    print(f"[DB MOCK] {user_id}: Q='{question}' | A='{answer}'")
    return True
