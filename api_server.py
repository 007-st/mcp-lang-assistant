import os
import json
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

# 1. 配置与初始化
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Configuration:
    def __init__(self) -> None:
        self.api_key: str = os.getenv("DASHSCOPE_API_KEY") or ""
        self.model: str = os.getenv("MODEL") or "qwen-plus"
        if not self.api_key:
            logging.warning("⚠️ 未找到 DASHSCOPE_API_KEY")

    @staticmethod
    def load_servers(file_path: str = "servers_config.json") -> Dict[str, Any]:
        if not os.path.exists(file_path):
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})

# 全局变量存储 agent 和 client
mcp_client: MultiServerMCPClient = None
agent = None
config = {"configurable": {"thread_id": "default"}}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动时连接 MCP，关闭时清理"""
    global mcp_client, agent
    
    logging.info("🚀 正在启动 MCP Agent API...")
    
    # 1. 读取配置
    cfg = Configuration()
    os.environ["DASHSCOPE_API_KEY"] = cfg.api_key
    servers_cfg = Configuration.load_servers()
    
    # 2. 连接 MCP 服务器
    try:
        mcp_client = MultiServerMCPClient(servers_cfg)
        tools = await mcp_client.get_tools()
        logging.info(f"✅ 已加载 {len(tools)} 个 MCP 工具: {[t.name for t in tools]}")
    except Exception as e:
        logging.error(f"❌ MCP 连接失败: {e}")
        tools = []

    # 3. 读取 Prompt
    prompt = "你是一个智能助手。"
    if os.path.exists("agent_prompts.txt"):
        with open("agent_prompts.txt", "r", encoding="utf-8") as f:
            prompt = f.read()

    # 4. 初始化 Agent
    # 彻底禁用流式，避免 LangGraph 内部索引错误
    model = ChatTongyi(model=cfg.model, streaming=False)
    checkpointer = InMemorySaver()
    
    agent = create_react_agent(
        model=model, 
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer
    )
    
    yield
    
    # 清理资源
    if mcp_client:
        await mcp_client.cleanup()
    logging.info("🧹 MCP 资源已清理")

app = FastAPI(lifespan=lifespan)

# 允许跨域（方便前端调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"

class ChatResponse(BaseModel):
    content: str
    status: str = "success"
    error: str = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """普通聊天接口（非流式），返回完整结果"""
    if not agent:
        raise HTTPException(status_code=500, detail="Agent 未初始化")
    
    run_config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        logging.info(f"收到请求: {request.message}")
        
        # 直接调用 invoke，等待完整结果
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            run_config
        )
        
        if "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            return ChatResponse(content=last_message.content)
        else:
            return ChatResponse(content="未获取到回复", status="empty")

    except Exception as e:
        logging.error(f"处理出错: {traceback.format_exc()}")
        # 返回错误信息给前端
        return ChatResponse(content="", status="error", error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
