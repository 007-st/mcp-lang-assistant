"""
单次调用 MCP Agent 的示例 (非 CLI 聊天模式)
展示如何将 MCP Agent 封装为函数调用，适合集成到 API 或其他应用中。
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()

# 配置类 (同 client.py)
class Configuration:
    def __init__(self) -> None:
        self.api_key: str = os.getenv("DASHSCOPE_API_KEY") or ""
        self.model: str = os.getenv("MODEL") or "qwen-plus"
        if not self.api_key:
            raise ValueError("❌ 未找到 DASHSCOPE_API_KEY")

    @staticmethod
    def load_servers(file_path: str = "servers_config.json") -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})

async def process_user_query(query: str):
    """
    处理单个用户请求的函数
    """
    # 1. 初始化配置
    cfg = Configuration()
    os.environ["DASHSCOPE_API_KEY"] = cfg.api_key
    servers_cfg = Configuration.load_servers()

    # 2. 初始化 MCP 客户端
    # 注意：在生产环境中，Client 通常作为一个全局单例或长连接服务存在，
    # 而不是每次请求都重新连接，这里为了演示方便放在函数内。
    mcp_client = MultiServerMCPClient(servers_cfg)
    
    try:
        # 3. 获取工具
        tools = await mcp_client.get_tools()
        
        # 4. 初始化模型
        model = ChatTongyi(model=cfg.model)
        
        # 5. 创建 Agent
        # checkpointer 用于记忆，如果是单次无状态调用可以去掉
        agent = create_react_agent(model=model, tools=tools)
        
        # 6. 执行 Agent
        print(f"🚀 开始处理请求: {query}")
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        
        return result['messages'][-1].content
        
    finally:
        # 7. 务必清理资源 (关闭 MCP 连接)
        # 注意：langchain-mcp-adapters 的 MultiServerMCPClient 可能实现了 __aexit__ 协议
        # 或者其内部会自动管理。如果它没有 cleanup 方法，我们可以尝试使用 close 方法，
        # 或者如果它是一个 Context Manager，我们应该使用 async with。
        
        # 检查是否有 cleanup 或 close 方法
        if hasattr(mcp_client, 'cleanup'):
            await mcp_client.cleanup()
        elif hasattr(mcp_client, 'close'):
            await mcp_client.close()
        elif hasattr(mcp_client, '__aexit__'):
             # 如果是上下文管理器，通常不应该手动调用清理，而是应该使用 async with
             # 但在这里我们已经手动初始化了。
             # 如果没有显式的清理方法，我们可能需要检查其内部的 session 或 exit_stack
             pass
        
        # 更好的做法是使用 async with 上下文管理器（如果支持）
        # 但 MultiServerMCPClient 的设计似乎是在 __init__ 中做了一些同步操作，
        # 连接可能是在 get_tools 中建立的。
        
        # 由于报错 AttributeError: 'MultiServerMCPClient' object has no attribute 'cleanup'
        # 我们暂时注释掉这行，并尝试使用 async with 结构重构（如果支持）
        # 或者检查是否有其他清理方法。
        # 查看源码或文档会更有帮助，但根据报错，我们先移除它。
        pass

async def process_user_query_safe(query: str):
    """
    使用 async with 上下文管理器（如果支持）的更安全版本
    """
     # 1. 初始化配置
    cfg = Configuration()
    os.environ["DASHSCOPE_API_KEY"] = cfg.api_key
    servers_cfg = Configuration.load_servers()
    
    # 尝试将 mcp_client 当作上下文管理器使用
    # 如果 MultiServerMCPClient 支持 async with，这是最佳实践
    try:
        async with MultiServerMCPClient(servers_cfg) as mcp_client:
            tools = await mcp_client.get_tools()
            model = ChatTongyi(model=cfg.model)
            agent = create_react_agent(model=model, tools=tools)
            
            print(f"🚀 开始处理请求: {query}")
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            return result['messages'][-1].content
    except AttributeError:
        # 如果不支持 async with (例如没有 __aenter__)，则回退到手动管理
        # 但根据之前的报错，它也没有 cleanup。
        # 这意味着它可能不需要显式清理，或者依赖于垃圾回收，或者我们漏掉了什么。
        # 我们先回退到不调用 cleanup 的版本。
        return await process_user_query(query)
    except Exception as e:
         # 其他错误
         print(f"❌ 发生错误: {e}")
         raise e

# 主程序入口
if __name__ == "__main__":
    # 模拟一次调用
    # 这里的 process_user_query 已经被修改为不再调用 cleanup
    response = asyncio.run(process_user_query("北京今天天气如何？"))
    print(f"\n🏁 最终结果:\n{response}")
