from langchain.agents.agent_toolkits import create_retriever_tool


def create_ret_tool(retriever, tool_name: str, description: str):
    tool = create_retriever_tool(retriever, tool_name, description)
    return tool
