from typing import List, Optional

from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.base_language import BaseLanguageModel
from langchain.prompts import MessagesPlaceholder
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import SystemMessage
from langchain.tools.base import BaseTool

from llmkit.core.langchain import get_agenttoken_memory
from llmkit.core.llms import get_chatopenai_llm
from llmkit.core.prompts import prompt_message_agent_retrieval


def _get_default_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary"
        )
    )


def get_conversational_retrieval_agent(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    remember_intermediate_steps: bool = True,
    system_message: Optional[SystemMessage] = None,
):
    return create_conversational_retrieval_agent(
        llm, tools, system_message=system_message, verbose=True
    )


def create_openai_document_retrieval_agent(
    tools: List[BaseTool],
    prompt: Optional[str],
    memory_key: str,
    chat_memory: Optional[BaseChatMessageHistory] = None,
    remember_intermediate_steps: bool = True,
):
    llm = get_chatopenai_llm(temperature=0.0)
    system_message = _get_default_system_message()
    if prompt:
        system_message = prompt_message_agent_retrieval(prompt)

    memory = get_agenttoken_memory(memory_key=memory_key, llm=llm, chat_memory=chat_memory)
    system_prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=system_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=remember_intermediate_steps,
    )
    return agent_executor
