from typing import Optional

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.base_language import BaseLanguageModel
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from langchain.schema import BaseChatMessageHistory


def get_conversation_buffer_memory(
    memory_key: str = "chat_history",
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    chat_memory: Optional[StreamlitChatMessageHistory] = None,
):
    if chat_memory:
        return ConversationBufferMemory(
            memory_key=memory_key,
            input_key=input_key,
            output_key=output_key,
            return_messages=True,
            chat_memory=chat_memory,
        )

    return ConversationBufferMemory(
        memory_key=memory_key,
        input_key=input_key,
        output_key=output_key,
        return_messages=True,
    )


def get_agenttoken_memory(
    llm: BaseLanguageModel,
    memory_key: str = "history",
    max_token_limit: int = 2000,
    chat_memory: Optional[BaseChatMessageHistory] = None,
):
    if chat_memory:
        return AgentTokenBufferMemory(
            memory_key=memory_key, llm=llm, max_token_limit=max_token_limit, chat_memory=chat_memory
        )
    return AgentTokenBufferMemory(memory_key=memory_key, llm=llm, max_token_limit=max_token_limit)


def get_conversationaltoken_memory(
    llm: BaseLanguageModel,
    memory_key: str = "history",
    max_token_limit: int = 2000,
    chat_memory: Optional[BaseChatMessageHistory] = None,
):
    if chat_memory:
        return ConversationTokenBufferMemory(
            memory_key=memory_key, llm=llm, max_token_limit=max_token_limit, chat_memory=chat_memory
        )
    return ConversationTokenBufferMemory(
        memory_key=memory_key, llm=llm, max_token_limit=max_token_limit
    )


def get_conversation_buffer_window_memory(
    memory_key: str = "chat_history",
    input_key: Optional[str] = None,
    output_key: Optional[str] = "answer",
    window_size: int = 5,
    chat_memory: Optional[StreamlitChatMessageHistory] = None,
):

    # conversation buffer memory with fixed size window
    if chat_memory:
        return ConversationBufferWindowMemory(
            memory_key=memory_key,
            input_key=input_key,
            output_key=output_key,
            return_messages=True,
            k=window_size,
            chat_memory=chat_memory,
        )

    return ConversationBufferWindowMemory(
        memory_key=memory_key,
        input_key=input_key,
        output_key=output_key,
        k=window_size,
        return_messages=True,
    )
