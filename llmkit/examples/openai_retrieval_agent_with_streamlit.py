import streamlit as st
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from torch import cuda

from llm_chatbot_devs.core.Agents.openai_conversational_retrieval_agent import (
    create_openai_document_retrieval_agent,
)
from llm_chatbot_devs.core.Agents.tools.openai_retrieval_tool import create_ret_tool
from llm_chatbot_devs.core.embeddings import get_instruct_embeddings
from llm_chatbot_devs.utils.preprocessing import load_chroma_db

BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/"  # to use on colab
COLLECTION_NAME = "sustain-xlarge"  # already stored embeddings db name.
DB_PATH = BASE_PATH + "sustain"
PROMPT = """You are a chatbot helping user to understand the annual report of a company\
Heidelberg Materials. Whenever user ask anything about the company try to find it\
using the available tools and return the relevant information.If you are unable\
to find the information, excuse for having not enough information and act accordingly."""


def main():
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
    embedder = get_instruct_embeddings(model_kwargs={"device": device})
    vectordb = load_chroma_db(
        embedding_model=embedder, collection_name=COLLECTION_NAME, persist_directory=DB_PATH
    )
    retriever = vectordb.as_retriever()
    tool_name = "heidelberg_materials_sustainability_report"
    tool_description = "Searches and returns documents regarding the annual\
          sustainability report of the heidelberg material."
    retriever_tool = create_ret_tool(
        retriever=retriever, tool_name=tool_name, description=tool_description
    )
    tools = [retriever_tool]
    msgs = StreamlitChatMessageHistory(key="chat_history_streamlit")
    agent_executor = create_openai_document_retrieval_agent(
        tools=tools,
        prompt=PROMPT,
        memory_key="history",
        chat_memory=msgs,
        remember_intermediate_steps=True,
    )
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    if query := st.chat_input():
        st.chat_message("human", avatar="👤").write(query)

        # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
        response = agent_executor({"input": query})
        st.chat_message("ai", avatar="🤖").write(response)


if __name__ == "__main__":
    main()
