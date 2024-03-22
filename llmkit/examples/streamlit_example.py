import fitz
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from llm_chatbot_devs.core.embeddings import get_instruct_embeddings
from llm_chatbot_devs.core.langchain import (
    get_conversation_buffer_memory,
    get_conversational_chain,
)
from llm_chatbot_devs.core.llms import create_hf_pipeline, get_falcon40b
from llm_chatbot_devs.utils.preprocessing import create_chroma_db, text_splitter


def load_file(
    file_path: str = "/content/drive/MyDrive/Colab Notebooks\
    /HM_Annual_and_Sustainability_Report_2022.pdf",
):
    mu_loader = PyMuPDFLoader(file_path)
    mu_data = mu_loader.load(flags=fitz.TEXTFLAGS_SEARCH, sort=True)
    return mu_data


def main():
    st.title("Sustainability report- chatbot")

    documents = load_file()
    splitted_text = text_splitter(documents=documents, chunk_size=500, chunk_overlap=50)
    embedder = get_instruct_embeddings(model_kwargs={"device": "gpu"})
    vectordb = create_chroma_db(
        embedding=embedder,
        texts=splitted_text,
        collection_name="sustain",
        persist_directory="streamlit",
    )
    model, tockenizer, stopping_criteria = get_falcon40b(chat_model=True)
    llm = create_hf_pipeline(model=model, tokenizer=tockenizer, stopping_criteria=stopping_criteria)
    msgs = StreamlitChatMessageHistory(key="chat_history_streamlit")

    memory = get_conversation_buffer_memory(output_key="answer", chat_memory=msgs)
    conversational_chain = get_conversational_chain(llm, vectordb.as_retriever(), memory)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    if query := st.chat_input():
        st.chat_message("human", avatar="👤").write(query)

        # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
        response = conversational_chain({"question": query})
        st.chat_message("ai", avatar="🤖").write(response)


if __name__ == "__main__":
    main()
