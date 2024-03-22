import os
import tempfile

import fitz
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import PyMuPDFLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from llm_chatbot_devs.core.embeddings import get_instruct_embeddings
from llm_chatbot_devs.core.langchain import (
    get_conversation_buffer_memory,
    get_conversational_chain,
)
from llm_chatbot_devs.core.llms import get_chatopenai_llm
from llm_chatbot_devs.utils.preprocessing import create_chroma_db, text_splitter

st.set_page_config(page_title="KBot", page_icon="💀")
st.title(" KBot: Document analysis helper")


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        mu_loader = PyMuPDFLoader(temp_filepath)
        mu_data = mu_loader.load(flags=fitz.TEXTFLAGS_SEARCH, sort=True)
        docs.extend(mu_data)

    # Split documents
    splitted_text = text_splitter(documents=docs, chunk_size=500, chunk_overlap=50)

    # Create embeddings and store in vectordb
    embedder = get_instruct_embeddings(model_kwargs={"device": "cuda"})
    vectordb = create_chroma_db(
        embedding=embedder,
        texts=splitted_text,
        collection_name="sustain",
        persist_directory="streamlit",
    )

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr")

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    # def on_chain_end(
    #     self,
    #     outputs : dict,
    #     **kwargs
    # ) -> None:
    #   if 'answer' in outputs:
    #     self.text = outputs['answer']
    #     self.container.markdown(self.text)
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = get_conversation_buffer_memory(chat_memory=msgs)

# Setup LLM and QA chain
llm = get_chatopenai_llm(temperature=0.0, streaming=True)
conversational_chain = get_conversational_chain(
    llm, retriever, memory, return_source_documents=False
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = conversational_chain.run(
            user_query, callbacks=[retrieval_handler, stream_handler]
        )
