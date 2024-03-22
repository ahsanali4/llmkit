import os
import tempfile

import fitz
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import PyMuPDFLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from transformers import StoppingCriteriaList

from llm_chatbot_devs.core.embeddings import get_hf_embeddings
from llm_chatbot_devs.core.langchain import get_conversational_chain
from llm_chatbot_devs.core.llms import create_hf_pipeline, get_llama2_german
from llm_chatbot_devs.core.memory import get_conversation_buffer_window_memory
from llm_chatbot_devs.utils.preprocessing import create_chroma_db, text_splitter

st.set_page_config(page_title="KBot")
st.title(" KBot: Document analysis helper")
COLLECTION_NAME = "washingmachine"
USECASE_NAME = "washingmachine"  # used as a persistant directory for vector db

msgs = StreamlitChatMessageHistory()


@st.cache_resource(ttl="1h")
def configure_bot(uploaded_files):
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
    embedder = get_hf_embeddings(model_kwargs={"device": "cuda"})
    vectordb = create_chroma_db(
        embedding=embedder,
        texts=splitted_text,
        collection_name=COLLECTION_NAME,
        persist_directory=USECASE_NAME,
    )

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr")

    # Setup memory for contextual conversation
    memory = get_conversation_buffer_window_memory(output_key="answer", chat_memory=msgs)

    # Setup LLM and QA chain
    model, tockenizer = get_llama2_german()
    llm = create_hf_pipeline(
        model=model,
        tokenizer=tockenizer,
        stopping_criteria=StoppingCriteriaList([]),
        streaming=False,
    )
    conversational_chain = get_conversational_chain(
        llm, retriever, memory, return_source_documents=False
    )

    return conversational_chain


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        if "answer" in outputs:
            self.text = outputs["answer"]
            self.container.markdown(self.text)

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        print("Prompts : ", prompts)
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

conversational_chain = configure_bot(uploaded_files)


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
