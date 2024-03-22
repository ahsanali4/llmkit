from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

# from langchain.evaluation import Criteria, load_evaluator
from langchain.memory.chat_memory import BaseChatMemory


def create_llm_chain(prompt: PromptTemplate, llm: BaseLanguageModel, verbose: bool = False):
    return LLMChain(prompt=prompt, llm=llm, verbose=verbose)


# create the chain to answer questions
def get_qa_chain(
    llm: BaseLanguageModel,
    retriever,
    chain_type: str = "stuff",
    return_source_documents: bool = True,
):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=return_source_documents,
    )


def get_conversational_chain(
    llm: BaseLanguageModel, retriever, memory: BaseChatMemory, return_source_documents: bool = True
):
    return ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory, return_source_documents=return_source_documents, verbose=True
    )
