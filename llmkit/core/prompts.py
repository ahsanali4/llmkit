from typing import Any, List

from langchain import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema.messages import SystemMessage

# Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
context = "CONTEXT:/n/n {context}/n"
INSTRUCTIONS = """
CONTEXT:/n/n {context}/n

Frage: {question}"""

DEFAULT_SYSTEM_PROMPT = """\
    Sie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent. \
    Antworten Sie immer so hilfreich wie möglich und achten Sie dabei auf die Sicherheit.\
    Ihre Antworten sollten keine schädlichen, unethischen, rassistischen, sexistischen, giftigen,\
    gefährlichen oder illegalen Inhalte enthalten. Bitte achten Sie darauf,\
    dass Ihre Antworten sozial unvoreingenommen und positiv sind.

    Wenn eine Frage keinen Sinn ergibt oder sachlich nicht kohärent ist, erklären Sie, warum,\
    anstatt etwas Falsches zu beantworten. Wenn Sie die Antwort auf eine Frage nicht wissen,\
    machen Sie bitte keine falschen Angaben.

"""

_NEW_EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned \
in the following passage together with their properties.

Only extract the properties mentioned in the 'format_instructions' and return in the Json format.
format_instructions:
{format_instructions}

Passage:
{input}
"""

_NEW_EXTRACTION_TEMPLATE_GERMAN = """
Extrahieren und speichern Sie die relevanten Entitäten, die \
in der folgenden Passage erwähnt werden, zusammen mit ihren Eigenschaften.

Extrahieren Sie nur die in den "format_instructions" genannten Eigenschaften.
Extrahieren Sie die Informationen immer im Json-Format.
format_instructions:
{format_instructions}

Passage:
{input}
"""


_NEW_NER_EXTRACTION_TEMPLATE = """You are helpful information extraction system.
Given the passage, your task is to extract all entities and identify thier entity type.
the output should be in a list of tuples of the followinfg format:
[("entity 1", "type of entity 1"),....].

Passage :
{input}
"""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def get_llama2_13_chat_german_prompt(
    sys_prompt: str = DEFAULT_SYSTEM_PROMPT, instructions: str = INSTRUCTIONS
):
    return get_prompt(instructions, sys_prompt)


def prompt_with_chat():
    template = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a long document and a question, create a final answer.
    if you don't find the answer from document excuse for not having enough
    information.

    {context}
    Human: {query}
    Chatbot:"""

    prompt = PromptTemplate(input_variables=["query", "context"], template=template)
    return prompt


def prompt_with_chat_history():
    template = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    return prompt


def chat_prompt_with_history(message_prompts: List[Any]):
    return ChatPromptTemplate.from_messages(message_prompts)


def chat_prompt_with_history_a4ba():
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """You are a helper bot helping humans with a concrete answer of the asked questions.
            Start by greeting the human and then by asking how can i help you?.
            Based on the question, extract the information from the context and history
            to answer the question.
            if the information is not available in context and history just say
            "unfortuantely, this information is not available".

            After each answer, ask the human if he/she wants to clarify any information
            from the answer and clarify it.

            At the end, if there are no more questions say Good Bye!
            """
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    return prompt


def prompt_message_agent_retrieval(system_prompt: str):
    system_message = SystemMessage(content=system_prompt)
    return system_message
