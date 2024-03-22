import os

from langfuse.callback import CallbackHandler


def get_langfuse_handler():
    """Use to monitor traces for chains.
    It should be included as a callback handler with each chain or llm

    Returns:
        _type_: _description_
    """
    return CallbackHandler(
        os.getenv('PUBLIC_KEY'), os.getenv('SECRET_KEY')
    )
