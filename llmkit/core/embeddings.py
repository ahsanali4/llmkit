from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings


def get_instruct_embeddings(
    model_name: str = "hkunlp/instructor-xl",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
) -> HuggingFaceInstructEmbeddings:
    """Wrapper around sentence_transformers embedding models.
        https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html
    Args:
        model_name (str, optional): model name to use. Defaults to "hkunlp/instructor-large".
        model_kwargs (dict, optional): model to run on cpu or gpu. Defaults to {'device': 'cpu'}.
        encode_kwargs (dict, optional): normalize the embeddings.

    Returns:
        HuggingFaceInstructEmbeddings: _description_
    """
    return HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


# def get_openai_embeddings(openai_api_key: str,):
#     from langchain.embeddings import OpenAIEmbeddings

#     return OpenAIEmbeddings(openai_api_key="...")


def get_hf_embeddings(model_name: str = "setu4993/LaBSE", model_kwargs={"device": "cpu"}):
    """Get any embedding model available on HuggingFace

    Args:
        model_name (str, optional): HF model name to load and use. Defaults to "setu4993/LaBSE".
                                    Default is multilingual and the SOTA.
        model_kwargs (dict, optional): model to run on cpu or gpu. Defaults to {'device': 'cpu'}.
    Returns:
        _type_: Embeddings Model
    """

    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
