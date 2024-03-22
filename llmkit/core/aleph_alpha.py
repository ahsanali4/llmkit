import os
from typing import List

from langchain.embeddings import (
    AlephAlphaAsymmetricSemanticEmbedding,
    AlephAlphaSymmetricSemanticEmbedding,
)
from langchain.llms import AlephAlpha


class AlephAlphaLLM:
    def __init__(self, params: dict) -> None:

        if not params:
            self.params = {"temperature": 0.5, "model": "luminous-extended", "maximum_tokens": 100}
        else:
            self.params = params
        # define the model
        self.aleph_alpha = AlephAlpha(aleph_alpha_api_key=os.getenv("AA_TOKEN"), **self.params)

    def get_llm(self):
        return self.aleph_alpha

    def prompting(self, prompt, stop: List = ["\n"]):
        return self.aleph_alpha(prompt=prompt, stop=stop)

    def get_symmetric_semantic_embeddings(
        self, compress_to_size: int = 128
    ) -> AlephAlphaSymmetricSemanticEmbedding:
        return AlephAlphaSymmetricSemanticEmbedding(
            client=self.aleph_alpha, compress_to_size=compress_to_size
        )

    def get_asymmetric_semantic_embeddings(
        self, compress_to_size: int = 128
    ) -> AlephAlphaAsymmetricSemanticEmbedding:
        return AlephAlphaAsymmetricSemanticEmbedding(
            client=self.aleph_alpha, compress_to_size=compress_to_size
        )
