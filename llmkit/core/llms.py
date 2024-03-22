import os
from typing import List

import torch
import transformers
from langchain.chat_models import ChatOpenAI
from langchain.llms import AlephAlpha, HuggingFacePipeline
from torch import bfloat16, cuda
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
    pipeline,
)

from llm_chatbot_devs.utils.config import Config

config = Config()


# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __init__(self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


def get_falcon40b(chat_model: bool = False, use_stopping_criteria: bool = True):
    """Get Falcon model especially 40b with chat(instruct-english)
      and without chat (multilingual). Should be used with stopping tokens

    Args:
        chat_model (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if chat_model:
        model_name: str = "tiiuae/falcon-40b-instruct"
    else:
        model_name: str = "tiiuae/falcon-40b"

    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    if use_stopping_criteria:
        stop_tokens = [["Human", ":"], ["AI", ":"]]
        stopping_criteria = StoppingCriteriaList(
            [StopOnTokens(stop_tokens, tokenizer, model.device)]
        )
    return model, tokenizer, stopping_criteria


def get_aleph_alpha_llm(params: dict) -> AlephAlpha:

    if not params:
        params = {"temperature": 0.5, "model": "luminous-extended", "maximum_tokens": 100}
    # define the model
    aleph_alpha = AlephAlpha(aleph_alpha_api_key=os.getenv("AA_TOKEN"), **params)
    return aleph_alpha


def create_wizard_lm_model_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")

    model = LlamaForCausalLM.from_pretrained(
        "TheBloke/wizardLM-7B-HF",
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def create_hf_pipeline(
    model,
    tokenizer,
    stopping_criteria,
    max_length: int = 1024,
    temperature: float = 0.0,
    streaming: bool = False,
    model_kwargs: dict = {"temperature": 0.0},
):
    if streaming:
        streamer = TextStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            return_full_text=True,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            return_full_text=True,
            stopping_criteria=stopping_criteria,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.95,
            repetition_penalty=1.15,
        )

    return HuggingFacePipeline(pipeline=pipe, model_kwargs=model_kwargs)


def get_chatopenai_llm(
    num_outputs: int = 512,
    temperature: float = 0.7,
    model_name: str = "gpt-3.5-turbo",
    streaming: bool = False,
):

    return ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        max_tokens=num_outputs,
        openai_api_key=config.openai_api_key.get_secret_value(),
        streaming=streaming,
    )


def get_llama2_german(model_name: str = None, quantization: bool = True):

    if not model_name:
        model_name = (
            "/content/drive/MyDrive/Colab Notebooks/LLMS/Llama-2-13b-chat-german"  # stored on colab
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantization:

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, quantization_config=bnb_config, device_map="auto"
        )
        return model, tokenizer
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="auto"
        )
        return model, tokenizer
