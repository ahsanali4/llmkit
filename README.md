# llmkit

Repo for all work using large language models (LLMs) for applications

# List of open LLMs
https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

# Pre-reqs(poetry installation): Optional

### Install the [poetry](https://python-poetry.org/docs/#installation)

NOTE : if you don't want to use poetry you can manually install packages listed in pyproject.toml

# Installation

1. clone and cd into the repo
2. Run

    ```shell
    poetry install
    ```

    to install packages
3. Get your own [OpenAPI key](https://platform.openai.com/account/api-keys)
4. Run

    ```shell
    poetry shell
    ```

    to to activate the environment
5. Run

    ```shell
    poetry run jupyter notebook [--no-browser]
    ```

    to open juypter notebook
6. Use any of the following [examples](examples) from the example folder

# Check usage of OpenAPI credits

[OpenAPI credits Usage](https://platform.openai.com/account/usage)

# Machine Manual testing with llama2-13b-chat-german
check the examples folder for different tests performed on machine manual using llama2 [machine_manual](llm_chatbot_devs/examples/machine_manual)

# Streamlit + OpenAI agent retrieval (English)

https://colab.research.google.com/drive/12NgFnqVsit1rtb9lyNCASaMZbSg4FR_E#scrollTo=qeMBGrAYy0OQ

# Falcon40b-Instruct - Sustainability Report (English)
https://colab.research.google.com/drive/1hQZpJHK4w1N_lNNZoPqLgFZcX37whbac#scrollTo=SBNL9U6y3mx9

# Falcon40b (Raw, not tuned for chat) - Multilingual(German) - Injection Moulding Manual
https://colab.research.google.com/drive/177CVK3nuHTP8mwEDe0oy50UxUKSSpRRv#scrollTo=8ow4ttMszJpo

Inference time for above models is high (in some minutes) even on A100 GPU with 40GB vRAM (need to research about quantization of models and its effects on model inference speed)

# WizardLM - No OpenAI (under construction)
https://colab.research.google.com/drive/1JwzfVRZMEUSRhZZldDZrz5s-MyjkGn-i?usp=sharing

# WizardLM with chat history and custom prompts
https://colab.research.google.com/drive/12o313CJW69mKnGBUQL5KtrdP2Rty5LoQ?usp=sharing

# WizardLM - Injection Moulding Manual (Don't use it yet. too expensive and still under progress)
https://colab.research.google.com/drive/1bARZZnxF8lYBxtPGYFtMD7Rfc9-ckbzz?usp=sharing

# LangChain + GPT4ALL
https://colab.research.google.com/drive/1Je6znV4sbr-ys4IRQFU43yMoCXzVkyM2#scrollTo=S6n0Ml7IvjC5

# LangChain + OpenAI
https://colab.research.google.com/drive/1k7BM_bnDXqOMOx1ZFDhabWZN2qW-2Wjw#scrollTo=iJZIckCVPtg8

# LangChain Summerization
https://colab.research.google.com/drive/1-cFy1gZBJ6AehZ77WZcGyBI_TOz5L9y1#scrollTo=qBwYQOIHgp17

# LangChain Agent based interactions (not working yet)
https://colab.research.google.com/drive/1XrapNN72dgkUG2lyT7Ky7swYYaoCmNZ_#scrollTo=7qe3yyVSz9jt

# LangChain + Aleph Alpha
https://colab.research.google.com/drive/1bixmqgFXArp86hOqDa5o2cWBvPFEk3GU?usp=sharing
