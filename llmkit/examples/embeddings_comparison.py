import streamlit as st

from llm_chatbot_devs.core.embeddings import get_hf_embeddings, get_instruct_embeddings

# from llm_chatbot_devs.utils.preprocessing import create_chroma_db

models = {
    "hkunlp/instructor-xl": get_instruct_embeddings(model_kwargs={"device": "cuda"}),
    "hkunlp/instructor-large": get_instruct_embeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"}
    ),
    "setu4993/LaBSE": get_hf_embeddings(model_kwargs={"device": "cuda"}),
    "setu4993/LaBSE_instruct": get_instruct_embeddings(
        model_name="setu4993/LaBSE", model_kwargs={"device": "cuda"}
    ),
}

options = models.keys()


# def load_embedder(embedding_modelA_name: str, embedding_modelB_name: str):
#     modelA = models.get(embedding_modelA_name)
#     modelB = models.get(embedding_modelB_name)
#     create_chroma_db


left, right = st.columns(2)


# # Define options for both select boxes
# options = ["Option 1", "Option 2", "Option 3", "Option 4"]

# Create a Streamlit app
st.title("Mutually Exclusive Select Boxes Example")

# First select box
selected_option_1 = left.selectbox("Select Option 1:", options)

# Second select box
# Remove the option selected in the first select box from the options for the second select box
remaining_options = [opt for opt in options if opt != selected_option_1]
selected_option_2 = right.selectbox("Select Option 2:", remaining_options)

# Display the selections
st.write("Selected Option 1:", selected_option_1)
st.write("Selected Option 2:", selected_option_2)

user_input = st.text_input("Enter some text:")
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()


# Callback function to process the user input and display the modified text
def process_text(input_text):
    if input_text:
        return f"You entered: {input_text.upper()}\
              with model {selected_option_1} and {selected_option_2}"
    else:
        return "Please enter some text."


# Button to trigger the callback function
if user_input:
    result = process_text(user_input)
    st.write(result)
