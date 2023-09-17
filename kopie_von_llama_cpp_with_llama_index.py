!pip install langchain llama-index sentence-transformers
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.78 numpy==1.23.4 --force-reinstall --upgrade --no-cache-dir --verbose

!cd data
!git clone https://github.com/Darthph0enix7/Swagger_StockInterpretation

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt

model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 64},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

from llama_index.llms.llama_utils import messages_to_prompt
from llama_index.prompts.base import ChatPromptTemplate
from llama_index.llms.base import ChatMessage, MessageRole
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import LlamaCPP

# text qa prompt
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# Tree Summarize
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)


# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "You are an expert Q&A system that stricly operates in two modes"
            "when refining existing answers:\n"
            "1. **Rewrite** an original answer using the new context.\n"
            "2. **Repeat** the original answer if the new context isn't useful.\n"
            "Never reference the original answer or context directly in your answer.\n"
            "When in doubt, just repeat the original answer."
            "New Context: {context_msg}\n"
            "Query: {query_str}\n"
            "Original Answer: {existing_answer}\n"
            "New Answer: "
        ),
        role=MessageRole.USER,
    )
]


CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)


# Table Context Refine Prompt
CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
    ChatMessage(content="{query_str}", role=MessageRole.USER),
    ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
    ChatMessage(
        content=(
            "We have provided a table schema below. "
            "---------------------\n"
            "{schema}\n"
            "---------------------\n"
            "We have also provided some context information below. "
            "{context_msg}\n"
            "---------------------\n"
            "Given the context information and the table schema, "
            "refine the original answer to better "
            "answer the question. "
            "If the context isn't useful, return the original answer."
        ),
        role=MessageRole.USER,
    ),
]
CHAT_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(
    message_templates=CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS
)

# [Prompt Template Definitions go here]

# Initialize the LlamaCPP and other services
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# Load documents
documents = SimpleDirectoryReader("/content").load_data()

# Combine the content of all the documents as context
context_data = documents

query_data = "When did the born?"

# Fill in the CHAT_TEXT_QA_PROMPT with the context and query
filled_prompt_msg = CHAT_TEXT_QA_PROMPT.message_templates[1].content.format(
    context_str=context_data, query_str=query_data
)

# Convert the filled prompt message to a ChatMessage object before passing it to messages_to_prompt
filled_prompt_message_obj = ChatMessage(content=filled_prompt_msg, role=MessageRole.USER)
filled_prompt = messages_to_prompt([filled_prompt_message_obj])


# Create vector store index
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# Set up query engine
query_engine = index.as_query_engine()

print(documents)
print(query_engine)
print(index)
print(embed_model)

# Pass the filled-in prompt to the LlamaCPP instance
response = llm.complete(filled_prompt)
print(response.text)

follow_up_question = "An where?"

# Combine original filled prompt, response, and follow-up question
combined_prompt_message = ChatMessage(
    content=filled_prompt_msg + "\n" + response.text + "\n" + follow_up_question,
    role=MessageRole.USER
)
combined_prompt = messages_to_prompt([combined_prompt_message])

# Get response to the follow-up question
new_response = llm.complete(combined_prompt)
print(new_response.text)

# Setting up prompt templates
from llama_index.llms.llama_utils import messages_to_prompt
from llama_index.prompts.base import ChatPromptTemplate
from llama_index.llms.base import ChatMessage, MessageRole
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import LlamaCPP
import requests

# text qa prompt
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# Tree Summarize
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)


# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "You are an expert Q&A system that stricly operates in two modes"
            "when refining existing answers:\n"
            "1. **Rewrite** an original answer using the new context.\n"
            "2. **Repeat** the original answer if the new context isn't useful.\n"
            "Never reference the original answer or context directly in your answer.\n"
            "When in doubt, just repeat the original answer."
            "New Context: {context_msg}\n"
            "Query: {query_str}\n"
            "Original Answer: {existing_answer}\n"
            "New Answer: "
        ),
        role=MessageRole.USER,
    )
]


CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)


# Table Context Refine Prompt
CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
    ChatMessage(content="{query_str}", role=MessageRole.USER),
    ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
    ChatMessage(
        content=(
            "We have provided a table schema below. "
            "---------------------\n"
            "{schema}\n"
            "---------------------\n"
            "We have also provided some context information below. "
            "{context_msg}\n"
            "---------------------\n"
            "Given the context information and the table schema, "
            "refine the original answer to better "
            "answer the question. "
            "If the context isn't useful, return the original answer."
        ),
        role=MessageRole.USER,
    ),
]
CHAT_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(
    message_templates=CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS
)
# Initialize the LlamaCPP and other services
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# Function to fetch stock price from Alpha Vantage
def fetch_stock_price(symbol):
    API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your actual API key
    endpoint = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={API_KEY}"
    response = requests.get(endpoint)
    data = response.json()

    # Extract the latest stock price
    latest_time = max(data['Time Series (5min)'])
    stock_price = data['Time Series (5min)'][latest_time]['1. open']

    return stock_price

# Modified process_query function to handle stock-related questions
def process_query(query):
    if "stock price" in query.lower() and "amazon" in query.lower():
        stock_price = fetch_stock_price("AMZN")

        # Create the context data
        context_data = f"The current stock price of Amazon is ${stock_price}."

        # Fill in the CHAT_TEXT_QA_PROMPT with the context and query
        filled_prompt_msg = CHAT_TEXT_QA_PROMPT.message_templates[1].content.format(
        context_str=context_data, query_str=query
        )

        # Convert the filled prompt message to a ChatMessage object before passing it to messages_to_prompt
        filled_prompt_message_obj = ChatMessage(content=filled_prompt_msg, role=MessageRole.USER)
        filled_prompt = messages_to_prompt([filled_prompt_message_obj])

        # Send the filled-in prompt to the LlamaCPP instance
        response = llm.complete(filled_prompt)

        return response.text

    # Handle other types of questions as previously
    else:
        documents = SimpleDirectoryReader("/content").load_data()

        # Combine the content of all the documents as context
        context_data = documents

        # Fill in the CHAT_TEXT_QA_PROMPT with the context and query
        filled_prompt_msg = CHAT_TEXT_QA_PROMPT.message_templates[1].content.format(
            context_str=context_data, query_str=query_data
        )

        # Convert the filled prompt message to a ChatMessage object before passing it to messages_to_prompt
        filled_prompt_message_obj = ChatMessage(content=filled_prompt_msg, role=MessageRole.USER)
        filled_prompt = messages_to_prompt([filled_prompt_message_obj])


        # Create vector store index
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        # Set up query engine
        query_engine = index.as_query_engine()

        print(documents)
        print(query_engine)
        print(index)
        print(embed_model)

        # Pass the filled-in prompt to the LlamaCPP instance
        response = llm.complete(filled_prompt)
        print(response.text)

        return "I'm not sure how to answer that."

# Sample usage:
query = "What is the stock price of Amazon today and how does it affect the market?"
response = process_query(query)
print(response)

