import os
import inspect
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_openai import AzureChatOpenAI


from embeddings import get_self_query_retriever
from prompt import QUESTION_PROMPT, EXAMPLE_PROMPT, COMBINE_PROMPT

load_dotenv()

API_TYPE = os.getenv("V")
API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
API_VERSION = os.getenv("OPENAI_API_VERSION")
COSMOSDB_CONNECTION_STR = os.getenv("COSMOS_DB_CONNECTION_STRING")
EMBEDDINGS_DEPLOYMENT = os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT")
COMPLETIONS_DEPLOYMENT = os.getenv("OPENAI_COMPLETIONS_DEPLOYMENT")

# WIP - may be necessary to configure how the chains are used, hence it is inherited
class PowerShellRetrival(RetrievalQAWithSourcesChain):
    def _call(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun | None = None) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(inputs)  # type: ignore[call-arg]
    

def get_retrieval_chain(retriever = get_self_query_retriever()):
    llm = AzureChatOpenAI(
        api_key = API_KEY,
        api_version = API_VERSION,
        azure_endpoint = AZURE_ENDPOINT,
        azure_deployment=COMPLETIONS_DEPLOYMENT,
        streaming=True
    )
    qa_chain = RetrievalQAWithSourcesChain.from_llm(
        llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        document_prompt=EXAMPLE_PROMPT,  #format of the documents fetched from the database collection
        question_prompt =QUESTION_PROMPT, # initial query asked from the user
        combine_prompt=COMBINE_PROMPT, # prompt to retrieve answer from the LLM based on the documents fetched and question
    )

    return qa_chain

def ask_question(query: str, qa_chain = get_retrieval_chain()):
    """One time ask to an LLM, no chat history to look back onto"""
    answer = qa_chain({
        "question": query,
    })
    print(answer)
    return answer["answer"]

def ask_chat(query: str, chat_memory, qa_chain = get_retrieval_chain()):
    """Continue a chat that may exist with previous search results"""
    # TODO: test this with existing chat history
    answer = qa_chain({
        "question": query,
        "memory": chat_memory
    })
    return answer["answer"]

if __name__ == "__main__":
    question = input("Ask me a question about powershell scripts:  ")
    print(ask_question(question))
