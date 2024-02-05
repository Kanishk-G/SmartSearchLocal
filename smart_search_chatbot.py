import os
import weaviate
import asyncio
import langchain
from typing import AsyncIterable
from dotenv import load_dotenv
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.weaviate import Weaviate
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain, LLMChain, create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.loading import _load_stuff_chain
from langchain.prompts import PromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from utils import TextExtractor
#from smart_search_prompt import COMBINE_PROMPT, EXAMPLE_PROMPT
from retrieve import get_retrieval_chain


load_dotenv()
CHUNK_SIZE = 500

class SmartSearchStreaming:
    def __init__(self, directory=None) -> None:

        # auth_config = weaviate.AuthApiKey(
        #     api_key=os.getenv("WEAVIATE_API_KEY"))

        # client = weaviate.Client(
        #     url=os.getenv("WEAVIATE_URL"),
        #     auth_client_secret=auth_config,
        #     additional_headers={
        #         "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
        #     }
        # )

        # if os.getenv("OPENAI_API_KEY") is not None:
        #     print("OPENAI_API_KEY is ready")
        # else:
        #     print("OPENAI_API_KEY environment variable not found")
        #     raise ValueError

        # if not client.is_ready():
        #     raise ValueError
        # else:
        #     print("Weaviate client is ready")

        # client.schema.delete_all()

        # embeddings = OpenAIEmbeddings()

        # docs, sources = self.fetch_files_with_source(
        #     directory) if directory is not None else []

        # self.db = Weaviate.from_texts(
        #     docs,
        #     embeddings,
        #     client=client,
        #     by_text=False,
        #     metadatas=[{"source": f"{source}"} for source in sources]
        # )
        # llm = ChatOpenAI(model="gpt-4", temperature=0.3, streaming=True)

        # combine_documents_chain = _load_stuff_chain(llm, COMBINE_PROMPT, EXAMPLE_PROMPT)
        self.chain = get_retrieval_chain()
        
    async def run_call(self, query, callback):
        response = await self.chain.ainvoke({"question": query}, callbacks=[callback])
        print(response)
        return response['answer']
    
    async def query(self, query: str) -> AsyncIterable[str]:
        
        callback = AsyncIteratorCallbackHandler()
        task = asyncio.create_task(self.run_call(query, callback))

        try:
            async for token in callback.aiter():
                yield token
        except Exception as e:
            print(f'Caught exception as {e}')
        finally:
            callback.done.set()

        await task

    def query_no_stream(self, query: str) -> str:
        return self.chain.invoke({"question": query})['answer']
    
    def query_std(self, query):
        self.chain({"question": query}, callbacks=[StreamingStdOutCallbackHandler()])


    def fetch_files_with_source(self, directory):
        extractor = TextExtractor()
        files = os.listdir(directory)
        total_docs = []
        total_sources = []
        for file in files:
            filename = f'{directory}/{file}'
            source_path = Path(filename)
            documents = extractor.read_text(filename)
            text_splitter = CharacterTextSplitter(
                chunk_size=350, chunk_overlap=0, separator="\r\n")
            texts = text_splitter.split_text(documents)
            sources = ["file/" + str(source_path.name) for _ in texts]
            total_docs.extend(texts)
            total_sources.extend(sources)
        return total_docs, total_sources

    def fetch_files(self, directory):
        files = os.listdir(directory)
        extractor = TextExtractor()
        total_docs = []
        for file in files:
            documents = extractor.read_text(f'{directory}/{file}')
            text_splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=0, separator="\r\n")
            total_docs.extend(text_splitter.split_text(documents))
        return total_docs
        

class SmartSearchChatBot:
    def __init__(self, directory=None, from_llm=False, with_source=False) -> None:

        auth_config = weaviate.AuthApiKey(
            api_key=os.getenv("WEAVIATE_API_KEY"))

        client = weaviate.Client(
            url=os.getenv("WEAVIATE_URL"),
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )

        client.schema.delete_all()
        embeddings = OpenAIEmbeddings()
        if with_source:
            docs, sources = self.fetch_files_with_source(
                directory) if directory is not None else []
            self.db = Weaviate.from_texts(
                docs,
                embeddings,
                client=client,
                by_text=False,
                metadatas=[{"source": f"{source}"} for source in sources]
            )
        else:
            docs = self.fetch_files(directory) if directory is not None else []
            self.db = Weaviate.from_texts(
                docs, embeddings, client=client, by_text=False)

        self.create_conversation_chain(from_llm)

    def fetch_files_with_source(self, directory):
        extractor = TextExtractor()
        files = os.listdir(directory)
        total_docs = []
        total_sources = []
        for file in files:
            filename = f'{directory}/{file}'
            documents = extractor.read_text(filename)
            text_splitter = CharacterTextSplitter(
                chunk_size=350, chunk_overlap=0, separator="\r\n")
            texts = text_splitter.split_text(documents)
            sources = [filename for x in texts]
            total_docs.extend(texts)
            total_sources.extend(sources)
        return total_docs, total_sources

    def fetch_files(self, directory):
        files = os.listdir(directory)
        extractor = TextExtractor()
        total_docs = []
        for file in files:
            documents = extractor.read_text(f'{directory}/{file}')
            text_splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=0, separator="\r\n")
            total_docs.extend(text_splitter.split_text(documents))
        return total_docs

    def create_conversation_chain(self, from_llm=False):
        llm = ChatOpenAI()

        if from_llm:
            messages = [
                SystemMessage(
                    content="You are a powerful search assistant that can answer users questions and requests."
                )
            ]
            llm(messages)
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True)
            self.chain = ConversationalRetrievalChain.from_llm(
                llm, self.db.as_retriever(), memory=memory)
        else:
            qa_chain = create_qa_with_sources_chain(llm=llm, )
            doc_prompt = PromptTemplate(
                template="Content: {page_content}\nSource: {source}",
                input_variables=["page_content", "source"],
            )

            final_qa_chain = StuffDocumentsChain(
                llm_chain=qa_chain,
                document_variable_name="context",
                document_prompt=doc_prompt,
            )

            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True)

            _template = """You are a powerful search assistant that can answer any question or query accurately.
                        Given the following conversation and a follow up input, rephrase the follow up input to be
                        a standalone question.\
                        Make sure to avoid using any unclear pronouns.

                        Chat History:
                        {chat_history}
                        Follow Up Input: {question}
                        Standalone Question:"""
            CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

            condense_question_chain = LLMChain(
                llm=llm,
                prompt=CONDENSE_QUESTION_PROMPT,
            )

            self.chain = ConversationalRetrievalChain(
                question_generator=condense_question_chain,
                retriever=self.db.as_retriever(),
                memory=memory,
                combine_docs_chain=final_qa_chain,
            )

    def query(self, query: str):
        response = self.chain({"question": query})
        return response["answer"]


async def main():
    chatbot = SmartSearchStreaming(directory="ki_files")
    query = input("Hi! How can I help you? \n")

    while query != "!quit()":
        async for result in chatbot.query(query):
                print(result, end="")
        query = input("\n")


if __name__ == "__main__":
    asyncio.run(main())
