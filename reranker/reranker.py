from flashrank import Ranker

from langchain_classic.retrievers.document_compressors import (
    FlashrankRerank,
    CohereRerank
)
from langchain_classic.retrievers import ContextualCompressionRetriever

# IMPORTANT
FlashrankRerank.model_rebuild(_types_namespace={"Ranker": Ranker})

from dotenv import load_dotenv
import os

load_dotenv()

cohere_api = os.getenv("Cohere_Key")


class Reranker:
    def __init__(self, reranker_type="flashrank"):
        if reranker_type == "flashrank":
            self.compressor = FlashrankRerank()

        elif reranker_type == "cohere":
            self.compressor = CohereRerank(
                cohere_api_key=cohere_api
            )

    def get_compression_retriever(self, base_retriever):
        return ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )