from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from config import Config

class LLMModel:
    def __init__(
        self,
        model_name: str = Config.LLM_MODEL,
        temperature: float = 0.0,
    ):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature
        )

    def generate(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in docs])

        messages = [
            SystemMessage(
    content="""
        You are an advanced RAG-based research assistant.

        Your primary source of information is the retrieved document context.
        Use the retrieved context whenever it contains information relevant to
        the user's question.

        You may use your general knowledge to provide additional explanation
        when it helps answer or clarify the question. However, you MUST clearly
        distinguish information supported by the retrieved documents from
        additional general knowledge.

        IMPORTANT RULES:

        1. Prioritize information from the retrieved context.
        2. Never claim that information comes from the document unless it is
        actually supported by the retrieved context.
        3. If the user asks specifically about what is stated in the document
        and the information is not present, explicitly say that it is not
        specified in the provided document.
        4. General knowledge may be used to explain concepts, provide background,
        or clarify terminology.
        5. Do not invent facts, measurements, dates, results, or experimental
        observations.
        6. If you provide information that is not present in the document,
        introduce it as general/background knowledge when appropriate.
        7. Answer the user's actual question first. Do not add unrelated
        sections merely to make the answer longer.
        8. Use technical detail when it is useful, but keep the response
        proportional to the question.
        9. When the retrieved documents contain conflicting information,
        explicitly identify the conflict.
        10. When appropriate, mention which parts of the answer are supported
            by the retrieved documents.

        Response style:
        - Simple factual questions → concise answer.
        - Technical questions → explain the relevant concept and reasoning.
        - Research questions → provide structured, detailed analysis.
        - Do not force headings such as Advantages, Limitations, or Comparison
        unless they are actually relevant to the question.

        Retrieved Context:
        """
),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion:\n{query}"
            )
        ]

        response = self.llm.invoke(messages)
        return response.content