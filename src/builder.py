import sys
sys.path.append("..")

import os
from typing import List, Dict, Union, Tuple, Optional

from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.output_parsers.regex import RegexParser
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAGenerateChain

from src.utils import info_str


def build_retriever(documents: List[Document], 
                    vs_type: str = "chroma", 
                    emb_type: str = "openai",
                    retr_type: str = "mmr",
                    retr_kwargs: Optional[dict] = None,
                    verbose: bool = True,
                    ) -> Tuple[VectorStoreRetriever, Chroma]:
    """build the retriever to retrieve relative documents from the vetor store

    Args:
        documents (List[Document]): the list of documents
        vs_type (str, optional): vector store class type. Defaults to "chroma".
        emb_type (str, optional): embeddings class type. Defaults to "openai".
        retr_type (str, optional): retriever search algorithm class type. Defaults to "mmr".
        retr_kwargs (Optional[dict], optional): kwargs for the retriever's search algorithm. Defaults to None to use the default parameters.
        verbose (bool, optional): whether to print the information. Defaults to True.

    Returns:
        Tuple[VectorStoreRetriever, Chroma]: the vector store retriever and the corresponding vector store
    """
    # build embeddings
    if emb_type == "openai":
        embeddings = OpenAIEmbeddings()
    else: raise NotImplementedError(f"embedding type {emb_type} is not supported")
    
        
    # build vectore store
    if vs_type == "chroma":
        vectorstore = Chroma(
            collection_name="full_documents",
            embedding_function=embeddings
        )
    else: raise NotImplementedError(f"vector store type {vs_type} is not supported")
    
    vectorstore.add_documents(documents)
    
    # build retriever
    if retr_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr", # maximum marginal relevance
            search_kwargs={
                "lambda_mult": 0.25, # the higher the value, the higher the diversity and the lower the relevance
            } if retr_kwargs is None else retr_kwargs 
        ) 
    else: raise NotImplementedError(f"retriever type {retr_type} is not supported")
    
    if verbose: print(info_str(f"Built a {vs_type} retriever using the {retr_type} searching strategy, based on the {emb_type} embeddings for {len(documents)} documents", side_num=10))
    
    return retriever, vectorstore


def build_qagen_chain(
        llm_name: str = "gpt-3.5-turbo-1106", # 16k
        llm_type: str = "openai",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        verbose: bool = True,
    ) -> QAGenerateChain:
    """build a QA generation chain to generate question and answer pairs from the documents

    Args:
        llm_name (str, optional): the llm name. Defaults to "gpt-3.5-turbo-1106".
        llm_type (str, optional): the llm type. Defaults to "openai".
        temperature (float, optional): softmax temperature for generation creativity. Defaults to 0.1.
        max_tokens (Optional[int], optional): maximum number of tokens to generate. Defaults to None to generate until the model stops.
        verbose (bool, optional): whether to print the information. Defaults to True.

    Returns:
        QAGenerateChain: the QA generation chain
    """
    
    # build the prompt template
    prompt_template_str = "You are a teacher coming up with questions to ask on a quiz. \n\
Given the following document, please generate a bunch of question and answer pairs based on that document in the language of {language}.\n\n\
Example Format:\n<Begin Document>\n...\n<End Document>\n\
<QUESTIONS>:\nquestions here organized as a list of strings seperated by newlines\n<ANSWERS>:\ncorresponding answers here organized as a list of strings seperated by newlines respectively\n\n\
These questions should be detailed and be based explicitly on extractive information, synthetic knowledge, important relationship, casual reasoning, etc in the document. \n\
REMEMBER: all the generated questions and answers (excluding the title <QUESTIONS> and <ANSWERS>) should use the language of {language}, except those like terms or abbreviations that are not used in {language} in that document.\n\
Begin!\n\n<Begin Document>\n{doc}\n<End Document>"
    
    prompt_template = PromptTemplate(input_variables=["doc", "language"], template=prompt_template_str)
    
    # build the output regex parser
    output_parser = RegexParser(
        regex=r"<QUESTIONS>:\n((?:.*?\n)+)<ANSWERS>:\n((?:.*?\n)+)",
        output_keys=["questions", "answers"],
    )
    
    if verbose: 
        print(info_str(
            f"\nThe prompt template for QA generation chain is as belows:\n{prompt_template_str}\n\n" + 
            f"with input variables: {prompt_template.input_variables} | output parser: {output_parser}\n",
            side_num=50
        ))
        
    # build the llm
    if llm_type == "openai":
        llm = ChatOpenAI(
            model_name=llm_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else: raise NotImplementedError(f"llm type {llm_type} is not supported")
    
    # build the chain
    chain = QAGenerateChain(
        llm=llm,
        prompt=prompt_template,
        output_parser=output_parser,
    )
    
    if verbose:
        print(info_str(f"Built a QA generation chain based on the {llm_type} LLM: {llm_name}", side_num=15))
        
    return chain