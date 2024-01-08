import sys
sys.path.append("..")

import os
from typing import List, Dict, Union, Tuple, Optional
import asyncio
import re
from collections import defaultdict
import random

from langchain.evaluation.qa import QAGenerateChain
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever

from src.utils import info_str

language_map = {
    "en": "ENGLISH",
    "zh": "CHINESE",
}

async def apply_qagen_chain(
        chain: QAGenerateChain,
        documents: List[Document],
        language: str = "en",
        async_apply: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
    """apply the qa generation chain to the list of documents to generate qa pairs for each document

    Args:
        chain (QAGenerateChain): the chain to generate the qa pairs
        documents (List[Document]): the list of documents to apply to
        language (str, optional): the language of the output. Defaults to "en".
        async_apply (bool, optional): whether to apply the chain asynchronously. Defaults to True to speed up.
        verbose (bool, optional): whether to print the information. Defaults to True.

    Returns:
        qa_pairs (List[dict]): the qa pair list generated from all the documents, each of which has the format as belows:
            {
                'question': str, # the question string
                'answer': str, # the answer string
                'doc_idx': int # the index of the document, to which the qa pair relates
            }
    """
    
    # transfer the language from abbreviation to the full name
    if language in language_map: language = language_map[language]
    else: raise NotImplementedError(f"language {language} is not supported")
    
    # apply the qa generation chain to the list of documents
    if verbose: print(info_str(f"Applying the QA generation chain to {len(documents)} documents"))
    
    if not async_apply:
        outputs = chain.apply([{"doc": doc, "language": language} for doc in documents])
    else:
        outputs = await chain.aapply([{"doc": doc, "language": language} for doc in documents])
    
    # format the output into qa-pair list
    qa_order_prefix_pattern = r'^\d+\.\s' # remove the order prefix like "1. ", "2. " in the question and answer
    qa_pairs =  []
    for i, output in enumerate(outputs):
        q_list, a_list = output['qa_pairs']['questions'].split('\n'), output['qa_pairs']['answers'].split('\n')
        for q, a in zip(q_list, a_list):
            q, a = re.sub(qa_order_prefix_pattern, '', q.strip('\n')), re.sub(qa_order_prefix_pattern, '', a.strip('\n'))
            qa_pairs.append({
                'question': q,
                'answer': a,
                'doc_idx': i,
            })
    
    if verbose: print(info_str(f"There are {len(qa_pairs)} QA pairs has been generated for {len(documents)} documents"))
    
    return qa_pairs


async def apply_context_retrieve(
        retriever: VectorStoreRetriever,
        qa_pairs: List[dict],
        documents: List[Document],
        language: str = "en",
        async_apply: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
    """apply the context retrieval to each qa pair with its corresponding document

    Args:
        retriever (VectorStoreRetriever): the vector store retriever
        qa_pairs (List[dict]): the list of qa pairs with only the keys of 'question', 'answer', 'doc_idx'
        documents (List[Document]): the list of documents corresponding to the 'doc_idx' in the qa_pairs
        language (str, optional): the language of the qa pairs. Defaults to "en".
        async_apply (bool, optional): whether to apply the retrieval asynchronously. Defaults to True.
        verbose (bool, optional): whether to print the information. Defaults to True.
        
    Returns:
        qa_pairs (List[dict]): the extended qa pair list, each of which has the format as belows:
            {
                'question': str, # the question string
                'answer': str, # the answer string
                'doc_idx': int # the index of the document, to which the qa pair relates
                
                'context': str # context string made up with documents relative to original document, question and the answer
                'metadata': dict # the dict of metadata for each document in the context
            }
    """
    
    if language in language_map:
        if language == "en":
            question_prefix, answer_prefix, split_prefix = "QUESTION: ", "ANSWER: ", "<Document{doc_idx}>"
        elif language == "zh":
            question_prefix, answer_prefix, split_prefix = "问题: ", "答案: ", "<文档{doc_idx}>"
    else: raise NotImplementedError(f"language {language} is not supported")
    
    # get the input list
    #   part1: each document itself
    #   part2: the question and answer pairs
    inputs = [d.page_content for d in documents]
    inputs.extend([f"{question_prefix}{qa_pair['question']}\n{answer_prefix}{qa_pair['answer']}" for qa_pair in qa_pairs])
    
    # retrieve relative documents for each input
    if verbose: print(info_str(f"Retrieving the relative documents for {len(inputs)} queries", side_num=15))
    
    if not async_apply:
        outputs = [retriever.invoke(input) for input in inputs]
    else:
        async def retrieve_reldoc(input):
            output = await retriever.ainvoke(input)
            return output
        outputs = await asyncio.gather(*[retrieve_reldoc(input) for input in inputs])
    
    # get the long context for each qa pair, as well as the metadata
    qa_start_idx = len(documents)
    for qa_idx, qa_pair in enumerate(qa_pairs):
        doc_idx = qa_pair['doc_idx']
        context_docs = {d.page_content:d.metadata for d in outputs[doc_idx]}
        context_docs.update({d.page_content:d.metadata for d in outputs[qa_start_idx + qa_idx]})
        
        context_docs_keys = list(context_docs.keys())
        random.shuffle(context_docs_keys)
        
        context = "\n\n".join([split_prefix.format(doc_idx=idx+1) + '\n' + d for idx, d in enumerate(context_docs_keys)]) # relative to original doc, question and the answer
        metadata = {split_prefix.format(doc_idx=idx+1) : context_docs[d] for idx, d in enumerate(context_docs_keys)}
        
        qa_pair['context'] = context
        qa_pair['metadata'] = metadata
        
    if verbose: print(info_str(f"Retrieved the context and metadata for {len(qa_pairs)} QA pairs", side_num=15))
        
    return qa_pairs


def apply_instruction_gen(
        qa_pairs: List[dict],
        gen_type: str = "simple",
        style: str = "alpaca",
        language: str = "en",
        verbose: bool = True,
    ) -> Tuple[List[dict], str]:
    """apply the instruction generation from QA pairs

    Args:
        qa_pairs (List[dict]): the list of qa pairs
        gen_type (str, optional): the type of instruction generation. Defaults to "simple" to just generate the instruction from the raw QA pairs without any modification.
        style (str, optional): the prompt template style. Defaults to "alpaca".
        language (str, optional): the language of the instructions. Defaults to "en".
        verbose (bool, optional): whether to print the information. Defaults to True.

    Returns:
        instructions (List[dict]): the list of instruction dict, where each dict contains the fields for one instruction-tuning sample
        prompt_template (str): the prompt template to format any instruction dict in the instructions into one prompt string
    """
    
    # generate the instruction prompt template
    if style == "alpaca":
        if language == "en":
            prompt_template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        elif language == "zh":
            prompt_template = "给定以下一个描述某种任务的指令，以及由若干篇文档组成的输入以提供相关上下文信息，请根据上下文，恰当回答指令中的问题或请求\n\n### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:\n{output}"
    else: raise NotImplementedError(f"The prompt template style: {style} is not supported")
    
    if gen_type == "simple":
        instructions = [
            {
                'instruction': qa_pair['question'],
                'input': qa_pair['context'],
                'output': qa_pair['answer'],
                'metadata': qa_pair['metadata']
            } for qa_pair in qa_pairs
        ]
    elif gen_type == "refine":
        raise NotImplementedError(f"The instruction generation type: {gen_type} is not supported")
    else: raise NotImplementedError(f"The instruction generation type: {gen_type} is not supported")
    
    if verbose: 
        print(info_str(f"Generated {len(instructions)} long-context instructions", side_num=15))
        print(info_str(f"With the prompt template:\n{prompt_template}\n", side_num=15))
    
    return instructions, prompt_template