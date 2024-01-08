import os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import contextlib
import json

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from langchain.prompts import PromptTemplate


def save_instructions(
        instructions: List[dict],
        prompt_template: str,
        save_dir: str,
    ):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # save the instructions jsonl file
    jsonl_str = "\n".join([json.dumps(instr_sample, ensure_ascii=False, indent=4) for instr_sample in instructions])
    with open(os.path.join(save_dir, "instructions.jsonl"), "w", encoding="utf-8") as f:
        f.write(jsonl_str)
    
    # save the prompt template
    prompt_template = PromptTemplate.from_template(prompt_template)
    with open(os.path.join(save_dir, "prompt_template.json"), "w", encoding="utf-8") as f:
        json.dump(
            prompt_template.to_json()['kwargs'], 
            f, ensure_ascii=False, indent=4
        )
    
    print(info_str(f"Saved the instructions and prompt template to {save_dir}"))


def stats_instructions(
        instructions: List[dict],
        prompt_template: str,
    ):
    instr_prompts = [prompt_template.format(**instr_sample) for instr_sample in instructions]
    lens = [len(instr_prompt) for instr_prompt in instr_prompts]
    
    print(info_str("Some statistics of the instruction samples", side_num=25))
    print(f"sample number: {len(instr_prompts)}\nsample length: average: {np.mean(lens):.0f} | min: {np.min(lens)} | max: {np.max(lens)}\n")
    
    print(info_str("Histogram of the instruction sample length", side_num=25))
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    sns.histplot(lens, ax=ax)
    plt.show()


def print_instructions(
        instructions: List[dict],
        prompt_template: str,
        sample_size: Optional[int] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        prompt_limit: Optional[int] = 1000,
    ):
    instructions = instructions[start_idx:(end_idx if end_idx is not None else len(instructions))]
    if sample_size is not None and sample_size < len(instructions):
        instructions = random.sample(instructions, sample_size)
        
    for instr_idx, instr_sample in enumerate(instructions):
        print(info_str(f"Instruction Sample {instr_idx+1}", side_num=25))
        instr_prompt = prompt_template.format(**instr_sample)
        print(f"{instr_prompt[:prompt_limit//2]}\n...\n{instr_prompt[-prompt_limit//2:]}\n")
    

def print_qa_pairs(
        qa_pairs: List[dict],
        sample_size: Optional[int] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        context_limit: Optional[int] = 1000,
    ):
    qa_pairs = qa_pairs[start_idx:(end_idx if end_idx is not None else len(qa_pairs))]
    if sample_size is not None and sample_size < len(qa_pairs):
        qa_pairs = random.sample(qa_pairs, sample_size)
    
    if 'context' in qa_pairs[0]:
        for qa_idx, qa_pair in enumerate(qa_pairs):
            print(info_str(f"QA pair {qa_idx+1}", side_num=25))
            print(info_str(f"context:\n{qa_pair['context'][:context_limit//2]}\n...\n{qa_pair['context'][-context_limit//2:]}\n", side_num=25, side_str='-'))
            print(f"question: {qa_pair['question']}")
            print(f"answer: {qa_pair['answer']}")
    else:
        doc2qa_pairs = defaultdict(list)
        for qa_pair in qa_pairs:
            doc2qa_pairs[qa_pair['doc_idx']].append(qa_pair)
            
        doc_idxs = sorted(doc2qa_pairs.keys())
        for doc_idx in doc_idxs:
            print(info_str(f"For document {doc_idx+1}", side_num=25))
            for i, qa_pair in enumerate(doc2qa_pairs[doc_idx]):
                print("-"*30)
                print(f"question{i+1}: ", qa_pair['question'])
                print(f"answer{i+1}: ", qa_pair['answer'])


def info_str(center_content: str = "", 
            side_str: str = "=", 
            side_num: int = 25) -> str:
    return "\n" + \
        side_str * side_num + " " + \
        center_content + " " + \
        side_str * side_num + \
        "\n"

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as null:
        with contextlib.redirect_stderr(null):
            yield