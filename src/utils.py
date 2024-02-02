import os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
import contextlib
import json
import logging
import logging.handlers
import warnings
from collections import namedtuple

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_instructions(
        instructions: List[dict],
        prompt_template: str,
        save_dir: str,
    ):
    from langchain.prompts import PromptTemplate
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


def info_dict(d, t=1, precision=2) -> str:
    s = "{\n"
    for k, v in d.items():
        s += "\t"*t + str(k)
        s += " : "
        if isinstance(v, dict):
            vd = info_dict(v, t+1)
            s += vd
        else:
            if isinstance(v, float):
                if len(str(v)) > len("0.001"): s += f"{v:.{precision}e}"
                else: s += str(v)
            else: s += str(v)
                    
        s += "\n"
    s +=  "\t"*(t-1) + "}"

    return s

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as null:
        with contextlib.redirect_stderr(null):
            yield
            
            
def stats_dataset_on_length(datasets: List[Any], 
                            keys: Optional[List[str]] = None,
                            hist: bool = False,
                            save_path: Optional[str] = None,
                            logger: Optional[logging.Logger] = None,
                            ) -> None:
    from datasets import Dataset
    sample_to_len_func = lambda sample: sum(len(sample[key]) for key in keys) if keys is not None else len(sample)
    log_func = print if logger is None else lambda x: logger.info(x)
    
    lens = []
    for dataset in datasets:
        lens.extend([sample_to_len_func(sample) for sample in dataset])

    stats_str = f"sample number: {len(lens):,} | mean length: {np.mean(lens):,.1f} | min length: {np.min(lens):,} | max length: {np.max(lens):,}"
    
    log_func(info_str("Statistics on dataset length", side_num=25))
    log_func(info_str(stats_str, side_num=25))
    
    if hist:
        log_func(info_str("Histogram of the dataset's length distribution", side_num=25))
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        sns.histplot(lens, ax=ax)
        if save_path is not None: 
            plt.savefig(save_path)
            log_func(info_str(f"Save the histogram to {save_path}", side_num=25))
           
def print_trainable_params(model, logger: Optional[logging.Logger] = None):
    log_func = print if logger is None else lambda x: logger.info(x)
    
    trainable_params, total_params = 0, 0
    for _, param in model.named_parameters():
        if param.requires_grad: trainable_params += param.numel()
        total_params += param.numel()
        
    log_func(info_str(
        f"Trainable params: {trainable_params:,} | Total params: {total_params:,} | Trainable rate: {100 * trainable_params / total_params:.2f}"
        , side_num=25
    ))
    
def set_avaiable_gpus(gpus_list: Optional[str] = None):
    if gpus_list is not None: 
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus_list)
                 
class MultiTargetLogger(object):
    class MultipleTargetHandler(logging.Handler):
        def __init__(self, handlers):
            super().__init__()
            self.handlers = handlers
        def emit(self, record):
            for handler in self.handlers:
                handler.emit(record)
    
    def __init__(self,
            log_path: str,
            mode: str = 'w',
        ):
        self.log_path = log_path
        open(self.log_path, mode=mode)
        # init the logger
        self.logger = logging.getLogger('logger')
        formatter = logging.Formatter('%(message)s')
        self.logger.setLevel(logging.INFO)
        # set the multiple target handler
        file_handler = logging.FileHandler(self.log_path, mode='a', delay=False)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        multi_handler = MultiTargetLogger.MultipleTargetHandler([file_handler, stream_handler])
        memory_handler = logging.handlers.MemoryHandler(capacity=100, target=multi_handler)
        
        self.logger.addHandler(memory_handler)
    def info(self, text: str):
        self.logger.info(text)
    def print(self):
        for handler in self.logger.handlers:
            handler.flush()
     
def get_logger(log_path: str, mode='w'):
    return MultiTargetLogger(log_path=log_path, mode=mode)

def get_dtype(dtype: str = "float32"):
    from torch import float32, float16, bfloat16
    
    return {
        "float32": float32,
        "float16": float16, 
        "bfloat16": bfloat16,
    }[dtype]
    
def get_quant_config(quant_type: str = None, quant_bit: int = 4, dtype = None, **kwargs):
    
    if quant_type != None:
        if quant_type == "awq": 
            from transformers import AwqConfig
            return AwqConfig(
                bits=quant_bit,
            )
        elif quant_type == "gptq": 
            from transformers import GPTQConfig
            return GPTQConfig(
                bits=quant_bit,
            )
        elif quant_type == "bnb":
            from transformers import BitsAndBytesConfig
            if quant_bit == 8:
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=kwargs.get('int8_threshold', 6.0),
                    llm_int8_enable_fp32_cpu_offload=kwargs.get('int8_fp32_offload', False),
                )
            elif quant_bit == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type=kwargs.get('4bit_quant_type', 'nf4'),
                    bnb_4bit_use_double_quant=kwargs.get('4bit_double_quant', True),
                )
            else: raise ValueError(f"quant_bit should be 4 or 8, but got {quant_bit} for type {quant_type}")

def get_target_modules(
        target_type: Union[List[str], str] = "linear",
        model_type: str = "llama2", 
    ) -> List[str]:
    if isinstance(target_type, list): return target_type
    
    ## initialize the basic target modules for each model type
    llama2_qkv_weights = ["q_proj", "k_proj", "v_proj"]
    llama2_attn_weights = llama2_qkv_weights + ["out_proj"]
    llama2_ffn_weights = ["up_proj", "down_proj"]
    llama2_linear_weights = llama2_attn_weights + llama2_ffn_weights + ["lm_head"]
    
    chatglm_qkv_weights = ["query_key_value"]
    chatglm_attn_weights = chatglm_qkv_weights + ["dense"]
    chatglm_ffn_weights = ["dense_h_to_4h", "dense_4h_to_h"]
    chatglm_linear_weights = chatglm_attn_weights + chatglm_ffn_weights + ["output_layer"]
    
    mistral_qkv_weights = llama2_qkv_weights
    mistral_attn_weights = mistral_qkv_weights + ["o_proj"]
    mistral_ffn_weights = llama2_ffn_weights
    mistral_linear_weights = mistral_attn_weights + mistral_ffn_weights + ["lm_head"]
    
    mixtral_moe_weights = ["w1", "w2", "w3"]
    
    ## set up the target modules mapping for each model type and each target type
    target_modules_map = {
        "llama2": {
            "linear": llama2_linear_weights,
            "decoder": llama2_attn_weights + llama2_ffn_weights,
            "attn": llama2_attn_weights,
            "ffn": llama2_ffn_weights,
            "qkv" : llama2_qkv_weights,
        },
        "chatglm2": {
            "linear": chatglm_linear_weights,
            "decoder": chatglm_attn_weights + chatglm_ffn_weights,
            "attn": chatglm_attn_weights,
            "ffn": chatglm_ffn_weights,
            "qkv": chatglm_qkv_weights,
        },
        "chatglm3": {
            "linear": chatglm_linear_weights,
            "decoder": chatglm_attn_weights + chatglm_ffn_weights,
            "attn": chatglm_attn_weights,
            "ffn": chatglm_ffn_weights,
            "qkv": chatglm_qkv_weights,
        },
        "mistral": {
            "linear": mistral_linear_weights,
            "decoder": mistral_attn_weights + mistral_ffn_weights,
            "attn": mistral_attn_weights,
            "ffn": mistral_ffn_weights,
            "qkv" : mistral_qkv_weights,
        },
        "mixtral": {
            "linear": mistral_linear_weights + mixtral_moe_weights,
            "decoder": mistral_attn_weights + mistral_ffn_weights + mixtral_moe_weights,
            "attn": mistral_attn_weights,
            "ffn": mistral_ffn_weights,
            "qkv" : mistral_qkv_weights,
            "moe": mixtral_moe_weights,
        }
    }
    
    supported_models = list(target_modules_map.keys())
    if model_type not in supported_models:
        raise ValueError(f"model_type should be in {supported_models}, but got {model_type}")
    supported_targets = list(target_modules_map[model_type].keys())
    if target_type not in supported_targets:
        raise ValueError(f"target should be in {supported_targets} for model {model_type}, but got {target_type}")
    
    return target_modules_map[model_type][target_type]
    
def get_optimizer(optim_type: str = "adamw", 
                  quant_type: Optional[str] = None,
                  quant_bit: Optional[int] = 4,
                  ):
    if optim_type == "adamw":
        if quant_type is None: return "adamw_torch"
        else: return "paged_adamw_8bit"
    elif optim_type == "paged_adamw":
        if quant_type is None: return "paged_adamw_32bit"
        else: return f"paged_adamw_8bit"
    else: raise NotImplementedError(f"optim_type {optim_type} is not supported")
    
def get_grad_ckpt(grad_ckpt_str: str = "true"):
    if grad_ckpt_str.lower() == "true": return True
    elif grad_ckpt_str.lower() == "false": return False
    else: raise ValueError(f"grad_ckpt should be either true or false in the lower case, but got {grad_ckpt_str}")
     
def get_mix_precision_args(mix_precision_str: Optional[str] = None) -> dict:
    if mix_precision_str is None: return {}
    elif mix_precision_str == "bf16": return {"bf16": True}
    elif mix_precision_str == "fp16": return {"fp16": False}
    else: raise ValueError(f"mix_precision should be either bf16 or fp16, but got {mix_precision_str}")
    
def load_jsonl_dataset(path: str) -> Any:
    from datasets import Dataset
    
    df = pd.read_json(path, lines=True)
    dataset = Dataset.from_pandas(df)
    return dataset

    # old version: FIXME: sometimes raise ArrowInvalid error
    # jsonl_dataset = []
    # with open(path, 'r', encoding='utf-8') as f:
    #     for line in f: jsonl_dataset.append(json.loads(line))
    # dataset = Dataset.from_list(jsonl_dataset)
        
    # return dataset

def save_args(args, path: str):
    arg_dict = vars(args)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(arg_dict, f, ensure_ascii=False, indent=4)

def get_prompt_generator(path: Optional[str] = None, 
                         name: Optional[str] = None) -> Callable[[dict], Tuple[List[str], List[int]]]:
    import importlib.util
    def default_prompt_generator(sample: dict):
        prompt = sample['prompt_template'].format(**sample)
        label = [1]
        return [prompt], label
    
    if path is None: 
        return default_prompt_generator
    
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("prompt_generator", path)    
        prompt_generator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt_generator_module)
        
        if name is None: name = "generate_prompt"
        
        if not hasattr(prompt_generator_module, name): 
            warnings.warn(f"Cannot find the prompt generator {name} in {path}, use default one instead.")
            return default_prompt_generator
        
        prompt_generator = getattr(prompt_generator_module, name)
        
        return prompt_generator
    
def get_truncator(strategy: str = "middle", max_length: int = 4096) -> Callable[[list], list]:
    if strategy == "middle":
        def truncate_middle(tokens: list) -> list:
            if len(tokens) <= max_length: return tokens
            return tokens[:max_length//2] + tokens[-max_length//2:]
        return truncate_middle
    elif strategy == "head":
        def truncate_head(tokens: list) -> list:
            if len(tokens) <= max_length: return tokens
            return tokens[:max_length]
        return truncate_head
    elif strategy == "tail":
        def truncate_tail(tokens: list) -> list:
            if len(tokens) <= max_length: return tokens
            return tokens[-max_length:]
        return truncate_tail
    else: raise NotImplementedError(f"Not supported truncation strategy: {strategy}")
    