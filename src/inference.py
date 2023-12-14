import os
import random
import warnings

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging

from codeblip_qformer import CodeQformer
from codeblip_t5 import CodeBlipT5
from dataset import CodeTranslationDataset
from codeblip_llama import Blip2Llama

logging.set_verbosity_error()  # Only show errors, not warnings


if __name__ == "__main__":
    stage1_checkpoint = 'models/stage1_out/stage1_best.pt'
    stage1_model = CodeQformer.from_config({'pretrained_path': stage1_checkpoint})

    stage1_qformer = stage1_model.Qformer
    stage1_query_tokens = stage1_model.query_tokens

    # t5_model = 'Salesforce/codet5-large'

    # model = CodeBlipT5(stage1_qformer, stage1_query_tokens, t5_model=t5_model).to('cuda' if torch.cuda.is_available() else 'cpu')

    model = Blip2Llama(stage1_qformer, stage1_query_tokens).to('cuda' if torch.cuda.is_available() else 'cpu')

    stage_2_checkpoint = 'models/stage2_out/llama_stage2_best.pt'
    model.load_state_dict(torch.load(stage_2_checkpoint))
    model.eval()

    source_code_samples = [
        "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } }",
        "public class Test { public static int add(int a, int b) { return a + b; } }"
    ]

    prompt = "Translate the given Java code to Python: "


# prompt = """You are an expert at translating code from one language to another.
# Translate the provided code in Java to Python. """ + """Use the following format for translation:
# Java: public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } } 
# Python: 
# class HelloWorld:
#     def main(args):
#         print("Hello, world!")

# HelloWorld.main([])
    
# Java: public class Test { public static int add(int a, int b) { return a + b; } }
# Python: 
# class Test:
#     @staticmethod
#     def add(a, b):
#         return a + b

# Java: public String toString() {return getClass().getName() + " [" +_value +"]";}
# Python: 
# def __str__(self):
#     return type(self).__name__ + " [" + str(self._value) + "]"


# Translate the given Java code to Python: 
# """ + f"Java: {source_code} \n Python: "



    print("Source code sample:", source_code_samples[1])
    print(model.generate({'source_code': source_code_samples[1]}, prompt = prompt))

    print("Source code sample:", source_code_samples[0])
    print(model.generate({'source_code': source_code_samples[0]}, prompt = prompt))

