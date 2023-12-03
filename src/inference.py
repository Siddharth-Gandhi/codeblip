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

logging.set_verbosity_error()  # Only show errors, not warnings


if __name__ == "__main__":
    stage2_checkpoint = "models/stage2_out/stage2_best.pt"
    model = CodeBlipT5.from_config({'pretrained_path': stage2_checkpoint})
    model.eval()

    source_code_samples = [
        "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } }",
        "public class Test { public static int add(int a, int b) { return a + b; } }"]

