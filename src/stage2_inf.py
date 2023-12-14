import os
import random
import warnings

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, logging

from codeblip_linear import CodeBlipLinear
from codeblip_qformer import CodeQformer
from codeblip_t5 import CodeBlipT5
from dataset import CodeTranslationDataset

logging.set_verbosity_error()  # Only show errors, not warnings


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_loss_dict(loss_dict, new_loss_dict):
    for key in new_loss_dict:
        if key not in loss_dict:
            loss_dict[key] = 0
        loss_dict[key] += new_loss_dict[key].item()
    return loss_dict

if __name__ == '__main__':

    set_seed(42)

    # paths
    source_lang = 'cs'
    target_lang = 'java'

    # train_source_file, train_target_file = f'data/train.java-cs.txt.{source_lang}', f'data/train.java-cs.txt.{target_lang}'
    # valid_source_file, valid_target_file = f'data/valid.java-cs.txt.{source_lang}', f'data/valid.java-cs.txt.{target_lang}'

    # # verify paths
    # assert os.path.exists(train_source_file) and os.path.exists(train_target_file)

    # # Parameters
    # num_epochs = 10
    # batch_size = 4
    # # learning_rate = 1e-4
    # # num_training_steps = 1000  # This should be adjusted based on your dataset size
    # weight_decay = 0.01
    # # num_workers = 4

    # learning_rate = 5e-5  # Initial learning rate
    # # min_lr = 1e-5   # Minimum learning rate
    # # warmup_lr = 1e-6  # Warmup learning rate
    # # num_warmup_steps = 3

    # # Load dataset
    # train_dataset = CodeTranslationDataset(train_source_file, train_target_file, is_t5=False)
    # valid_dataset = CodeTranslationDataset(valid_source_file, valid_target_file, is_t5=False)

    # # train_dataset = train_dataset[:1000]
    # # valid_dataset = valid_dataset[:100]

    # # print dataset sizes
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Valid dataset size: {len(valid_dataset)}")


    # # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    # model = CodeQformer(
    #     num_query_token=32,
    #     cross_attention_freq=2,
    #     embed_dim=768,
    #     max_source_len=512,
    #     max_target_len=512,
    # )

    # v_name = 'codet5_ft_prompt'
    v_name = 'codet5_linear'

    # stage1_checkpoint = 'models/stage1_out/stage1_best.pt'
    # stage1_model = CodeQformer.from_config({'pretrained_path': stage1_checkpoint})

    # stage1_qformer = stage1_model.Qformer
    # stage1_query_tokens = stage1_model.query_tokens

    t5_tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained(f'Salesforce/codet5-base-codexglue-translate-{source_lang}-{target_lang}')
    # t5_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

    prompt = f'Translate {source_lang} to Python'
    # prompt = ''

    # model = CodeBlipT5(stage1_qformer, stage1_query_tokens, t5_tokenizer=t5_tokenizer, t5_model=t5_model, prompt=prompt).to('cuda' if torch.cuda.is_available() else 'cpu')

    model = CodeBlipLinear(t5_tokenizer=t5_tokenizer, t5_model=t5_model, prompt=prompt).to('cuda' if torch.cuda.is_available() else 'cpu')

    model_prefix = 'java2cs' if source_lang == 'java' else 'cs2java'

    saved_model_path = os.path.join('models', 'stage2_out', f'{model_prefix}_{v_name}_next_stage2_best.pt')

    model.load_state_dict(torch.load(saved_model_path))

    # source_code_samples = [
    #    "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } }",
        # "public class Test { public static int add(int a, int b) { return a + b; } }"]

    samples = {'java': [
        "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } }",
        "public class Test { public static int add(int a, int b) { return a + b; } }"],
               'cs': [
        "class HelloWorld { static void Main(string[] args) { Console.WriteLine(\"Hello, world!\"); } }",
        "class Test { static int Add(int a, int b) { return a + b; } }"]}

    # def sanity_check():
    #     for source_code in source_code_samples:
    #         # Prepare input for the model
    #         # samples = {"source_code": source_code_samples, "target_code": target_code_samples}
    #         samples = {"source_code": [source_code]}
    #         print(f'Current Sample: {source_code}')
    #         # Perform a forward pass
    #         output = model.generate(source_code, max_length=512)
    #         print(f"Current Output: {output}")
    #         print()

    # def sanity_check():
    #     for source_code in samples[source_lang]:
    #         # Prepare input for the model
    #         # samples = {"source_code": source_code_samples, "target_code": target_code_samples}
    #         # samples = {"source_code": [source_code]}
    #         print(f'Current Sample: {source_code}')
    #         # Perform a forward pass
    #         output = model.generate(source_code, max_length=512)
    #         print(f"Current Output: {output}")
    #         print()

    # sanity_check()

    # mode = 'combined'

    # # test_data = os.path.join('data', f'test.java-cs.txt.{source_lang}')
    # test_data = os.path.join('data', 'exps', f'exp_java2cs.txt.{source_lang}')

    # output_file =  f'exp_{model_prefix}/{mode}_exp_linear_{model_prefix}.txt'

    # for each line in test_Data do model.generate and write to output_file

    # mode_list = ['easy', 'medium', 'hard', 'has_class_or_function', 'non_class_or_function']
    # mode_list = ['non_class_or_function']

    # for mode in mode_list:
        # test_data = os.path.join('data', f'test.java-cs.txt.{source_lang}')
    # test_data = os.path.join('data', 'exps', f'{mode}.{source_lang}')
    test_data = 'test.txt'

    # output_file =  f'exp_{model_prefix}/{mode}_exp_linear_{model_prefix}.txt'
    output_file = 'python.txt'

    with open(test_data, 'r') as f:
        with open(output_file, 'w') as f_out:
            for line in tqdm(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                output = model.generate(line, num_beams=5,
                                        max_length=512,
                                        min_length=1,
                                        top_p=0.9,
                                        repetition_penalty=1.0,
                                        length_penalty=1.0,
                                        num_captions=1,
                                        temperature=0.8,)
                f_out.write(output + '\n')
                # print(f'Input: {line}')
                # print(f'Output: {output}')
            # print()

