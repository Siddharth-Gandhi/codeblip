import os
import random
import warnings

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, logging

from codeblip_qformer import CodeQformer
from dataset import CodeTranslationDataset
from train import update_loss_dict

logging.set_verbosity_error()  # Only show errors, not warnings


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# get validation loss
def get_loss(model, loader):
    model.eval()
    total_loss_dict = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            loss_dict = model(batch)
            total_loss_dict = update_loss_dict(total_loss_dict, loss_dict)
    for key in total_loss_dict:
        total_loss_dict[key] /= len(loader)
    return total_loss_dict

if __name__ == '__main__':

    set_seed(42)

    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # paths
    source_lang = 'java'
    target_lang = 'cs'

    best_model_path = os.path.join('models', 'stage1_out' ,'stage1_best.pt')

    valid_source_file, valid_target_file = f'data/valid.java-cs.txt.{source_lang}', f'data/valid.java-cs.txt.{target_lang}'
    test_source_file, test_target_file = f'data/test.java-cs.txt.{source_lang}', f'data/test.java-cs.txt.{target_lang}'

    # verify paths
    assert os.path.exists(valid_source_file) and os.path.exists(valid_target_file)

    # Load dataset
    valid_dataset = CodeTranslationDataset(valid_source_file, valid_target_file)
    test_dataset = CodeTranslationDataset(test_source_file, test_target_file)

    # print dataset sizes
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    cfg = {
        'num_query_token': 32,
        'cross_attention_freq': 2,
        'embed_dim': 768,
        'max_source_len': 512,
        'max_target_len': 512,
        'pretrained_path': best_model_path,
    }

    model = CodeQformer.from_config(cfg)
    model.to(device)
    model.eval()
    print(f'Loaded model from {best_model_path}')



    valid_loss_dict = get_loss(model, valid_loader)
    for key in valid_loss_dict:
        print(f"{key}: {valid_loss_dict[key]}")
