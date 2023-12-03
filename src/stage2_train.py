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
    source_lang = 'java'
    target_lang = 'cs'

    train_source_file, train_target_file = f'data/train.java-cs.txt.{source_lang}', f'data/train.java-cs.txt.{target_lang}'
    valid_source_file, valid_target_file = f'data/valid.java-cs.txt.{source_lang}', f'data/valid.java-cs.txt.{target_lang}'

    # verify paths
    assert os.path.exists(train_source_file) and os.path.exists(train_target_file)

    # Parameters
    num_epochs = 10
    batch_size = 4
    # learning_rate = 1e-4
    # num_training_steps = 1000  # This should be adjusted based on your dataset size
    weight_decay = 0.01
    # num_workers = 4

    learning_rate = 5e-5  # Initial learning rate
    # min_lr = 1e-5   # Minimum learning rate
    # warmup_lr = 1e-6  # Warmup learning rate
    # num_warmup_steps = 3

    # Load dataset
    train_dataset = CodeTranslationDataset(train_source_file, train_target_file)
    valid_dataset = CodeTranslationDataset(valid_source_file, valid_target_file)

    # train_dataset = train_dataset[:1000]
    # valid_dataset = valid_dataset[:100]

    # print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    # model = CodeQformer(
    #     num_query_token=32,
    #     cross_attention_freq=2,
    #     embed_dim=768,
    #     max_source_len=512,
    #     max_target_len=512,
    # )

    stage1_checkpoint = 'models/stage1_out/stage1_best.pt'
    stage1_model = CodeQformer.from_config({'pretrained_path': stage1_checkpoint})

    stage1_qformer = stage1_model.Qformer
    stage1_query_tokens = stage1_model.query_tokens

    t5_model = 'Salesforce/codet5-large'

    model = CodeBlipT5(stage1_qformer, stage1_query_tokens, t5_model=t5_model).to('cuda' if torch.cuda.is_available() else 'cpu')

    stage_2_checkpoint = 'models/stage2_out/stage2_best.pt'
    model.load_state_dict(torch.load(stage_2_checkpoint))

    # Create optimizer
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    # Calculate the number of trainable parameters
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_parameters}")

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, verbose=True)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss_list = []
    valid_loss_list = []

    best_val_loss = float('inf')

    samples = {}
    samples["source_code"] = "public void serialize(LittleEndianOutput out) {out.writeShort(field_1_vcenter);}"
    model.eval()
    print("Sample:", samples["source_code"])
    output_text = model.generate(samples)
    print("Output:", output_text)

    # Training loop with tqdm
    for epoch in range(num_epochs):
        model.train()
        # train_loss = 0
        train_loss = {}
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            train_loss_dict = model(batch)
            loss = train_loss_dict['loss']
            loss.backward()
            optimizer.step()
            # train_loss += loss.item()
            train_loss = update_loss_dict(train_loss, train_loss_dict)

        # scheduler.step()
        # train_loss /= len(train_loader)
        for key in train_loss:
            train_loss[key] /= len(train_loader)

        train_loss_list.append(train_loss['loss'])
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}")

        # Validation loop
        model.eval()
        # valid_loss = 0
        valid_loss = {}
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                valid_loss_dict = model(batch)
                loss = valid_loss_dict['loss']
                # valid_loss += loss.item()
                valid_loss = update_loss_dict(valid_loss, valid_loss_dict)

        # valid_loss /= len(valid_loader)
        for key in valid_loss:
            valid_loss[key] /= len(valid_loader)

        valid_loss_list.append(valid_loss['loss'])
        print(f"Epoch {epoch + 1}: Validation Loss: {valid_loss}")

        # Save the Qformer state after latest epoch
        torch.save(model.Qformer.state_dict(), 'models/stage2_out/qformer_stage2_latest.pt') # type: ignore
        torch.save(model.state_dict(), 'models/stage2_out/stage2_latest.pt')

        # Save the Qformer state after best validation loss
        if valid_loss['loss'] < best_val_loss:
            best_val_loss = valid_loss['loss']
            print(f'New best validation loss: {best_val_loss}')
            torch.save(model.Qformer.state_dict(), 'models/stage2_out/qformer_stage2_best.pt') # type: ignore
            torch.save(model.state_dict(), 'models/stage2_out/stage2_best.pt')

    print(f"Training completed. Best validation loss: {best_val_loss}")
    print(f"Training loss list: {train_loss_list}")
    print(f"Validation loss list: {valid_loss_list}")