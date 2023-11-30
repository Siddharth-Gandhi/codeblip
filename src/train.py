import torch
import os
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from codeblip_qformer import CodeQformer
from dataset import CodeTranslationDataset
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# paths
source_lang = 'java'
target_lang = 'cs'

train_source_file, train_target_file = f'data/train.java-cs.txt.{source_lang}', f'data/train.java-cs.txt.{target_lang}'
valid_source_file, valid_target_file = f'data/valid.java-cs.txt.{source_lang}', f'data/valid.java-cs.txt.{target_lang}'

# verify paths
assert os.path.exists(train_source_file) and os.path.exists(train_target_file)

# Parameters
num_epochs = 5
batch_size = 16
learning_rate = 5e-5
num_warmup_steps = 0
num_training_steps = 1000  # This should be adjusted based on your dataset size

# Load dataset
train_dataset = CodeTranslationDataset(train_source_file, train_target_file)
valid_dataset = CodeTranslationDataset(valid_source_file, valid_target_file)

# print dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Create model
model = CodeQformer.from_config({
    "num_query_token": 32,
    "cross_attention_freq": 2,
    "embed_dim": 768,
    "max_source_len": 512,
    "max_target_len": 512,
})

# Create optimizer
# optimizer = AdamW(model.parameters(), lr=learning_rate)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
# Calculate the number of trainable parameters
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_parameters}")

# Create learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train model
# Training loop with tqdm
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        loss = model(batch)['loss']
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}: Train Loss: {train_loss}")

    # Validation loop
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating"):
            loss = model(batch)
            valid_loss += loss.item()

    valid_loss /= len(valid_loader)
    print(f"Epoch {epoch + 1}: Validation Loss: {valid_loss}")

# Save the Qformer state
torch.save(model.Qformer.state_dict(), 'models/stage1_out/qformer_stage1.pth')
