import random

from torch.utils.data import Dataset


class CodeTranslationDataset(Dataset):
    """Custom Dataset for loading code translation pairs"""

    def __init__(self, source_file, target_file, is_t5=False, is_llama=False):
        self.source_data = self._load_data(source_file)
        self.target_data = self._load_data(target_file)
        self.is_t5 = is_t5
        self.is_llama = is_llama

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        assert len(self.source_data) == len(self.target_data), "Source and target data must be the same length"
        return len(self.source_data)

    def __getitem__(self, idx):
        source_sample = self.source_data[idx]
        target_sample = self.target_data[idx]
        if self.is_t5:
            middle = random.randint(0, len(target_sample))
            ti = target_sample[:middle]
            to = target_sample[middle:]
            return {"source_code": source_sample, "target_code_input": ti, "target_code_output": to}
        if self.is_llama:
            return source_sample

        return {"source_code": source_sample, "target_code": target_sample}