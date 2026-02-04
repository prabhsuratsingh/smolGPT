import torch

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i+2 for i, ch in enumerate(chars)}
    stoi["<pad>"] = 0
    stoi["<bos>"] = 1
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return [stoi[ch] for ch in text]

def make_dataset(data, block_size):
    inputs = []
    targets = []

    for i in range(len(data) - block_size):
        chunk = data[i : i + block_size]
        inputs.append(
            torch.cat([torch.tensor([1]), chunk[:-1]])  # <bos> + shift
        )
        targets.append(chunk)

    return torch.stack(inputs), torch.stack(targets)


from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, stoi, block_size):
        self.data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size]
        x = torch.cat([torch.tensor([1]), chunk[:-1]])  # <bos> + shift
        y = chunk
        return x, y
