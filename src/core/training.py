import time

import torch
from tokenizers import Encoding
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class LSTMDataset(Dataset):
    def __init__(self, pairs: list[(Encoding, Encoding)]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src = torch.Tensor(self.pairs[idx][0].ids).to(torch.int64)
        trg = torch.Tensor(self.pairs[idx][1].ids).to(torch.int64)
        return src, trg


class TransformerDataset(Dataset):
    def __init__(self, encodings: list[Encoding]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        src = torch.Tensor(self.encodings[idx].ids[:-1]).to(torch.int64)
        trg = torch.Tensor(self.encodings[idx].ids[1:]).to(torch.int64)
        return src, trg


def generate_src_mask(dim):
    return torch.triu(torch.ones(dim, dim) * float('-inf'), diagonal=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sec2min(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_lstm(model, data_loader, optimizer, criterion, device=torch.device('cpu'), tf_ratio=0.5):
    model = model.to(device)
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(data_loader):
        start = time.time()
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, tf_ratio=tf_ratio)
        vocab_size = output.shape[-1]

        output = output.view(-1, vocab_size)
        output = output.to(device)
        trg = trg.view(-1)
        trg = trg.to(device)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        end = time.time()

        print(f'Training batch {i + 1}/{len(data_loader)} - Loss : {loss.item()} - Time : {int(end - start)}s')

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def train_transformer(model, data_loader, optimizer, criterion, device=torch.device('cpu')):
    model = model.to(device)
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(total=len(data_loader))
    for i, (src, trg) in enumerate(data_loader):
        start = time.time()

        src = torch.permute(src, (1, 0)).to(device)
        trg = torch.permute(trg, (1, 0)).to(device)

        optimizer.zero_grad()

        seq_length = src.shape[0]
        output = model(src, generate_src_mask(seq_length).to(device))
        vocab_size = output.shape[-1]

        output = torch.permute(output, (1, 0, 2))
        output = output.reshape(-1, vocab_size)
        trg = trg.reshape(-1)
        trg = trg.to(device)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        end = time.time()

        # print(f'Training batch {i + 1}/{len(data_loader)} - Loss : {loss.item()} - Time : {int(end - start)}s')

        epoch_loss += loss.item()
        progress_bar.update(1)

    return epoch_loss / len(data_loader)


def evaluate_lstm(model, data_loader, criterion, tokenizer, file_path, device=torch.device('cpu')):
    model = model.to(device)
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, tf_ratio=0)

            src_strings = tokenizer.decode(src.tolist(), skip_special_tokens=False)
            trg_strings = tokenizer.decode(trg.tolist(), skip_special_tokens=False)
            output_strings = tokenizer.decode(output.argmax(-1).tolist(), skip_special_tokens=False)

            vocab_size = output.shape[-1]

            output = output.view(-1, vocab_size)
            output = output.to(device)

            trg = trg.view(-1)

            loss = criterion(output, trg)
            print(f'Validation batch {i + 1}/{len(data_loader)} - Loss : {loss.item()}')

            epoch_loss += loss.item()

            # writing model outputs to disk
            with open(file_path, 'a', encoding='utf-8') as file:
                for k in range(len(src_strings)):
                    file.write(f'<SOURCE> : {src_strings[k]}\n')
                    file.write(f'<TARGET> : {trg_strings[k]}\n')
                    file.write(f'<OUTPUT> : {output_strings[k]}\n\n')

    return epoch_loss / len(data_loader)


def evaluate_transformer(model, data_loader, criterion, tokenizer, file_path, device=torch.device('cpu')):
    model = model.to(device)
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(total=len(data_loader))
        for i, (src, trg) in enumerate(data_loader):
            src = torch.permute(src, (1, 0)).to(device)
            trg = torch.permute(trg, (1, 0)).to(device)
            seq_length = src.shape[0]
            output = model(src, generate_src_mask(seq_length).to(device))
            output = torch.permute(output, (1, 0, 2))

            src_strings = tokenizer.decode(torch.permute(src, (1, 0)).tolist(), skip_special_tokens=False)
            trg_strings = tokenizer.decode(torch.permute(trg, (1, 0)).tolist(), skip_special_tokens=False)
            output_strings = tokenizer.decode(output.argmax(-1).tolist(), skip_special_tokens=False)

            vocab_size = output.shape[-1]

            output = output.reshape(-1, vocab_size)
            output = output.to(device)

            trg = trg.reshape(-1)

            loss = criterion(output, trg)
            # print(f'Validation batch {i + 1}/{len(data_loader)} - Loss : {loss.item()}')

            epoch_loss += loss.item()

            # writing model outputs to disk
            with open(file_path, 'a', encoding='utf-8') as file:
                for k in range(len(src_strings)):
                    file.write(f'<SOURCE> : {src_strings[k]}\n')
                    file.write(f'<TARGET> : {trg_strings[k]}\n')
                    file.write(f'<OUTPUT> : {output_strings[k]}\n\n')
            progress_bar.update(1)

    return epoch_loss / len(data_loader)


if __name__ == '__main__':
    pass
