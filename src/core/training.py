import pickle
import time

import torch
from tokenizers import Encoding
from torch import nn, optim
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.core.model import Encoder, Decoder, Seq2seq
from src.core.tokenizing import RedditTokenizer


class ChatbotDataset(Dataset):
    def __init__(self, pairs: list[(Encoding, Encoding)]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src = torch.Tensor(self.pairs[idx][0].ids).to(torch.int64)
        trg = torch.Tensor(self.pairs[idx][1].ids).to(torch.int64)
        return src, trg


def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sec2min(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, data_loader, optimizer, criterion, tf_ratio=0.5):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()
        output = model(src, trg, tf_ratio=tf_ratio)

        vocab_size = output.shape[-1]

        output = output.view(-1, vocab_size)
        trg = trg.view(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(data_loader):
            output = model(src, trg, tf_ratio=0)

            vocab_size = output.shape[-1]

            output = output.view(-1, vocab_size)
            trg = trg.view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


if __name__ == '__main__':
    with open('config.yaml') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)

    # -------------------- MODEL DEFINITION --------------------

    encoder = Encoder(params['Tokenizer']['vocab_size'],
                      params['Model']['lstm']['embedding_dim'],
                      params['Model']['lstm']['hidden_dim'],
                      params['Model']['lstm']['n_layers'],
                      params['Model']['lstm']['dropout'])
    decoder = Decoder(params['Tokenizer']['vocab_size'],
                      params['Model']['lstm']['embedding_dim'],
                      params['Model']['lstm']['hidden_dim'],
                      params['Model']['lstm']['n_layers'],
                      params['Model']['lstm']['dropout'])

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    m = Seq2seq(encoder, decoder, device)
    m.apply(init_weights)

    tkn = RedditTokenizer.load('data/tokenizers/vocab10000-max256.json')
    pad_id = tkn.tokenizer.token_to_id('[PAD]')

    opt = optim.Adam(m.parameters())
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)

    # -------------------- DATA LOADING --------------------
    with open('data/encoded_pairs/explainlikeimfive/top_comments.pickle', 'rb') as f:
        encoded_dataset = pickle.load(f)
    nb_examples = len(encoded_dataset)

    train_encoded_dataset = encoded_dataset[:int(0.8 * nb_examples)]
    val_encoded_dataset = encoded_dataset[int(0.8 * nb_examples):int(0.9 * nb_examples)]
    test_encoded_dataset = encoded_dataset[int(0.9 * nb_examples):]

    print(f'Train dataset : {len(train_encoded_dataset)} examples')
    print(f'Validation dataset : {len(val_encoded_dataset)} examples')
    print(f'Test dataset : {len(test_encoded_dataset)} examples')

    train_dataset = ChatbotDataset(train_encoded_dataset)
    val_dataset = ChatbotDataset(val_encoded_dataset)
    test_dataset = ChatbotDataset(test_encoded_dataset)

    train_data_loader = DataLoader(train_dataset, batch_size=params['Training']['batch_size'], shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=params['Training']['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=params['Training']['batch_size'], shuffle=False)

    # -------------------- TRAINING --------------------
    best_validation_loss = float('inf')

    for epoch in range(params['Training']['epochs']):
        start = time.time()

        train_loss = train(m, train_data_loader, opt, crit)
        val_loss = evaluate(m, val_data_loader, crit)

        end = time.time()

        minutes, seconds = sec2min(start, end)

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss

        print(f'[Epoch {epoch+1:02}] - [{minutes:02}min {seconds:02}sec]')
        print(f'\tLosses - [Train : {train_loss:.4f}] - [Validation : {val_loss:.4f}/{best_validation_loss:.4f}]')
