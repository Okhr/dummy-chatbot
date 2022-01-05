import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (h, c) = self.lstm(embedded)
        return h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, trg, h, c):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        output, (h, c) = self.lstm(embedded, (h, c))
        prediction = self.fc(output.squeeze(1))
        return prediction, h, c


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, tf_ratio=0.5):
        src = src.to(self.device)
        trg = trg.to(self.device)
        batch_size = trg.shape[0]
        sequence_length = trg.shape[1]
        vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(batch_size, sequence_length, vocab_size).to(self.device)
        h, c = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, sequence_length):
            output, h, c = self.decoder(input_token, h, c)
            outputs[:, t, :] = output
            tf = random.random() < tf_ratio
            top_prediction = output.argmax(1)
            input_token = trg[:, t] if tf else top_prediction
        return outputs


if __name__ == '__main__':
    pass
