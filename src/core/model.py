import math
import random

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class LSTMEncoder(nn.Module):
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


class LSTMDecoder(nn.Module):
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


class LSTMSeq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, tf_ratio=0.5):
        batch_size = trg.shape[0]
        sequence_length = trg.shape[1]
        vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(batch_size, sequence_length, vocab_size)
        h, c = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, sequence_length):
            output, h, c = self.decoder(input_token, h, c)
            outputs[:, t, :] = output
            tf = random.random() < tf_ratio
            top_prediction = output.argmax(1)
            input_token = trg[:, t] if tf else top_prediction
        return outputs

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


class GPTLike(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, hidden_dim, n_layers, dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == '__main__':
    pass
