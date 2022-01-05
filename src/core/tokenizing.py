import json
import os
import pickle
from pprint import pp

import tokenizers
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece as WordPieceModel
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm


class RedditTokenizer:
    def __init__(self, vocab_size: int, max_seq_length: int):
        tokenizer = Tokenizer(WordPieceModel(unk_token="[UNK]"))
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_seq_length)
        tokenizer.enable_truncation(max_length=max_seq_length)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = WordPieceDecoder()
        trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

    def train(self, corpus: list[str]):
        self.tokenizer.train_from_iterator(corpus, self.trainer)

    def save(self):
        if not os.path.exists('data/tokenizers'):
            os.makedirs('data/tokenizers')
        self.tokenizer.save(f'data/tokenizers/vocab{self.vocab_size}-max{self.max_seq_length}.json', pretty=True)

    @staticmethod
    def load(path):
        tokenizer = tokenizers.Tokenizer.from_file(path)
        vocab_length = int(path.split('vocab')[1].split('-')[0])
        max_seq_length = int(path.split('max')[1].split('.')[0])
        reddit_tkn = RedditTokenizer(vocab_length, max_seq_length)
        reddit_tkn.tokenizer = tokenizer

        return reddit_tkn

    def encode_sequence(self, sequence: str):
        self.tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS]:0 $A:0 $B:1 [EOS]:1",
            special_tokens=[("[BOS]", self.tokenizer.token_to_id("[BOS]")),
                            ("[EOS]", self.tokenizer.token_to_id("[EOS]"))]
        )
        return self.tokenizer.encode(sequence)

    def encode_pairs(self, pairs: list[(str, str)]):
        encodings = []
        dismissed_pairs = 0
        for src_sentence, trg_sentence in tqdm(pairs):
            encoded_src = self.encode_sequence(src_sentence)
            encoded_trg = self.encode_sequence(trg_sentence)
            if encoded_src.overflowing or encoded_trg.overflowing:
                dismissed_pairs += 1
            else:
                encodings.append((encoded_src, encoded_trg))

        return dismissed_pairs, encodings

    def decode(self, ids: list[list[int]], skip_special_tokens=False):
        return self.tokenizer.decode_batch(ids, skip_special_tokens=skip_special_tokens)


if __name__ == '__main__':
    with open('data/pairs/explainlikeimfive/top_comments.json', 'r') as f:
        data = json.load(f)
    list_data = [item for sublist in data for item in sublist]

    tkn = RedditTokenizer.load('data/tokenizers/vocab10000-max256.json')
    # tkn = RedditTokenizer(10_000, 8)
    # tkn.train(list_data)
    # tkn.save()
    # print(data[:10])
    # dismissed, encoded = tkn.encode_pairs(data[:10])
    # print(dismissed)
    # print(len(encoded))
    # decoded_strings = tkn.decode([elem[1].ids for elem in encoded], skip_special_tokens=True)
    # pp(decoded_strings)
    dismissed, encoded = tkn.encode_pairs(data)
    print(dismissed)
    print(len(encoded))

    if not os.path.exists('data/encoded_pairs/explainlikeimfive'):
        os.makedirs('data/encoded_pairs/explainlikeimfive')
    with open('data/encoded_pairs/explainlikeimfive/top_comments.pickle', 'wb') as f:
        pickle.dump(encoded, f)
