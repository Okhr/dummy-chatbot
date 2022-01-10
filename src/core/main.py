import pickle
import time

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from src.core.model import LSTMEncoder, LSTMDecoder, LSTMSeq2seq, GPTLike
from src.core.processing import make_pairs
from src.core.stats import *

from src.core.collecting import download_top_threads
from dotenv import load_dotenv
import yaml

from src.core.tokenizing import RedditTokenizer
from src.core.training import LSTMDataset, train_lstm, evaluate_lstm, sec2min, TransformerDataset, train_transformer, \
    evaluate_transformer, count_parameters

if __name__ == '__main__':
    load_dotenv()

    with open('config.yaml') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)

    print()
    print('-------------------------------------------------------------------------------')
    print('------------------------------ COMMENT GATHERING ------------------------------')
    print('-------------------------------------------------------------------------------')
    print()

    subreddit = params['Collector']['subreddit']

    if params['Collector']['active']:
        for time_filter in params['Collector']['time_filters']:
            print(f'Downloading <{time_filter}> top comments')
            download_top_threads(subreddit, time_filter)
        print(f"Total downloaded comments : {get_number_of_comments(subreddit)}")
    else:
        print('This step is not active')

    print()
    print('---------------------------------------------------------------------------')
    print('------------------------------ COMMENT PAIRS ------------------------------')
    print('---------------------------------------------------------------------------')
    print()

    if params['Pairs']['active']:
        print('Making pairs including all comments')
        make_pairs(params['Collector']['subreddit'], top_comments=False)
        if params['Pairs']['top_comments']:
            print('Making pairs including top comments')
            make_pairs(params['Collector']['subreddit'], top_comments=True)
    else:
        print('This step is not active')

    print()
    print('--------------------------------------------------------------------------------')
    print('------------------------------ TOKENIZER TRAINING ------------------------------')
    print('--------------------------------------------------------------------------------')
    print()

    vocab_size = params['Tokenizer']['vocab_size']
    max_seq_length = params['Tokenizer']['max_seq_length']

    if params['Tokenizer']['training_active']:
        print(f'Training tokenizer with parameters <vocab_size:{vocab_size}> <max_seq_length:{max_seq_length}>')
        with open(f'data/pairs/{subreddit}/all_comments.json', 'r') as f:
            all_data = json.load(f)
        list_data = [item for sublist in all_data for item in sublist]
        tkn = RedditTokenizer(vocab_size, max_seq_length)
        tkn.train(list_data)
        print('Saving tokenizer')
        tkn.save()

    if params['Tokenizer']['dataset_making_active']:
        with open(f'data/pairs/{subreddit}/all_comments.json', 'r') as f:
            all_data = json.load(f)
        with open(f'data/pairs/{subreddit}/top_comments.json', 'r') as f:
            top_data = json.load(f)

        tkn = RedditTokenizer.load(f'data/tokenizers/vocab{vocab_size}_seq{max_seq_length}.json')

        if not os.path.exists(f'data/encoded_pairs/{subreddit}'):
            os.makedirs(f'data/encoded_pairs/{subreddit}')
        if not os.path.exists(f'data/concat_encoded_pairs/{subreddit}'):
            os.makedirs(f'data/concat_encoded_pairs/{subreddit}')

        # LSTM data
        print('Making encoded dataset with all comments')
        dismissed, encoded = tkn.encode_pairs(all_data)
        print(f'Encoded {len(encoded)} pairs from all comments, {dismissed} dismissed pairs')
        with open(f"data/encoded_pairs/{subreddit}/all_vocab{vocab_size}_seq{max_seq_length}.pickle", 'wb') as f:
            pickle.dump(encoded, f)

        print('Making encoded dataset with top comments')
        dismissed, encoded = tkn.encode_pairs(top_data)
        print(f'Encoded {len(encoded)} pairs from top comments, {dismissed} dismissed pairs')
        with open(f"data/encoded_pairs/{subreddit}/top_vocab{vocab_size}_seq{max_seq_length}.pickle", 'wb') as f:
            pickle.dump(encoded, f)

        # Transformer data
        print('Making concat encoded dataset with all comments')
        dismissed, encoded = tkn.encode_pairs_concat(all_data)
        print(f'Encoded {len(encoded)} concatenated pairs from all comments, {dismissed} dismissed pairs')
        with open(f"data/concat_encoded_pairs/{subreddit}/all_vocab{vocab_size}_seq{max_seq_length}.pickle", 'wb') as f:
            pickle.dump(encoded, f)

        print('Making concat encoded dataset with top comments')
        dismissed, encoded = tkn.encode_pairs_concat(top_data)
        print(f'Encoded {len(encoded)} concatenated pairs from top comments, {dismissed} dismissed pairs')
        with open(f"data/concat_encoded_pairs/{subreddit}/top_vocab{vocab_size}_seq{max_seq_length}.pickle", 'wb') as f:
            pickle.dump(encoded, f)

    if not params['Tokenizer']['training_active'] and not params['Tokenizer']['dataset_making_active']:
        print('This step is not active')

    print()
    print('----------------------------------------------------------------------------')
    print('------------------------------ MODEL TRAINING ------------------------------')
    print('----------------------------------------------------------------------------')
    print()

    if params['Training']['active']:

        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Device : {torch.cuda.get_device_name(dev)}')

        epochs_done = 0
        dataset_type = params['Training']['dataset_type']

        # --------------- MODEL DEFINITION ---------------
        model_type = params['Training']['model_type']

        assert model_type == 'lstm' or model_type == 'transformer'

        if model_type == 'lstm':
            embedding_dim = params['Training']['lstm']['embedding_dim']
            hidden_dim = params['Training']['lstm']['hidden_dim']
            n_layers = params['Training']['lstm']['n_layers']
            dropout = params['Training']['lstm']['dropout']
            timestamp = int(time.time())

            model_name = f"lstm_vocab{vocab_size}_emb{embedding_dim}_hidden{hidden_dim}_nlayers{n_layers}_dropout{dropout}_{timestamp}.ckpt"

            if params['Training']['model'] != 'new':
                model_name = params['Training']['model']
                embedding_dim = int(model_name.split('emb')[1].split('_')[0])
                hidden_dim = int(model_name.split('hidden')[1].split('_')[0])
                n_layers = int(model_name.split('nlayers')[1].split('_')[0])
                dropout = float(model_name.split('dropout')[1].split('_')[0])

            encoder = LSTMEncoder(vocab_size,
                                  embedding_dim,
                                  hidden_dim,
                                  n_layers,
                                  dropout)

            decoder = LSTMDecoder(vocab_size,
                                  embedding_dim,
                                  hidden_dim,
                                  n_layers,
                                  dropout)

            m = LSTMSeq2seq(encoder, decoder)
            m.to(dev)
            m.init_weights()

            tkn = RedditTokenizer.load(f"data/tokenizers/vocab{vocab_size}_seq{max_seq_length}.json")
            pad_id = tkn.tokenizer.token_to_id('[PAD]')

            opt = optim.Adam(m.parameters())
            crit = nn.CrossEntropyLoss(ignore_index=pad_id)

        else:
            # transformer type
            d_model = params['Training']['transformer']['d_model']
            n_heads = params['Training']['transformer']['n_heads']
            hidden_dim = params['Training']['transformer']['hidden_dim']
            n_layers = params['Training']['transformer']['n_layers']
            dropout = params['Training']['transformer']['dropout']
            timestamp = int(time.time())

            model_name = f"transformer_vocab{vocab_size}_dmodel{d_model}_nheads{n_heads}_hiddendim{hidden_dim}_nlayers{n_layers}_dropout{dropout}_{timestamp}.ckpt"

            if params['Training']['model'] != 'new':
                model_name = params['Training']['model']
                d_model = int(model_name.split('dmodel')[1].split('_')[0])
                n_heads = int(model_name.split('nheads')[1].split('_')[0])
                hidden_dim = int(model_name.split('hiddendim')[1].split('_')[0])
                n_layers = int(model_name.split('nlayers')[1].split('_')[0])
                dropout = float(model_name.split('dropout')[1].split('_')[0])

            m = GPTLike(vocab_size, d_model, n_heads, hidden_dim, n_layers, dropout)
            m.to(dev)

            tkn = RedditTokenizer.load(f"data/tokenizers/vocab{vocab_size}_seq{max_seq_length}.json")
            pad_id = tkn.tokenizer.token_to_id('[PAD]')

            # opt = optim.Adam(m.parameters(), lr=2.5e-4)
            opt = torch.optim.SGD(m.parameters(), lr=5.0)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.95)

            crit = nn.CrossEntropyLoss(ignore_index=pad_id)

        # load the appropriate weights and epoch number if we are training an already existing model
        if params['Training']['model'] != 'new':
            checkpoint = torch.load(f'models/{model_name}')
            m.load_state_dict(checkpoint['model_state_dict'])
            m.to(dev)
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs_done = checkpoint['epoch']
            saved_loss = checkpoint['loss']

        print(f'Model parameters : {count_parameters(m)}')

        # --------------- DATA LOADING ---------------
        if model_type == 'lstm':
            with open(f'data/encoded_pairs/{subreddit}/{dataset_type}_vocab{vocab_size}_seq{max_seq_length}.pickle',
                      'rb') as f:
                encoded_dataset = pickle.load(f)
            nb_examples = len(encoded_dataset)
        else:
            with open(
                    f'data/concat_encoded_pairs/{subreddit}/{dataset_type}_vocab{vocab_size}_seq{max_seq_length}.pickle',
                    'rb') as f:
                encoded_dataset = pickle.load(f)
            nb_examples = len(encoded_dataset)

        train_encoded_dataset = encoded_dataset[:int(0.9 * nb_examples)]
        val_encoded_dataset = encoded_dataset[int(0.9 * nb_examples):]

        print(f'Train dataset : {len(train_encoded_dataset)} examples')
        print(f'Validation dataset : {len(val_encoded_dataset)} examples')
        print()

        if model_type == 'lstm':
            train_dataset = LSTMDataset(train_encoded_dataset)
            val_dataset = LSTMDataset(val_encoded_dataset)
        else:
            train_dataset = TransformerDataset(train_encoded_dataset)
            val_dataset = TransformerDataset(val_encoded_dataset)

        train_data_loader = DataLoader(train_dataset, batch_size=params['Training']['batch_size'], shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=params['Training']['batch_size'], shuffle=False)
        del train_encoded_dataset
        del val_encoded_dataset

        # --------------- TRAINING ---------------
        best_validation_loss = float('inf')

        if params['Training']['model'] == 'new':
            print(f"Training a new model with parameters :")
            print(' '.join([f'<{k}:{v}>' for k, v in params['Training'][params['Training']['model_type']].items()]))
            print()
        else:
            print(f'Training model : <{model_name}>')
            print()
            best_validation_loss = saved_loss

        for epoch in range(1, params['Training']['epochs'] + 1):
            start = time.time()

            actual_epoch = epochs_done + epoch

            print(f"[Epoch {actual_epoch:02}][Training session epoch {epoch}/{params['Training']['epochs']}]")

            if model_type == 'lstm':
                train_loss = train_lstm(m, train_data_loader, opt, crit, device=dev)
            else:
                train_loss = train_transformer(m, train_data_loader, opt, crit, device=dev)

            if not os.path.exists(f"logs/{model_name.split('.ckpt')[0]}"):
                os.makedirs(f"logs/{model_name.split('.ckpt')[0]}")

            if model_type == 'lstm':
                val_loss = evaluate_lstm(m, val_data_loader, crit, tkn,
                                         f"logs/{model_name.split('.ckpt')[0]}/epoch{actual_epoch}.txt", device=dev)
            else:
                val_loss = evaluate_transformer(m, val_data_loader, crit, tkn,
                                                f"logs/{model_name.split('.ckpt')[0]}/epoch{actual_epoch}.txt", device=dev)

            end = time.time()

            minutes, seconds = sec2min(start, end)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss

                # saving model
                if not os.path.exists('models'):
                    os.makedirs('models')
                m.to(torch.device('cpu'))
                torch.save({
                    'epoch': actual_epoch,
                    'loss': val_loss,
                    'model_state_dict': m.state_dict(),
                    'optimizer_state_dict': opt.state_dict()
                }, f'models/{model_name}')

            print(
                f'Epoch summary : [Train loss : {train_loss:.4f}] - [Validation loss : {val_loss:.4f}/{best_validation_loss:.4f}] - [{minutes:02}min {seconds:02}sec] - [LR : {scheduler.get_last_lr()}]')
            print()

            scheduler.step()

    else:
        print('This step is not active')
