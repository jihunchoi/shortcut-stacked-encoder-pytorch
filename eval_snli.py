import argparse
import logging
import os

import numpy as np
import torch
from torchtext import data, datasets

from models import NLIModel


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def evaluate(args):
    lstm_hidden_dims = [int(d) for d in args.lstm_hidden_dims.split(',')]

    logging.info('Loading data...')
    text_field = data.Field(lower=True, include_lengths=True,
                            batch_first=False)
    label_field = data.Field(sequential=False)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    dataset_splits = datasets.SNLI.splits(
        text_field=text_field, label_field=label_field, root=args.data_dir)
    test_dataset = dataset_splits[2]
    text_field.build_vocab(*dataset_splits)
    label_field.build_vocab(*dataset_splits)
    _, _, test_loader = data.BucketIterator.splits(
        datasets=dataset_splits, batch_size=args.batch_size, device=args.gpu)

    logging.info('Building model...')
    num_classes = len(label_field.vocab)
    num_words = len(text_field.vocab)
    model = NLIModel(num_words=num_words, word_dim=args.word_dim,
                     lstm_hidden_dims=lstm_hidden_dims,
                     mlp_hidden_dim=args.mlp_hidden_dim,
                     mlp_num_layers=args.mlp_num_layers,
                     num_classes=num_classes, dropout_prob=0)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda(args.gpu)

    num_total_params = sum(np.prod(p.size()) for p in model.parameters())
    num_word_embedding_params = np.prod(model.word_embedding.weight.size())

    logging.info(f'# of total parameters: {num_total_params}')
    logging.info(f'# of intrinsic parameters: '
                 f'{num_total_params - num_word_embedding_params}')
    logging.info(f'# of word embedding parameters: '
                 f'{num_word_embedding_params}')

    num_correct = 0
    num_data = len(test_dataset)
    for batch in test_loader:
        pre_input, pre_lengths = batch.premise
        hyp_input, hyp_lengths = batch.hypothesis
        label = batch.label
        model_output = model(pre_input=pre_input, pre_lengths=pre_lengths,
                             hyp_input=hyp_input, hyp_lengths=hyp_lengths)
        label_pred = model_output.max(1)[1]
        num_correct_batch = torch.eq(label, label_pred).long().sum()
        num_correct_batch = num_correct_batch.data[0]
        num_correct += num_correct_batch
    print(f'# of test sentences: {num_data}')
    print(f'# of correct predictions: {num_correct}')
    print(f'Accuracy: {num_correct / num_data:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/snli')
    parser.add_argument('--word-dim', type=int, default=300)
    parser.add_argument('--lstm-hidden-dims', default='512,1024,2048')
    parser.add_argument('--mlp-hidden-dim', type=int, default=1600)
    parser.add_argument('--mlp-num-layers', type=int, default=2)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
