import argparse
import logging
import os

import numpy as np
import tensorboard
import torch
from tensorboard import summary
from torch import nn, optim
from torch.optim import lr_scheduler
from torchtext import data, datasets

from models import NLIModel


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def train(args):
    experiment_name = (f'w{args.word_dim}_lh{args.lstm_hidden_dims}'
                       f'_mh{args.mlp_hidden_dim}_ml{args.mlp_num_layers}'
                       f'_d{args.dropout_prob}')
    save_dir = os.path.join(args.save_root_dir, experiment_name)
    train_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(save_dir, 'log', 'train'))
    valid_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(save_dir, 'log', 'valid'))

    lstm_hidden_dims = [int(d) for d in args.lstm_hidden_dims.split(',')]

    logging.info('Loading data...')
    text_field = data.Field(lower=True, include_lengths=True,
                            batch_first=False)
    label_field = data.Field(sequential=False)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    dataset_splits = datasets.SNLI.splits(
        text_field=text_field, label_field=label_field, root=args.data_dir)
    text_field.build_vocab(*dataset_splits, vectors=args.pretrained)
    label_field.build_vocab(*dataset_splits)
    train_loader, valid_loader, _ = data.BucketIterator.splits(
        datasets=dataset_splits, batch_size=args.batch_size, device=args.gpu)

    logging.info('Building model...')
    num_classes = len(label_field.vocab)
    num_words = len(text_field.vocab)
    model = NLIModel(num_words=num_words, word_dim=args.word_dim,
                     lstm_hidden_dims=lstm_hidden_dims,
                     mlp_hidden_dim=args.mlp_hidden_dim,
                     mlp_num_layers=args.mlp_num_layers,
                     num_classes=num_classes, dropout_prob=args.dropout_prob)
    num_total_params = sum(np.prod(p.size()) for p in model.parameters())
    num_word_embedding_params = np.prod(model.word_embedding.weight.size())
    if args.pretrained:
        model.word_embedding.weight.data.set_(text_field.vocab.vectors)
    model.cuda(args.gpu)

    logging.info(f'# of total parameters: {num_total_params}')
    logging.info(f'# of intrinsic parameters: '
                 f'{num_total_params - num_word_embedding_params}')
    logging.info(f'# of word embedding parameters: '
                 f'{num_word_embedding_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=2e-4)
    # Halve LR every two epochs
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=2,
                                    gamma=0.5)

    def run_iter(batch, is_training):
        pre_input, pre_lengths = batch.premise
        hyp_input, hyp_lengths = batch.hypothesis
        label = batch.label
        model.train(is_training)
        model_output = model(pre_input=pre_input, pre_lengths=pre_lengths,
                             hyp_input=hyp_input, hyp_lengths=hyp_lengths)
        label_pred = model_output.max(1)[1]
        loss = criterion(input=model_output, target=label)
        accuracy = torch.eq(label, label_pred).float().mean()
        if is_training:
            model.zero_grad()
            loss.backward()
            optimizer.step()
        return loss, accuracy

    def add_scalar_summary(summary_writer, name, value, step):
        summ = summary.scalar(name=name, scalar=value)
        summary_writer.add_summary(summary=summ, global_step=step)

    logging.info('Training starts!')
    cur_epoch = 0
    for iter_count, train_batch in enumerate(train_loader):
        train_loss, train_accuracy = run_iter(
            batch=train_batch, is_training=True)
        add_scalar_summary(
            summary_writer=train_summary_writer,
            name='loss', value=train_loss.data[0], step=iter_count)
        add_scalar_summary(
            summary_writer=train_summary_writer,
            name='accuracy', value=train_accuracy.data[0], step=iter_count)

        if int(train_loader.epoch) > cur_epoch:
            cur_epoch = int(train_loader.epoch)
            num_valid_batches = len(valid_loader)
            valid_loss_sum = valid_accracy_sum = 0
            for valid_batch in valid_loader:
                valid_loss, valid_accuracy = run_iter(
                    batch=valid_batch, is_training=False)
                valid_loss_sum += valid_loss.data[0]
                valid_accracy_sum += valid_accuracy.data[0]
            valid_loss = valid_loss_sum / num_valid_batches
            valid_accuracy = valid_accracy_sum / num_valid_batches
            add_scalar_summary(
                summary_writer=valid_summary_writer,
                name='loss', value=valid_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=valid_summary_writer,
                name='accuracy', value=valid_accuracy, step=iter_count)
            scheduler.step()
            progress = train_loader.epoch
            logging.info(f'Epoch {progress:.2f}: '
                         f'valid loss = {valid_loss:.4f}, '
                         f'valid accuracy = {valid_accuracy:.4f}')
            model_filename = (f'model-{progress:.2f}'
                              f'-{valid_loss:.4f}'
                              f'-{valid_accuracy:.4f}.pkl')
            model_path = os.path.join(save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            logging.info(f'Saved the model to: {model_path}')

            if progress > args.max_epoch:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/snli')
    parser.add_argument('--word-dim', type=int, default=300)
    parser.add_argument('--lstm-hidden-dims', default='512,1024,2048')
    parser.add_argument('--mlp-hidden-dim', type=int, default=1600)
    parser.add_argument('--mlp-num-layers', type=int, default=2)
    parser.add_argument('--pretrained', default='glove.840B.300d')
    parser.add_argument('--dropout-prob', type=float, default=0.1)
    parser.add_argument('--save-root-dir', default='./trained/snli')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--max-epoch', type=int, default=5)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
