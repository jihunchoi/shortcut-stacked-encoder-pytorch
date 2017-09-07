import torch
from torch import nn

from utils import pack_for_rnn_seq, unpack_from_rnn_seq


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        mlp_layers = []
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = hidden_dim
            linear_layer = nn.Linear(in_features=layer_input_dim,
                                     out_features=hidden_dim)
            relu_layer = nn.ReLU()
            dropout_layer = nn.Dropout(dropout_prob)
            mlp_layer = nn.Sequential(linear_layer, relu_layer, dropout_layer)
            mlp_layers.append(mlp_layer)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, input):
        """
        Args:
            input (Variable): A float variable of size
                (batch_size, input_dim).

        Returns:
            output (Variable): A float variable of size
                (batch_size, hidden_dim), which is the result of
                applying MLP to the input argument.
        """

        return self.mlp(input)


class NLIClassifier(nn.Module):

    def __init__(self, sentence_dim, hidden_dim, num_layers, num_classes,
                 dropout_prob):
        super().__init__()
        self.sentence_dim = sentence_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        self.mlp = MLP(input_dim=4 * sentence_dim, hidden_dim=hidden_dim,
                       num_layers=num_layers, dropout_prob=dropout_prob)
        self.clf_linear = nn.Linear(in_features=hidden_dim,
                                    out_features=num_classes)

    def forward(self, pre, hyp):
        mlp_input = torch.cat([pre, hyp, (pre - hyp).abs(), pre * hyp], dim=1)
        mlp_output = self.mlp(mlp_input)
        output = self.clf_linear(mlp_output)
        return output


class ShortcutStackedEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        for i in range(self.num_layers):
            lstm_input_dim = input_dim + 2*sum(hidden_dims[:i])
            lstm_layer = nn.LSTM(
                input_size=lstm_input_dim, hidden_size=hidden_dims[i],
                bidirectional=True, batch_first=False)
            setattr(self, f'lstm_layer_{i}', lstm_layer)

    def get_lstm_layer(self, i):
        return getattr(self, f'lstm_layer_{i}')

    def forward(self, input, lengths):
        prev_lstm_output = None
        lstm_input = input
        for i in range(self.num_layers):
            if i > 0:
                lstm_input = torch.cat([lstm_input, prev_lstm_output], dim=2)
            lstm_input_packed, reverse_indices = pack_for_rnn_seq(
                inputs=lstm_input, lengths=lengths)
            lstm_layer = self.get_lstm_layer(i)
            lstm_output_packed, _ = lstm_layer(lstm_input_packed)
            lstm_output = unpack_from_rnn_seq(
                packed_seq=lstm_output_packed, reverse_indices=reverse_indices)
            prev_lstm_output = lstm_output
        sentence_vector = torch.max(prev_lstm_output, dim=0)[0]
        return sentence_vector

class NLIModel(nn.Module):

    def __init__(self, num_words, word_dim, lstm_hidden_dims,
                 mlp_hidden_dim, mlp_num_layers, num_classes, dropout_prob):
        super().__init__()
        self.num_words = num_words
        self.word_dim = word_dim
        self.lstm_hidden_dims = lstm_hidden_dims
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        self.encoder = ShortcutStackedEncoder(
            input_dim=word_dim, hidden_dims=lstm_hidden_dims)
        self.classifier = NLIClassifier(
            sentence_dim=2 * lstm_hidden_dims[-1], hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers, num_classes=num_classes,
            dropout_prob=dropout_prob)

    def forward(self, pre_input, pre_lengths, hyp_input, hyp_lengths):
        """
        Args:
            pre_input (Variable): A long variable containing indices for
                premise words. Size: (max_length, batch_size).
            pre_lengths (Tensor): A long tensor containing lengths for
                sentences in the premise batch.
            hyp_input (Variable): A long variable containing indices for
                hypothesis words. Size: (max_length, batch_size).
            pre_lengths (Tensor): A long tensor containing lengths for
                sentences in the hypothesis batch.

        Returns:
            output (Variable): A float variable containing the
                unnormalized probability for each class
        :return:
        """

        pre_input_emb = self.word_embedding(pre_input)
        hyp_input_emb = self.word_embedding(hyp_input)
        pre_vector = self.encoder(input=pre_input_emb, lengths=pre_lengths)
        hyp_vector = self.encoder(input=hyp_input_emb, lengths=hyp_lengths)
        classifier_output = self.classifier(pre=pre_vector, hyp=hyp_vector)
        return classifier_output
