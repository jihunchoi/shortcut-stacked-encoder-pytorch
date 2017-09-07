"""
RNN packing/unpacking utility functions taken from
Yixin Nie's implementation
(https://github.com/easonnie/multiNLI_encoder)
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def pack_for_rnn_seq(inputs, lengths):
    """
    :param inputs: [T * B * D]
    :param lengths:  [B]
    :return:
    """
    _, sorted_indices = lengths.sort()
    '''
        Reverse to decreasing order
    '''
    r_index = reversed(list(sorted_indices))

    s_inputs_list = []
    lengths_list = []
    reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

    for j, i in enumerate(r_index):
        s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
        lengths_list.append(lengths[i])
        reverse_indices[i] = j

    reverse_indices = list(reverse_indices)

    s_inputs = torch.cat(s_inputs_list, 1)
    packed_seq = pack_padded_sequence(s_inputs, lengths_list)

    return packed_seq, reverse_indices


def unpack_from_rnn_seq(packed_seq, reverse_indices):
    unpacked_seq, _ = pad_packed_sequence(packed_seq)
    s_inputs_list = []

    for i in reverse_indices:
        s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
    return torch.cat(s_inputs_list, 1)
