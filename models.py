import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var

import numpy as np

# Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
# Works for both non-batched and batched inputs
class EmbeddingLayer(nn.Module):
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, input_dim, full_dict_size, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        # nn.Embedding(num_embeddings = size of dictionary of embeddings, embedding_dim = size of each embedding vector)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input):
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings

# One-layer RNN encoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class RNNEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, dropout, bidirect):
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=dropout, bidirectional=self.bidirect)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
            # print("h_t is new_h, new_c. new_h size: {}, new_c size: {}".format(new_h.size(), new_c.size()))
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)

        return (output, context_mask, h_t)

class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, dropout = 0.2, batch_first = True):
        super().__init__()
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=batch_first,)
        self.W = nn.Linear(hidden_size, out_size)
        # self.logsoft = nn.LogSoftmax(dim=2)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, emb, hid_in):
        # emb should be [seq len x batch size x embedding size] (Default [1 x 1 x 100]

        # Implementing dropout layer on input
        inp = self.dropout(emb)

        # hid_out is tuple with (h, c)
        output, hid_out = self.lstm(inp, hid_in)
        (hn, cn) = hid_out
        # print("hid_out size ", hid_out.size())

        # return a softmax over the output, and the hidden layer to pass in to the next decoder cell
        return self.logsoft(self.W(hn[0])), hid_out

class AttnDecoder(nn.Module):
    def __init__(self, inp_size, hid_size, out_size, args, dropout = 0.2):
        super().__init__()
        self.args = args
        self.lstm = nn.LSTM(inp_size, hid_size,)
        self.out = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.enc_reduce = nn.Linear(2*hid_size, hid_size)
        # in-between linear layer after output of lstm
        self.attention = nn.Linear(hid_size, hid_size)
        self.softmax = nn.Softmax(dim=1)
        self.soft0 = nn.Softmax(dim=0)
        self.logsoft = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        # to be used after concatenation, input to Wc is 2x hidden size, output is hidden size
        self.Wc = nn.Linear(hid_size * 2, hid_size)
        # used with softmax to get output. Not using softmax in this model because I'm using CrossEntropyLoss
        self.Ws = nn.Linear(hid_size, out_size)
        self.Wout = nn.Linear(hid_size * 2, out_size)


    def forward(self, emb, hidden, enc_outs):
        """

        :param emb: embedded input of one word, should be three dimensional for LSTM. Comes in as Tensor[Sent_len, Batch_size, embedding_size]. Currently Tensor[19,1,200]
        :param hidden: tuple containing (h0, c0) inputs to LSTM
        :param enc_outs: Encoder outputs for each word in the sentence. Shape is (seq x batch x embedded)
        :return:
        """

        # If encoder is bidirectional, then its output will be
        # [sent len x batch size x 2*hidden size], so we need to reduce 3rd dimension to hidden size
        if self.args.bidirectional:
            enc_outs = self.enc_reduce(enc_outs)

        x = self.dropout(emb)

        # call this output i as seen in class notes
        # out_i, h_out, c_out size = Tensor[1,1,200].
        out_i, (h_out, c_out) = self.lstm(x, hidden)

        # removing "batch" dimension from encoder outputs so they can be multiplied below
        enc_outs = enc_outs.squeeze(1)
        # Copy h_out and make it 2D for matrix math below. Will return h_out from forward() because it is used
        # as input to LSTM which expects 3D tensor
        h_i = h_out.squeeze(1)

        # Performing bilinear operation for f(), called "general" in Luong attention paper
        # self.attention(h_i) is multiplying weight vector by h_i output from lstm
        # Multiplying matrices as such:
        e_ij = torch.matmul(h_i, self.attention(enc_outs).t())   # tensor.t() transposes 2d tensor
        # e_ij = torch.mm(h_i, enc_outs.t())
        attn_weights = self.softmax(e_ij)
        c_ij = torch.mm(attn_weights, enc_outs)

        # concatenate along dimension 1
        # Note: Look at attention paper from graham
        hc = torch.cat([h_i, c_ij], dim=1)
        # ht_bar = self.tanh(self.Wc(hc))
        # attn_out = self.logsoft(self.Ws(ht_bar))
        attn_out = self.logsoft(self.Wout(hc))
        # print("attnout size: ", attn_out.size())

        return attn_out, (h_out, c_out)
