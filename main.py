import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *

PAD_POS = 0
SOS_POS = 1
UNK_POS = 1
EOS_POS = 2

exact = 0.0
total_sentences = 0.0

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=20, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')
    parser.add_argument('--no_reverse_input', dest='reverse_input', default=True, action='store_false', help='disable_input_reversal')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help='dropout rate internal to encoder RNN')
    parser.add_argument('--dec_dropout', type=float, default=0.2, help='dropout for input to decoder')

    # Additional arguments:
    parser.add_argument('--debug', dest='debug', default=False, action="store_true", help="Set into debug mode and use less training data")
    parser.add_argument('--copy', dest='copy', default=False, action="store_true", help="Test that the decoder model can copy")
    args = parser.parse_args()
    return args


# Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
# returns the associated logical form.
class NearestNeighborSemanticParser(object):
    # Take any arguments necessary for parsing
    def __init__(self, training_data):
        self.training_data = training_data

    # decode should return a list of k-best lists of Derivations. A Derivation consists of the underlying Example,
    # a probability, and a tokenized output string. If you're just doing one-best decoding of example ex and you
    # produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
    def decode(self, test_data):
        # Find the highest word overlap with the test data
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(object):
    def __init__(self, model_dec, model_enc, model_input_emb, model_output_emb, output_indexer, args):
        self.model_dec = model_dec
        self.model_enc = model_enc
        self.model_input_emb = model_input_emb
        self.model_output_emb = model_output_emb
        self.max_out_len = args.decoder_len_limit
        self.output_indexer = output_indexer

    def decode(self, test_data):
        self.model_dec.eval()
        self.model_enc.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()

        test_derivs = []    # Will hold final Derivation objects from decoder
        test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
        # Iterate through all of the test data
        for pair_idx in range(len(test_data)):
            predictions, pred_values = [], []   # These will hold the predictions from a sentence, and their values

            # Get the input sequence for a single pair, then get it's length
            in_seq = torch.as_tensor(test_data[pair_idx].x_indexed).unsqueeze(0)
            # in_len is 1D size [sentence length] tensor
            in_len = torch.as_tensor(len(test_data[pair_idx].x_indexed)).view(1)
            # Get the output sequence for a pair
            # out_seq = torch.as_tensor(test_data[pair_idx].y_indexed).view(-1)

            (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = encode_input_for_decoder(
                in_seq, in_len, self.model_input_emb, self.model_enc)

            # Set up first inputs to decoder
            dec_input = torch.as_tensor(SOS_POS).unsqueeze(0).unsqueeze(0)
            #
            dec_hidden = enc_final_states_reshaped

            for out_idx in range(self.max_out_len):
                prediction, pred_val, dec_hidden = self.dec_and_predict(dec_input, dec_hidden)
                dec_input = torch.as_tensor(prediction).unsqueeze(0).unsqueeze(0)
                if prediction != EOS_POS:
                    # Append the predicted TOKEN
                    predictions.append(self.output_indexer.get_object(prediction))
                    pred_values.append(pred_val)
                else:
                    # if the decode predicts EOS, then break the loop on this sentence
                    break
            # print(predictions)
            test_derivs.append([Derivation(test_data[pair_idx], pred_values, predictions)])

        return test_derivs


    def dec_and_predict(self, dec_input, dec_hidden):
        # decode_ouput embeds dec_input, then passes it and dec_hidden to the decoder model
        # hid_out is tuple, each element is 3D tensor w/ size [1 x 1 x hidden_size]
        # dec_out is 3D tensor w/ size [1, 1, output vocab size = 153]
        dec_out, dec_hidden = decode_output(dec_input, dec_hidden, self.model_dec, self.model_output_emb)
        # Determine predicted index and its value
        pred_val, pred_idx = dec_out.topk(1)

        return int(pred_idx), pred_val, dec_hidden

# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


# Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
# inp_lens_tensor lengths.
# x_tensor: batch size x sent len tensor of input token indices
# inp_lens: batch size length vector containing the length of each sentence in the batch
# model_input_emb: EmbeddingLayer
# model_enc: RNNEncoder
# Returns the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting

# E.g., calling this with x_tensor (0 is pad token):
# [[12, 25, 0, 0],
#  [1, 2, 3, 0],
#  [2, 0, 0, 0]]
# inp_lens = [2, 3, 1]
# will return outputs with the following shape:
# enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
# enc_final_states = 3 x dim
def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    # Assuming I am not going to implement batching:
    # x_tensor is size [1 x sentence length], inp_lens_tensor is size [sentence_length]
    # input_emb is 3D Tensor w/ size [1 x sentence length x embedding size]
    input_emb = model_input_emb.forward(x_tensor)
    # enc_output_each_word is 3D tensor w/ size [Sentence Length x Batch Size x 2 * hidden_size (400)]
    # enc_final_states_reshaped is
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    # enc_final_states_reshaped is tuple w/ two 3D tensors of size [1 x 1 x hidden_size]
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))

    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data, test_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)


    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    # len(output_indexer) is 153 and represents the size of the output vocabulary
    model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer), dropout=args.dec_dropout)

    # pack all models to pass to decode_forward function
    all_models = (model_input_emb, model_output_emb, model_enc, model_dec)
    # Create optimizers for every model
    inp_emb_optim = torch.optim.Adam(model_input_emb.parameters(), 1e-3)
    out_emb_optim = torch.optim.Adam(model_output_emb.parameters(), 1e-3)
    enc_optim = torch.optim.Adam(model_enc.parameters(), 1e-3)
    dec_optim = torch.optim.Adam(model_dec.parameters(), 1e-3)

    criterion = torch.nn.NLLLoss()

    # Iterate through epochs
    for epoch in range(1, args.epochs + 1):
        global total_sentences
        global exact
        total_sentences = 0.0
        exact = 0.0
        print("Epoch ", epoch)
        total_loss = 0.0
        # Loop over all examples in training data
        for pair_idx in range(len(train_data)):
            # extract data from train_data
            # Zero gradients
            inp_emb_optim.zero_grad()
            out_emb_optim.zero_grad()
            enc_optim.zero_grad()
            dec_optim.zero_grad()

            # Forward Pass
            loss = decode_forward(train_data, all_models, pair_idx, criterion, args)
            total_loss += loss

            # Backpropogation
            loss.backward()

            # Optimizer step
            inp_emb_optim.step()
            out_emb_optim.step()
            enc_optim.step()
            dec_optim.step()

        print("Total loss is {}".format(total_loss))
        if args.copy:
            print("{}% correct on copy task".format(100*float(exact/total_sentences)))

    if args.copy:
        print("Done with copy task, exiting before evaluation")
        exit()

    parser = Seq2SeqSemanticParser(model_dec, model_enc, model_input_emb, model_output_emb, output_indexer, args)
    return parser

def decode_forward(train_data, all_models, pair_idx, criterion,  args):
    global exact
    global total_sentences
    loss = 0.0
    (model_input_emb, model_output_emb, model_enc, model_dec) = all_models
    # in_seq is 2D size [batch size x sentence length] tensor
    in_seq = torch.as_tensor(train_data[pair_idx].x_indexed).unsqueeze(0)
    # in_len is 1D size [sentence length] tensor
    in_len = torch.as_tensor(len(train_data[pair_idx].x_indexed)).view(1)

    if args.copy:
        out_seq = torch.as_tensor(train_data[pair_idx].x_indexed).view(-1)
        gold, pred = [], []
    else:
        out_seq = torch.as_tensor(train_data[pair_idx].y_indexed).view(-1)
    # Run encoder with embedding here:
    (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = encode_input_for_decoder(
                                                                            in_seq, in_len, model_input_emb, model_enc)

    # Set up first inputs to decoder
    dec_input = torch.as_tensor(SOS_POS).unsqueeze(0).unsqueeze(0)
    #
    dec_hidden = enc_final_states_reshaped

    # Step through each word in the output sequence, feeding into the decoder
    for out_idx in range(len(out_seq)):
        # decode_ouput embeds dec_input, then passes it and dec_hidden to the decoder model
        # hid_out is tuple, each element is 3D tensor w/ size [1 x 1 x hidden_size]
        # dec_out is 3D tensor w/ size [1, 1, output vocab size = 153]
        dec_out, dec_hidden = decode_output(dec_input, dec_hidden, model_dec, model_output_emb)
        # print(dec_out)
        # Determine predicted index and its value
        pred_val, pred_idx = dec_out.topk(1)

        # calculate loss from decoder output and expected value
        loss += criterion(dec_out, out_seq[out_idx].unsqueeze(0))

        # Use teacher forcing to input correct word at next decoder step
        dec_input = out_seq[out_idx].unsqueeze(0).unsqueeze(0)
        if args.copy:
            gold.append(int(out_seq[out_idx]))
            pred.append(int(pred_idx))

        if int(pred_idx) == EOS_POS and not args.copy:
            break

    if args.copy:
        total_sentences += 1
        if gold == pred:
            print("Gold: {}\nPred: {}".format(gold, pred))
            exact += 1

    return loss

def decode_output(dec_input, dec_hidden, model_dec, model_output_emb):
    # Returns input vector with 3rd dimension of size 100, so if input is [1 x 1], output is [1 x 1 x 100]
    embedded = model_output_emb(dec_input)
    # return logsoftmax over decoder output, and hidden state tuple, both 3D Tensors
    dec_out, hid_out = model_dec(embedded, dec_hidden)

    return dec_out, hid_out

# Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
# every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
# executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
# example with a valid denotation (if you've provided more than one).
def evaluate(test_data, decoder, example_freq=50, print_output=True, outfile=None):
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    for i, ex in enumerate(test_data):
        if i % example_freq == 0:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % selected_derivs[i].y_toks)
        # Compute accuracy metrics
        y_pred = ' '.join(selected_derivs[i].y_toks)
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(selected_derivs[i].y_toks, ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if denotation_correct[i]:
            num_denotation_match += 1
    if print_output:
        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()


def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)

    if args.debug:
        train_data_indexed = train_data_indexed[:20]

    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    elif args.copy:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, input_indexer, args)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=True, outfile="geo_test_output.tsv")


