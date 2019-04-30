import argparse
import random
import numpy as np

from manage_data import load_datasets, index_datasets, PAD_SYMBOL
from parsers import *
from recombination import *
from train import *


PAD_POS = 0
UNK_POS = 1
SOS_POS = 1
EOS_POS = 2

exact = 0.0
total_sentences = 0.0

# Todo: Look into ModuleList to make sure backpropogation is being performed over all the models (ModelList?)

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
    parser.add_argument('--epochs', type=int, default=5, help='num epochs to train for')
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
    parser.add_argument('--recomb', dest='recomb', default=False, action="store_true", help="Run recombination instead of training")
    parser.add_argument('--copy', dest='copy', default=False, action="store_true", help="Test that the decoder model can copy")
    parser.add_argument('--eval_file', type=str, default="eval_results.txt", help="Filepath to store evaluation results")
    parser.add_argument('--attn', dest='attn', default=False, action="store_true", help="Run decoder with attention enabled")
    parser.add_argument('--abs_ent_ratio', type=float, default=0.6, help="The ratio for abstract entities in recombination. ")
    parser.add_argument('--concat_ratio', type=float, default=0.4, help="The ratio for concatentation in recombination. ")
    parser.add_argument('--recomb_size', type=int, default=400, help="The amount of recombination examples to add to training set")
    parser.add_argument('--no_concat', dest="concat", default=True, action="store_false", help="turn off concatenation from recombination training")
    parser.add_argument('--no_abs_ent', dest="absent", default=True, action="store_false", help="turn off Abstract Entities from recombination training")
    args = parser.parse_args()
    return args


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

def main():
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)

    # for ex in train_data_indexed:
    #     print("x: {}, y: {}\nx_tok: {}\ny_tok: {}".format(ex.x, ex.y, ex.x_tok, ex.y_tok))
    if args.debug:
        train_data_indexed = train_data_indexed[:20]
        args.recomb_size = 20

    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    # print("Here are some examples post tokenization and indexing:")

    # print("\n\nSOS position is {}".format(output_indexer.get_index(SOS_SYMBOL)))
    # for i in range(0, min(len(train_data_indexed), 10)):
    #     print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder, args)
    elif args.copy:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, input_indexer, args)
    elif args.recomb:
        decoder = train_recombination(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    # evaluate(test_data_indexed, decoder, print_output=True, outfile="geo_test_output.tsv")
    evaluate(dev_data_indexed, decoder, args, print_output=True, outfile="geo_test_output.tsv")

if __name__ == '__main__':
    # main()
    def get_denotations(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Denotation" in line:
                    acc = line.split(" ")[-1].strip()
                    print(acc)


    if __name__ == '__main__':
        files = ["eval_attn_40ep.txt", "eval_base_40ep.txt", "eval_recomb_attn_1e-4lr.txt",
                 "eval_recomb_attn_40ep.txt", "eval_recomb_base_100ep_2.txt",
                 "eval_recomb_no-abs-ent_attn.txt", "eval_recomb_no-concat_attn.txt"]

        for file in files:
            print("FILE IS ", file)
            # print(os.getcwd())
            get_denotations(r"eval\{}".format(file))


