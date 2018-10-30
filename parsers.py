import torch

from manage_data import Derivation
from train import encode_input_for_decoder, SOS_POS, EOS_POS, decode_output

SOS_POS = 1
EOS_POS = 2

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

