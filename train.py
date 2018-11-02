import torch

from lf_evaluator import GeoqueryDomain
import parsers
from models import EmbeddingLayer, RNNEncoder, RNNDecoder, AttnDecoder
from copy import copy, deepcopy
from recombination import recombine
from manage_data import maybe_add_feature

import random

max_denotation = 0.0


PAD_POS = 0
SOS_POS = 1
UNK_POS = 1
EOS_POS = 2

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


def train_model_encdec(train_data, dev_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    global max_denotation

    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    dev_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    # len(output_indexer) is 153 and represents the size of the output vocabulary
    if args.attn:
        model_dec = AttnDecoder(args.output_dim, args.hidden_size, len(output_indexer), args, dropout=args.dec_dropout)
    else:
        model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer), dropout=args.dec_dropout)

    # pack all models to pass to decode_forward function
    all_models = (model_input_emb, model_output_emb, model_enc, model_dec)
    # Create optimizers for every model
    inp_emb_optim = torch.optim.Adam(model_input_emb.parameters(), args.lr)
    out_emb_optim = torch.optim.Adam(model_output_emb.parameters(), args.lr)
    enc_optim = torch.optim.Adam(model_enc.parameters(), args.lr)
    dec_optim = torch.optim.Adam(model_dec.parameters(), args.lr)

    criterion = torch.nn.NLLLoss()

    # Iterate through epochs
    for epoch in range(1, args.epochs + 1):
        global total_sentences
        global exact
        total_sentences = 0.0
        exact = 0.0

        model_output_emb.train()
        model_input_emb.train()
        model_enc.train()
        model_dec.train()

        print("Epoch ", epoch)
        with open(args.eval_file, "a") as f:
            f.write("Epoch {}\n".format(epoch))

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
            if args.attn:
                loss = attn_forward(train_data, all_models, pair_idx, criterion, args)
            else:
                loss = decode_forward(train_data, all_models, pair_idx, criterion, args)
            total_loss += loss

            # Backpropogation
            loss.backward()

            # Optimizer step
            inp_emb_optim.step()
            out_emb_optim.step()
            enc_optim.step()
            dec_optim.step()

        with open(args.eval_file, "a") as f:
            f.write("Total loss is {}\n".format(total_loss))

        print("Total loss is {}".format(total_loss))

        if args.attn:
            parser = parsers.AttnParser(model_dec, model_enc, model_input_emb, model_output_emb, output_indexer, args)
        else:
            parser = parsers.Seq2SeqSemanticParser(model_dec, model_enc, model_input_emb, model_output_emb, output_indexer, args)

        if args.copy:
            print("{}% correct on copy task".format(100*float(exact/total_sentences)))
        else:
            pass
            # evaluate(dev_data, parser, args, print_output=True, outfile="geo_test_output.tsv")
            denotation = float(evaluate(dev_data, parser, args, print_output=True))
            if denotataion > max_denotation:
                max_parser = parser
                max_denotation = denotation

    if args.copy:
        print("Done with copy task, exiting before evaluation")
        exit()

    try:
        return max_parser
    except:
        return parser

def attn_forward(train_data, all_models, pair_idx, criterion, args):
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
    # 3D hidden and cell states from encoder,
    dec_hidden = enc_final_states_reshaped

    # Step through each word in the output sequence, feeding into the decoder
    for out_idx in range(len(out_seq)):
        # decode_ouput embeds dec_input, then passes it and dec_hidden to the decoder model
        # hid_out is tuple, each element is 3D tensor w/ size [1 x 1 x hidden_size]
        # dec_out is 3D tensor w/ size [1, 1, output vocab size = 153]
        dec_out, dec_hidden = attn_output(dec_input, dec_hidden, enc_output_each_word, model_dec, model_output_emb)
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
            # print("Gold: {}\nPred: {}".format(gold, pred))
            exact += 1

    return loss

def attn_output(attn_input, attn_hidden, enc_outputs, model_attn, model_output_emb):
    # First we embed the decoder input
    embedded = model_output_emb.forward(attn_input)

    # Then we pass dec_input, dec_hidden, and enc_outputs into the attention decoder
    attn_out, attn_hidden = model_attn.forward(embedded, attn_hidden, enc_outputs)
    return attn_out, attn_hidden


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
        # print(out_seq[out_idx].unsqueeze(0))

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
            # print("Gold: {}\nPred: {}".format(gold, pred))
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
def evaluate(test_data, decoder, args, example_freq=50, print_output=True, outfile=None):
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
        with open(args.eval_file, "a") as f:
            f.write("Exact logical form matches: %s\n" % (render_ratio(num_exact_match, len(test_data))))
            f.write("Token-level accuracy: %s\n" % (render_ratio(num_tokens_correct, total_tokens)))
            f.write("Denotation matches: %s\n" % (render_ratio(num_denotation_match, len(test_data))))

        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
    # Writes to the output file if needed
    if outfile is not None:
        print("PRINTING OUTFILE NOW!!!")
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()

    return render_ratio(num_denotation_match, len(test_data))


def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)

def train_recombination(train_data, dev_data, input_indexer, output_indexer, args):
    global max_denotation

    maybe_add_feature([], input_indexer, True, "CITYID")
    maybe_add_feature([], input_indexer, True, "CITYSTATEID")
    maybe_add_feature([], output_indexer, True, "CITYID")
    maybe_add_feature([], output_indexer, True, "CITYSTATEID")

    # Add state placeholders to indexers
    maybe_add_feature([], input_indexer, True, "STATEID")
    maybe_add_feature([], output_indexer, True, "STATEID")

    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    # train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    # dev_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    ratios = [args.abs_ent_ratio/2, args.abs_ent_ratio/2, args.concat_ratio]
    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    # len(output_indexer) is 153 and represents the size of the output vocabulary
    if args.attn:
        model_dec = AttnDecoder(args.output_dim, args.hidden_size, len(output_indexer), args, dropout=args.dec_dropout)
    else:
        model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer), dropout=args.dec_dropout)

    # pack all models to pass to decode_forward function
    all_models = (model_input_emb, model_output_emb, model_enc, model_dec)
    # Create optimizers for every model
    inp_emb_optim = torch.optim.Adam(model_input_emb.parameters(), args.lr)
    out_emb_optim = torch.optim.Adam(model_output_emb.parameters(), args.lr)
    enc_optim = torch.optim.Adam(model_enc.parameters(), args.lr)
    dec_optim = torch.optim.Adam(model_dec.parameters(), args.lr)

    criterion = torch.nn.NLLLoss()

    # Iterate through epochs
    for epoch in range(1, args.epochs + 1):
        train_data_recomb = deepcopy(train_data)
        # Add the recombination data to the training set
        train_data_recomb.extend(recombine(train_data, input_indexer, output_indexer, args.recomb_size, args, ratios=ratios))
        random.shuffle(train_data_recomb)

        max_out_len = max([len(ex.y_indexed) for ex in train_data_recomb])
        global total_sentences
        global exact
        total_sentences = 0.0
        exact = 0.0

        model_output_emb.train()
        model_input_emb.train()
        model_enc.train()
        model_dec.train()

        print("Epoch ", epoch)
        with open(args.eval_file, "a") as f:
            f.write("Epoch {}\n".format(epoch))

        total_loss = 0.0
        # Loop over all examples in training data
        for pair_idx in range(len(train_data_recomb)):
            # extract data from train_data
            # Zero gradients
            inp_emb_optim.zero_grad()
            out_emb_optim.zero_grad()
            enc_optim.zero_grad()
            dec_optim.zero_grad()

            # Forward Pass
            if args.attn:
                if epoch==1 and pair_idx == 0:
                    print("Running Attention Model")
                loss = attn_forward(train_data_recomb, all_models, pair_idx, criterion, args)
            else:
                if epoch==1 and pair_idx == 0:
                    print("Running Base Model")
                loss = decode_forward(train_data_recomb, all_models, pair_idx, criterion, args)
            total_loss += loss

            # Backpropogation
            loss.backward()

            # Optimizer step
            inp_emb_optim.step()
            out_emb_optim.step()
            enc_optim.step()
            dec_optim.step()

        with open(args.eval_file, "a") as f:
            f.write("Total loss is {}\n".format(total_loss))

        print("Total loss is {}".format(total_loss))

        if args.attn:
            parser = parsers.AttnParser(model_dec, model_enc, model_input_emb, model_output_emb, output_indexer, args, max_output_len = max_out_len)
        else:
            parser = parsers.Seq2SeqSemanticParser(model_dec, model_enc, model_input_emb, model_output_emb, output_indexer, args, max_output_len=max_out_len)

        if args.copy:
            print("{}% correct on copy task".format(100*float(exact/total_sentences)))
        else:
            # pass
            denotation = float(evaluate(dev_data, parser, args, print_output=True))
            if denotation > max_denotation:
                max_parser = parser
                max_denotation = denotation


    if args.copy:
        print("Done with copy task, exiting before evaluation")
        exit()

    try:
        return max_parser
    except:
        return parser
