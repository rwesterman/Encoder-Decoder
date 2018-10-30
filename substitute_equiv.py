import random
import numpy as np

# Todo: Incorporate opening geo_train.tsv, appending, and saving as new training file

# Todo: Output sequences are all surrounded by quotes " ". Why???


class SubbedSentence():
    def __init__(self, x_indexed, y_indexed, x_tok, y_tok):
        self.x_indexed = x_indexed
        self.x_tok = x_tok
        self.y_indexed = y_indexed
        self.y_tok = y_tok


class States():
    def __init__(self):
        self.state_in_tok = []
        self.state_out_tok = []
        self.state_in_idx = []
        self.state_out_idx = []

    def add_state(self, st_in_tok, st_out_tok, st_in_idx, st_out_idx):
        self.state_in_tok.append(st_in_tok)
        self.state_out_tok.append(st_out_tok)
        self.state_in_idx.append(st_in_idx)
        self.state_out_idx.append(st_out_idx)

    def get_all_state_idxs(self):
        """Returns two lists of state indexes. First list is input indexes, second is output indexes"""
        return self.state_in_idx, self.state_out_idx

    def get_by_st_in_token(self, st_in_token):
        st_in_token = st_in_token.lower()
        if st_in_token in self.state_in_tok:
            st_pos = self.state_in_tok.index(st_in_token)
            # return all data for this state
            return (self.state_in_tok[st_pos], self.state_out_tok[st_pos],
                    self.state_in_idx[st_pos], self.state_out_idx[st_pos])
        else:
            return (None, None, None, None)

    def get_by_st_out_token(self, st_out_token):
        st_out_token = st_out_token.lower()
        if st_out_token in self.state_out_tok:
            st_pos = self.state_out_tok.index(st_out_token)
            # return all data for this state
            return (self.state_in_tok[st_pos], self.state_out_tok[st_pos],
                    self.state_in_idx[st_pos], self.state_out_idx[st_pos])
        else:
            return (None, None, None, None)

    def get_by_st_in_idx(self, st_in_idx):
        if st_in_idx in self.state_in_idx:
            st_pos = self.state_in_idx.index(st_in_idx)
            # return all data for this state
            return (self.state_in_tok[st_pos], self.state_out_tok[st_pos],
                    self.state_in_idx[st_pos], self.state_out_idx[st_pos])
        else:
            return (None, None, None, None)

    def get_by_st_out_idx(self, st_out_idx):
        if st_out_idx in self.state_out_idx:
            st_pos = self.state_out_idx.index(st_out_idx)
            # return all data for this state
            return (self.state_in_tok[st_pos], self.state_out_tok[st_pos],
                    self.state_in_idx[st_pos], self.state_out_idx[st_pos])
        else:
            return (None, None, None, None)

    def check_state_exists(self, st_in_token = None, st_in_idx = None):
        # If searching by token, return position of token if it exists
        if st_in_token:
            if st_in_token in self.state_in_tok:
                return self.state_in_tok.index(st_in_token)
            else: return None
        # if searching by index, return position of index if it exists
        elif st_in_idx:
            if st_in_idx in self.state_in_idx:
                return self.state_in_idx.index(st_in_idx)
            else: return None
        # If neither was searched, print a warning and return None
        else:
            print("check_state_exists must have either st_in_token or st_in_idx passed as argument")
            return None

    def get_k_other_states(self, st_in_idx, k=3):
        """Returns k number of other states to substitute into new sentence"""
        choices = np.random.choice(len(self.state_in_tok), k, replace = False)
        if st_in_idx:
            if st_in_idx in self.state_in_idx:
                st_pos = self.state_in_idx.index(st_in_idx)
                st_in_tokens = [self.state_in_tok[idx] for idx in choices if idx != st_pos]
                st_out_tokens = [self.state_out_tok[idx] for idx in choices if idx != st_pos]
                st_in_idxs = [self.state_in_idx[idx] for idx in choices if idx != st_pos]
                st_out_idxs = [self.state_out_idx[idx] for idx in choices if idx != st_pos]
                return (st_in_tokens, st_out_tokens, st_in_idxs, st_out_idxs)
            else:
                return None

    def get_all_other_states(self, st_in_token = None, st_in_idx = None):
        """Determines if the checked state exists, then returns 4 lists (as a tuple) that contain all other states"""
        if st_in_token:
            if st_in_token in self.state_in_tok:
                st_pos = self.state_in_tok.index(st_in_token)
                st_in_tokens = [x for x in self.state_in_tok if x != self.state_in_tok[st_pos]]
                st_out_tokens = [x for x in self.state_out_tok if x != self.state_out_tok[st_pos]]
                st_in_idxs = [x for x in self.state_in_idx if x != self.state_in_idx[st_pos]]
                st_out_idxs = [x for x in self.state_out_idx if x != self.state_out_idx[st_pos]]
                return (st_in_tokens, st_out_tokens, st_in_idxs, st_out_idxs)
            else: return None
        # if searching by index, return position of index if it exists
        elif st_in_idx:
            if st_in_idx in self.state_in_idx:
                st_pos = self.state_in_idx.index(st_in_idx)
                st_in_tokens = [x for x in self.state_in_tok if x != self.state_in_tok[st_pos]]
                st_out_tokens = [x for x in self.state_out_tok if x != self.state_out_tok[st_pos]]
                st_in_idxs = [x for x in self.state_in_idx if x != self.state_in_idx[st_pos]]
                st_out_idxs = [x for x in self.state_out_idx if x != self.state_out_idx[st_pos]]
                return (st_in_tokens, st_out_tokens, st_in_idxs, st_out_idxs)
            else: return None
        # If neither was searched, print a warning and return None
        else:
            print("get_all_other_states must have either st_in_token or st_in_idx passed as argument")
            return None

    def check_x_indexed_for_state(self, indexed_sentence):
        """Checks if the x_indexed sentence (input sentence) contains a state and returns boolean"""
        return not set(indexed_sentence).isdisjoint(self.state_in_idx)


def get_states(input_indexer, output_indexer):

    state_list = "texas, colorado, nebraska, ohio, washington, montana, maryland, michigan, tennessee, california, " \
    "illinois, georgia, alabama, kansas, oregon, missouri, kentucky, alaska, wyoming, delaware, wisconsin, louisiana, " \
                 "pennsylvania, arkansas, oklahoma, utah, arizona".split(", ")

    states = States()
    for state in state_list:
        states.add_state(state, state, input_indexer.get_index(state), output_indexer.get_index(state))

    # return the states object that holds all the information
    return states
    # print(states.get_by_st_in_token("texas"))
    # for idx in range(len(other_st_in)):
    #     print(other_st_in[idx], other_st_out[idx], other_st_in_idx[idx], other_st_out_idx[idx])

def sub_equivs(train_data, input_indexer, output_indexer):
    states = get_states(input_indexer, output_indexer)
    # initialize these to zero so they can be
    new_sents = []
    for idx in range(len(train_data)):
        if "west virginia" in train_data[idx].x_tok:
            print(train_data[idx].y_tok)
            print(train_data[idx].y_indexed)

        st_in_tokens, st_out_tokens, st_in_idxs, st_out_idxs = 0, 0, 0, 0
        # This line checks whether
        if states.check_x_indexed_for_state(train_data[idx].x_indexed):
            # If the sentence contains a state, find out which state, and at what index
            match_idx_lst = find_matching_index(train_data[idx].x_indexed, states.state_in_idx)
            for st_idx, tr_in_idx in match_idx_lst:
                # Also get the output sequence index from training data. Needs to be done at token level because indices
                # are inconsistent between input and output indexers
                try:
                    tr_out_idx = train_data[idx].y_indexed.index(
                        output_indexer.get_index(states.state_in_tok[st_idx]))
                except ValueError as e:
                    print("State is mentioned in input tokens, but not output tokens. Skipping this instance")
                    break
                    # This returns all other states in token and index form for input and output
                (st_in_tokens, st_out_tokens, st_in_idxs, st_out_idxs) = states.get_k_other_states(
                                                                            st_in_idx=states.state_in_idx[st_idx])

                for state_pos in range(len(st_in_tokens)):
                    # Now replace a state in the in/out sequences with every other state, across both tokens and indices
                    new_x_indexed = train_data[idx].x_indexed.copy()
                    new_x_indexed[tr_in_idx] = st_in_idxs[state_pos]

                    new_y_indexed = train_data[idx].y_indexed.copy()
                    new_y_indexed[tr_out_idx] = st_out_idxs[state_pos]

                    new_x_toks = train_data[idx].x_tok.copy()
                    new_x_toks[tr_in_idx] = st_in_tokens[state_pos]

                    new_y_toks = train_data[idx].y_tok.copy()
                    new_y_toks[tr_out_idx] = st_out_tokens[state_pos]

                    # print(new_x_toks)
                    # newsent = SubbedSentence(new_x_indexed, new_y_indexed, new_x_toks, new_y_toks)
                    # print(newsent.x_tok)
                    new_sents.append(SubbedSentence(new_x_indexed, new_y_indexed, new_x_toks, new_y_toks))
                    # new_sents.append((new_x_toks, new_y_toks))
                    # new_sents.append(newsent)

    # print(new_sents)
    print_subs_to_file(new_sents, "Subbed_States.tsv")



def find_matching_index(list1, list2):
    """Finds all matching elements in two lists and outputs their indices in pairs"""
    inverse_index = {element: index for index, element in enumerate(list1)}

    return [(index, inverse_index[element])
            for index, element in enumerate(list2) if element in inverse_index]


def print_subs_to_file(sub_sent_list, filename):
    with open(filename, "w") as f:
        for sub in sub_sent_list:
            print(" ".join(sub.x_tok))
            f.write(" ".join(sub.x_tok))
            f.write("\t")
            outseq = " ".join(sub.y_tok)
            outseq = outseq.strip("\"")
            f.write(outseq)
            f.write("\n")