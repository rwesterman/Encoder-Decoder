from manage_data import Example
from utils import maybe_add_feature
import numpy as np
from copy import copy, deepcopy

class State():
    def __init__(self, state_in, state_out, input_indexer, output_indexer):
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer

        self.state_in = state_in
        self.state_out = state_out
        self.state_in_idx = self.get_in_indexes(self.state_in)
        self.state_out_idx = self.get_out_indexes(self.state_out)

    def get_in_indexes(self, in_toks):
        return [self.input_indexer.get_index(x, False) for x in in_toks]

    def get_out_indexes(self, out_toks):
        return [self.output_indexer.get_index(x, False) for x in out_toks]

class City():
    def __init__(self, city_in, city_out, state_out, input_indexer, output_indexer, state_in = None):
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer

        self.city_in = city_in
        self.city_out = city_out
        self.state_out = state_out
        self.city_in_idx = self.get_in_indexes(self.city_in)
        self.city_out_idx = self.get_out_indexes(self.city_out)
        self.state_out_idx = self.get_out_indexes(self.state_out)

        if state_in:
            self.state_in = [state_in]
            self.state_in_idx = self.get_in_indexes(self.state_in)
        else:
            self.state_in = None
            self.state_in_idx = None

    def get_in_indexes(self, in_toks):
        in_idx = [self.input_indexer.get_index(x, False) for x in in_toks]
        # if -1 in in_idx:
        #     print("Indexer attribution error")
        return in_idx

    def get_out_indexes(self, out_toks):
        out_idx = [self.output_indexer.get_index(x, False) for x in out_toks]
        # if -1 in out_idx:
        #     print("Indexer attribution error")

        return out_idx

    def __key(self):
        full_list = self.city_in + self.city_out + self.state_out
        if self.state_in:
            full_list.extend(self.state_in)
        return " ".join(full_list)

    def __repr__(self):
        return "Cities: {} -> {}\n" \
               "States: {} -> {}\n".format(self.city_in, self.city_out, self.state_in, self.state_out)

    def __eq__(self, other):
        return type(self) == type(other) and self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

def recombine(train_data, input_indexer, output_indexer, total_examples, args, ratios = (0.3, 0.3, 0.4)):
    """

    :param train_data: list of Example objects that hold training data
    :param input_indexer:
    :param output_indexer:
    :param ratios: Triple of ratio for city/state/concat examples to add. Must sum to 1, else concat will get remaining amount
    :param total_examples: The total number of recombinant examples to add to the training set
    :return:
    """
    if not args.concat:
        CITY_RATIO = 0.5
        STATE_RATIO = 0.5
    elif not args.absent:
        CONCAT_RATIO = 1.0
    else:
        CITY_RATIO = ratios[0]
        STATE_RATIO = ratios[1]
        CONCAT_RATIO = ratios[2]

    # Abstract the entities from sentences in the training data
    city_exs, state_exs, cities, states = generalize_entities(train_data,input_indexer, output_indexer)
    # Pick a random subset of half the sentences from the training data
    recomb_examples = []
    if args.concat:
        recomb_examples.extend(concat2(train_data, int(total_examples*CONCAT_RATIO)))
    if args.absent:
        recomb_examples.extend(recomb_entities(city_exs, state_exs, list(cities), list(states),
                                               int(total_examples*CITY_RATIO), int(total_examples*STATE_RATIO)))

    # check_indexed_vs_tok(rec_ents_exs, input_indexer, output_indexer)
    # check_indexed_vs_tok(concat_exs, input_indexer, output_indexer)
    # check_indexed_vs_tok(recomb_examples, input_indexer, output_indexer)
    return recomb_examples

def concat2(train_data, output_number):
    concat_sents = []
    # pick random indices for first sentence and second sentence
    sents_1 = list(np.random.choice(len(train_data), output_number))
    sents_2 = list(np.random.choice(len(train_data), output_number))

    # Concatenate the sentences of the random choices
    for idx in range(len(sents_1)):
        new_x = train_data[sents_1[idx]].x + train_data[sents_2[idx]].x
        new_y = train_data[sents_1[idx]].y + train_data[sents_2[idx]].y
        new_x_tok = train_data[sents_1[idx]].x_tok + train_data[sents_2[idx]].x_tok
        # Removing _answer from second sentence
        new_y_tok = train_data[sents_1[idx]].y_tok + train_data[sents_2[idx]].y_tok[1:]
        new_x_indexed = train_data[sents_1[idx]].x_indexed + train_data[sents_2[idx]].x_indexed
        # For y_indexed combination, need to remove the <EOS> token from first sentence so it is handled properly
        # Todo: Try not removing _answer token from second sentence
        new_y_indexed = train_data[sents_1[idx]].y_indexed[:-1] + train_data[sents_2[idx]].y_indexed[1:]
        # new_y_indexed = train_data[sents_1[idx]].y_indexed[:-1] + train_data[sents_2[idx]].y_indexed
        concat_sents.append(Example(new_x, new_x_tok, new_x_indexed, new_y, new_y_tok, new_y_indexed))

    return concat_sents

def recomb_entities(city_exs, state_exs, cities, states, num_city_exs, num_state_exs):
    # Randomly sample a number of indices on which to recombine sentences
    rand_city_sents = np.random.choice(len(city_exs), num_city_exs)
    rand_state_sents = np.random.choice(len(state_exs), num_state_exs)
    out_examples = []

    out_examples.extend(recomb_cities(city_exs, cities, rand_city_sents))
    out_examples.extend(recomb_states(state_exs, states, rand_state_sents))
    return out_examples

def recomb_states(state_exs, states, rand_state_sents):
    out_examples = []

    for sidx in rand_state_sents:
        chosen_state = states[int(np.random.choice(len(states),1))]
        gst_ex = deepcopy(state_exs[sidx])

        stid_y_idx = gst_ex.y_tok.index("STATEID")
        stid_x_idx = gst_ex.x_tok.index("STATEID")

        gst_ex.y_tok = gst_ex.y_tok[:stid_y_idx] + chosen_state.state_out + gst_ex.y_tok[stid_y_idx+1:]
        gst_ex.y_indexed = gst_ex.y_indexed[:stid_y_idx] + chosen_state.state_out_idx + gst_ex.y_indexed[stid_y_idx+1:]

        gst_ex.x_tok = gst_ex.x_tok[:stid_x_idx] + chosen_state.state_in + gst_ex.x_tok[stid_x_idx+1:]
        gst_ex.x_indexed = gst_ex.x_indexed[:stid_x_idx] + chosen_state.state_in_idx + gst_ex.x_indexed[stid_x_idx+1:]

        gst_ex.x = " ".join(gst_ex.x_tok)
        gst_ex.y = " ".join(gst_ex.y_tok)

        out_examples.append(gst_ex)

    return out_examples

def recomb_cities(city_exs, cities, rand_city_sents):
    out_examples = []
    # for each random index in rand_city_sents, get the sentence, and recombine w/ random city
    for cidx in rand_city_sents:
        # Pick one random city and use to populate generic sentence. chosen_city is City object
        chosen_city = cities[int(np.random.choice(len(cities), 1))]
        # gen_city_ex is an Example object with a generic CITYID and CITYSTATEID
        gen_city_ex = deepcopy(city_exs[cidx])

        # First do operation on Y tokens/indices
        cityid_y_idx = gen_city_ex.y_tok.index("CITYID")
        stateid_y_idx = cityid_y_idx + 2

        # Repopulate example with chosen city
        gen_city_ex.y_tok = gen_city_ex.y_tok[:cityid_y_idx] + chosen_city.city_out + [gen_city_ex.y_tok[cityid_y_idx+1]]\
                            + chosen_city.state_out + gen_city_ex.y_tok[stateid_y_idx+1:]
        gen_city_ex.y_indexed = gen_city_ex.y_indexed[:cityid_y_idx] + chosen_city.city_out_idx + [gen_city_ex.y_indexed[cityid_y_idx+1]]\
                            + chosen_city.state_out_idx + gen_city_ex.y_indexed[stateid_y_idx+1:]

        # Then do operation on X tokens/indices
        cityid_x_idx = gen_city_ex.x_tok.index("CITYID")

        if chosen_city.state_in:
            gen_city_ex.x_tok = gen_city_ex.x_tok[:cityid_x_idx] + chosen_city.city_in\
                                + chosen_city.state_in + list(gen_city_ex.x_tok[cityid_x_idx + 1:])
            gen_city_ex.x_indexed = gen_city_ex.x_indexed[:cityid_x_idx] + chosen_city.city_in_idx\
                                    + chosen_city.state_in_idx + list(gen_city_ex.x_indexed[cityid_x_idx + 1:])

        # if output sequence doesn't have city,state combo, then don't include it in input sequence
        else:
            gen_city_ex.x_tok = gen_city_ex.x_tok[:cityid_x_idx] + chosen_city.city_in + list(
                gen_city_ex.x_tok[cityid_x_idx + 1:])
            gen_city_ex.x_indexed = gen_city_ex.x_indexed[:cityid_x_idx] + chosen_city.city_in_idx + list(
                gen_city_ex.x_indexed[cityid_x_idx + 1:])

        # Add this finalized example to the out_examples list
        gen_city_ex.x = " ".join(gen_city_ex.x_tok)
        gen_city_ex.y = " ".join(gen_city_ex.y_tok)
        out_examples.append(gen_city_ex)
    return out_examples

def generalize_entities(train_data, input_indexer, output_indexer):
    # Add city placeholders to indexers
    # maybe_add_feature([], input_indexer, True, "CITYID")
    # maybe_add_feature([], input_indexer, True, "CITYSTATEID")
    # maybe_add_feature([], output_indexer, True, "CITYID")
    # maybe_add_feature([], output_indexer, True, "CITYSTATEID")
    #
    # # Add state placeholders to indexers
    # maybe_add_feature([], input_indexer, True, "STATEID")
    # maybe_add_feature([], output_indexer, True, "STATEID")

    city_examples, state_examples = [], []
    cities, states = set(), set()

    for ex in train_data:
        example, city = generalize_cities(ex.y_tok, ex.x_tok, ex.y_indexed, ex.x_indexed, input_indexer, output_indexer)
        if example:
            city_examples.append(example)
            cities.add(city)
        else:
            example, state = generalize_states(ex.y_tok, ex.x_tok, ex.y_indexed, ex.x_indexed, input_indexer, output_indexer)
            if example:
                # Todo: Decide if I want to lump all my examples together or leave them separate for easier sorting later
                state_examples.append(example)
                states.add(state)

    return city_examples, state_examples, cities, states

def check_indexed_vs_tok(examples, input_indexer, output_indexer):
    """Test that examples are indexed properly"""
    for ex in examples:
        indexed_x = [input_indexer.get_index(x, False) for x in ex.x_tok]
        indexed_y = [output_indexer.get_index(y, False) for y in ex.y_tok]

        if indexed_x != ex.x_indexed:
            print("X's don't match!")
            print("From x_tok: {}\nFrom x_idx: {}".format(" ".join(ex.x_tok), " ".join(
                # [input_indexer.get_object(x) for x in ex.x_indexed])))
                [input_indexer.get_object(x) for x in ex.x_indexed])))
        if indexed_y != ex.y_indexed[:-1]:
            print("Y's don't match!")
            print("From y_tok: {}\nFrom y_idx: {}".format(" ".join(ex.y_tok), " ".join([output_indexer.get_object(y) for y in ex.y_indexed])))
            for idx in range(len(ex.y_indexed[:-1])):
                if indexed_y[idx] != ex.y_indexed[idx]:
                    print("At index {}, mismatch occurs: {} vs {}".format(idx,indexed_y[idx], ex.y_indexed[idx]))

def generalize_cities(y_tok, x_tok, y_indexed, x_indexed, input_indexer, output_indexer):
    # Get the indexes of the generalized placeholders so they can be added into indexed examples
    in_city_idx = [input_indexer.get_index("CITYID", False)]
    in_st_idx = [input_indexer.get_index("CITYSTATEID", False)]
    out_city_idx = [output_indexer.get_index("CITYID", False)]
    out_st_idx = [output_indexer.get_index("CITYSTATEID", False)]


    # Check if output contains a city id
    if "_cityid" in y_tok:
        # Get the index of the _cityid token, and make a pointer to walk through city and state
        city_pointer = y_tok.index("_cityid")
        # open_paren_idx will hold the index of the open parentheses before the city name
        open_paren_idx = city_pointer + 1
        # Loop until you hit the first comma, this is the end range of the city id
        while y_tok[city_pointer] != ",":
            city_pointer += 1
        # When the loop stops, we have the index of the first comma. Save this as comma_idx
        comma_idx = city_pointer
        # Now we know that the tokens between comma_idx and open_paren_idx are necessarily part of a city name of variable length
        while y_tok[city_pointer] != ")":
            city_pointer += 1
        #Now city_pointer is on the index of the close parentheses of the city id
        close_paren_idx = city_pointer

        gen_y_tok = y_tok.copy()
        gen_y_indexed = y_indexed.copy()

        city_y_id = gen_y_tok[open_paren_idx+1: comma_idx]
        st_y_id = gen_y_tok[comma_idx + 1: close_paren_idx]

        # populate the city id and state id positions in y_tok with placeholders so they can
        # later be substituted in
        gen_y_tok = gen_y_tok[:open_paren_idx + 1] + ['CITYID']\
                    + [gen_y_tok[comma_idx]] + ['CITYSTATEID'] + gen_y_tok[close_paren_idx:]

        gen_y_indexed = gen_y_indexed[:open_paren_idx + 1] + out_city_idx + [gen_y_indexed[comma_idx]]\
                         + out_st_idx + gen_y_indexed[close_paren_idx:]

        # Now handle the x input sequence in a similar way
        gen_x_tok = x_tok.copy()
        gen_x_indexed = x_indexed.copy()

        if len(city_y_id) > 1:
            # if the city id is longer than one token, we change how we search
            # first, ignore the "'" tokens on either side of the city name
            city1 = city_y_id[1]

            # Then, search for index of first section of city name in x_tok sentence
            city_pointer = x_tok.index(city1)   # We reuse city_pointer here
            city_start_idx = city_pointer
            city_x_id = x_tok[city_pointer: city_pointer + 2]
            # we increment city_pointer so that it points to the state position
            city_pointer += 2
            city_end_idx = city_pointer

            # Todo: check that indexes are correct here - convert indexed to tokens and print
            gen_x_tok = gen_x_tok[:city_start_idx] + ["CITYID"] + gen_x_tok[city_end_idx:]
            gen_x_indexed = gen_x_indexed[:city_start_idx] + in_city_idx + gen_x_indexed[city_end_idx:]

        else:
            city_pointer = x_tok.index(city_y_id[0])
            # explicityly make this a list, because it is only one element
            # city_x_id = [x_tok[city_pointer: city_pointer+1]]
            city_x_id = x_tok[city_pointer: city_pointer+1]
            # we increment city_pointer so that it points to the state position
            gen_x_tok = gen_x_tok[:city_pointer] + ["CITYID"] + gen_x_tok[city_pointer + 1:]
            gen_x_indexed = gen_x_indexed[:city_pointer] + in_city_idx + gen_x_indexed[city_pointer + 1:]

            city_pointer += 1

        # Check if there is a state id in the x input
        if st_y_id != ["_"]:
            # if the state name is multiple words
            st_x_id = x_tok[city_pointer]
            # gen_x_tok = gen_x_tok[:city_pointer] + ["CITYSTATEID"] + gen_x_tok[city_pointer+1:]
            gen_x_tok = gen_x_tok[:city_pointer] + gen_x_tok[city_pointer+1:]
            # gen_x_indexed = gen_x_indexed[:city_pointer] + in_st_idx + gen_x_indexed[city_pointer+1:]
            gen_x_indexed = gen_x_indexed[:city_pointer] + gen_x_indexed[city_pointer+1:]
        else:
            st_x_id = None

        x = " ".join(gen_x_tok)
        y = " ".join(gen_y_tok)
        # generalized_examples.append(Example(x,gen_x_tok, gen_x_indexed, y, gen_y_tok, gen_y_indexed))

        # if there is a city in the example, return the new example sentence and the new city object
        return Example(x,gen_x_tok, gen_x_indexed, y, gen_y_tok, gen_y_indexed), City(city_x_id, city_y_id, st_y_id, input_indexer, output_indexer, state_in=st_x_id)

    else:
        return None, None

def generalize_states(y_tok, x_tok, y_indexed, x_indexed, input_indexer, output_indexer):
    in_state_idx = [input_indexer.get_index("STATEID", False)]
    out_state_idx = [output_indexer.get_index("STATEID", False)]

    if "_stateid" in y_tok:
        # Get index of "_stateid" token. We know the next token is "(", so set that index as well
        state_pointer = y_tok.index("_stateid")
        state_pointer += 1
        open_paren_idx = state_pointer

        # find the index of the closing paren
        while y_tok[state_pointer] != ")":
            state_pointer += 1

        close_paren_idx = state_pointer
        state_y_id = list(y_tok[open_paren_idx + 1: close_paren_idx])

        gen_y_tok = y_tok[:open_paren_idx+1] + ["STATEID"] + y_tok[close_paren_idx:]
        gen_y_indexed = y_indexed[:open_paren_idx+1] + out_state_idx + y_indexed[close_paren_idx:]
        # Manipulate the input side as well
        if len(state_y_id) > 1:
            st1 = state_y_id[1]

            state_pointer = x_tok.index(st1)
            state_start_idx = state_pointer
            state_end_idx = state_pointer + 2
        else:
            state_pointer = x_tok.index(state_y_id[0])
            state_start_idx = state_pointer
            state_end_idx = state_pointer + 1

        state_x_id = x_tok[state_start_idx: state_end_idx]
        gen_x_tok = x_tok[:state_start_idx] + ["STATEID"] + x_tok[state_end_idx:]
        gen_x_indexed = x_indexed[:state_start_idx] + in_state_idx + x_indexed[state_end_idx:]

        x = " ".join(gen_x_tok)
        y = " ".join(gen_y_tok)

        return Example(x, gen_x_tok, gen_x_indexed, y, gen_y_tok, gen_y_indexed), State(state_x_id, state_y_id, input_indexer, output_indexer)

    else:
        return None, None


