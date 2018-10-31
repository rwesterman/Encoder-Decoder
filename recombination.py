from manage_data import Example
from utils import maybe_add_feature

class State():
    def __init__(self, state_in, state_out, input_indexer, output_indexer):
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer

        self.state_in = state_in
        self.state_out = state_out
        self.state_in_idx = self.get_in_indexes(self.state_in)
        self.state_out_idx = self.get_out_indexes(self.state_out)

    def get_in_indexes(self, in_toks):
        return [self.input_indexer.get_index(x) for x in in_toks]

    def get_out_indexes(self, out_toks):
        return [self.output_indexer.get_index(x) for x in out_toks]

class City():
    def __init__(self, city_in, city_out, state_out, input_indexer, output_indexer, state_in = None):
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer

        self.city_in = city_in
        self.city_out = city_out
        self.state_out = state_out
        self.city_in_idx = self.get_in_indexes(city_in)
        self.city_out_idx = self.get_out_indexes(city_out)
        self.state_out_idx = self.get_out_indexes(state_out)

        if state_in:
            self.state_in = state_in
            self.state_in_idx = self.get_in_indexes(state_in)
        else:
            self.state_in = None
            self.state_in_idx = None

    def get_in_indexes(self, in_toks):
        return [self.input_indexer.get_index(x) for x in in_toks]

    def get_out_indexes(self, out_toks):
        return [self.output_indexer.get_index(x) for x in out_toks]

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


def find_matching_index(list1, list2):
    """Finds all matching elements in two lists and outputs their indices in pairs"""
    inverse_index = {element: index for index, element in enumerate(list1)}

    return [(index, inverse_index[element])
            for index, element in enumerate(list2) if element in inverse_index]

def generalize_entities(train_data, input_indexer, output_indexer):
    # Add city placeholders to indexers
    maybe_add_feature([], input_indexer, True, "CITYID")
    maybe_add_feature([], input_indexer, True, "CITYSTATEID")
    maybe_add_feature([], output_indexer, True, "CITYID")
    maybe_add_feature([], output_indexer, True, "CITYSTATEID")

    # Add state placeholders to indexers
    maybe_add_feature([], input_indexer, True, "STATEID")
    maybe_add_feature([], output_indexer, True, "STATEID")


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

    # check_indexed_vs_tok(examples, input_indexer, output_indexer)

def check_indexed_vs_tok(examples, input_indexer, output_indexer):
    """Test that examples are indexed properly"""
    for ex in examples:
        indexed_x = [input_indexer.get_index(x) for x in ex.x_tok]
        indexed_y = [output_indexer.get_index(y) for y in ex.y_tok]

        # if indexed_x != ex.x_indexed:
        #     print("X's don't match!")
        if indexed_y != ex.y_indexed[:-1]:
            print("Y's don't match!")
            print("From y_tok: {}\nFrom y_idx: {}".format(" ".join(ex.y_tok), " ".join([output_indexer.get_object(y) for y in ex.y_indexed])))
            for idx in range(len(ex.y_indexed[:-1])):
                if indexed_y[idx] != ex.y_indexed[idx]:
                    print("At index {}, mismatch occurs: {} vs {}".format(idx,indexed_y[idx], ex.y_indexed[idx]))

def generalize_cities(y_tok, x_tok, y_indexed, x_indexed, input_indexer, output_indexer):
    # Get the indexes of the generalized placeholders so they can be added into indexed examples
    in_city_idx = [input_indexer.get_index("CITYID")]
    in_st_idx = [input_indexer.get_index("CITYSTATEID")]
    out_city_idx = [output_indexer.get_index("CITYID")]
    out_st_idx = [output_indexer.get_index("CITYSTATEID")]


    # generalized_examples= []
    # cities = set()
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
            gen_x_tok = gen_x_tok[:city_pointer] + ["CITYSTATEID"] + gen_x_tok[city_pointer+1:]
            gen_x_indexed = gen_x_indexed[:city_pointer] + in_st_idx + gen_x_indexed[city_pointer+1:]
        else:
            st_x_id = None

        # print("Modified sequences:\nInput: {}\nOutput: {}".format(gen_x_tok, gen_y_tok))
        x = " ".join(gen_x_tok)
        y = " ".join(gen_y_tok)
        # generalized_examples.append(Example(x,gen_x_tok, gen_x_indexed, y, gen_y_tok, gen_y_indexed))

        # if there is a city in the example, return the new example sentence and the new city object
        return Example(x,gen_x_tok, gen_x_indexed, y, gen_y_tok, gen_y_indexed), City(city_x_id, city_y_id, st_y_id, input_indexer, output_indexer, state_in=st_x_id)

    else:
        return None, None

def generalize_states(y_tok, x_tok, y_indexed, x_indexed, input_indexer, output_indexer):
    in_state_idx = [input_indexer.get_index("STATEID")]
    out_state_idx = [output_indexer.get_index("STATEID")]

    if "_stateid" in y_tok:
        # Get index of "_stateid" token. We know the next token is "(", so set that index as well
        state_pointer = y_tok.index("_stateid")
        state_pointer += 1
        open_paren_idx = state_pointer

        # find the index of the closing paren
        while y_tok[state_pointer] != ")":
            state_pointer += 1

        close_paren_idx = state_pointer
        state_y_id = y_tok[open_paren_idx + 1: close_paren_idx]
        print(state_y_id)

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


