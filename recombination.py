from manage_data import Example
from utils import maybe_add_feature

class City():
    def __init__(self, city_in, city_out, state_out, state_in = None):
        self.city_in = city_in
        self.city_out = city_out
        self.state_out = state_out

        if state_in:
            self.state_in = state_in
        else:
            self.state_in = None

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
    pass


def generalize_cities(train_data, input_indexer, output_indexer):
    # Add generalized words to the indexers, get single element lists that have the new indices in them
    in_city_idx, out_city_idx, in_st_idx, out_st_idx = [], [], [], []
    maybe_add_feature(in_city_idx, input_indexer, True, "CITYID")
    maybe_add_feature(in_st_idx, input_indexer, True, "CITYSTATEID")
    maybe_add_feature(out_city_idx, output_indexer, True, "CITYID")
    maybe_add_feature(out_st_idx, output_indexer, True, "CITYSTATEID")

    generalized_examples= []
    cities = set()

    for ex in train_data:
        y_tok = ex.y_tok
        x_tok = ex.x_tok
        y_indexed = ex.y_indexed
        x_indexed = ex.x_indexed
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

            gen_y_indexed = gen_y_indexed[:open_paren_idx + 1] + in_city_idx + [gen_y_indexed[comma_idx]]\
                             + in_st_idx + gen_y_indexed[close_paren_idx:]

            # Now handle the x input sequence in a similar way
            gen_x_tok = x_tok.copy()
            gen_x_indexed = x_indexed.copy()

            if len(city_y_id) > 1:
                # if the city id is longer than one token, we change how we search
                # first, ignore the "'" tokens on either side of the city name
                city1 = city_y_id[1]
                city2 = city_y_id[2]

                # Then, search for index of first section of city name in x_tok sentence
                city_pointer = x_tok.index(city1)   # We reuse city_pointer here
                city_start_idx = city_pointer
                city_x_id = x_tok[city_pointer: city_pointer + 2]
                # we increment city_pointer so that it points to the state position
                city_pointer += 2
                city_end_idx = city_pointer

                gen_x_tok = gen_x_tok[:city_start_idx] + ["CITYID"] + gen_x_tok[city_end_idx:]
                gen_x_indexed = gen_x_indexed[:city_start_idx] + out_city_idx + gen_x_indexed[city_end_idx:]

            else:
                city_pointer = x_tok.index(city_y_id[0])
                city_x_id = x_tok[city_pointer: city_pointer+1]
                # we increment city_pointer so that it points to the state position
                gen_x_tok = gen_x_tok[:city_pointer] + ["CITYID"] + gen_x_tok[city_pointer + 1:]
                gen_x_indexed = gen_x_indexed[:city_pointer] + out_city_idx + gen_x_indexed[city_pointer + 1:]

                city_pointer += 1

            # Check if there is a state id in the x input
            if st_y_id != ["_"]:
                # if the state name is multiple words
                st_x_id = x_tok[city_pointer]
                gen_x_tok = gen_x_tok[:city_pointer] + ["CITYSTATEID"] + gen_x_tok[city_pointer+1:]
                gen_x_indexed = gen_x_indexed[:city_pointer] + out_st_idx + gen_x_indexed[city_pointer+1:]
            else:
                st_x_id = None

            print("Modified sequences:\nInput: {}\nOutput: {}".format(gen_x_tok, gen_y_tok))
            x = " ".join(gen_x_tok)
            y = " ".join(gen_y_tok)
            generalized_examples.append(Example(x,gen_x_tok, gen_x_indexed, y, gen_y_tok, gen_y_indexed))
            cities.add(City(city_x_id, city_y_id, st_y_id, st_x_id))
            # cities.append(City(city_x_id, city_y_id, st_y_id, st_x_id))

            example = Example(x, gen_x_tok, gen_x_indexed, y, gen_y_tok, gen_y_indexed)
            city = City(city_x_id, city_y_id, st_y_id, st_x_id)

            return Example(x,gen_x_tok, gen_x_indexed, y, gen_y_tok, gen_y_indexed),

    print("cities: {}".format(cities))
    # print("Gen examples ", generalized_examples)
    return generalized_examples, cities