import itertools
from os import walk
import re
from lib.python.pairs_to_filename import pairs_to_filename
from lib.python.match_dictionaries import match_dictionaries
from lib.python.insert_in_dict import insert_in_dict


def find_files(pairs):

    # Combine input values
    comb = list(itertools.product(*pairs.values()))
    inputs = [dict(zip(pairs.keys(), comb[i])) for i in range(len(comb))]

    for i in range(len(inputs)):
        dic = inputs[i]
        files = next(walk(dic['set'] + dic['fol']))[2]
        found = [dict(re.findall('([a-zA-Z]+)-([a-zA-Z0-9]+)', files[i])) for i in range(len(files))]

        for j in range(len(found)):
            found[j] = insert_in_dict(found[j], 'set', dic['set'], 0)
            found[j] = insert_in_dict(found[j], 'fol', dic['fol'], 1)

        # Convert values of strings of int to int
        for j in range(len(found)):
            for key in found[j].keys():
                if found[j][key].isdigit():
                    found[j][key] = int(found[j][key])

        # Order found pairs
        for key in reversed(found[0].keys()):
            found = sorted(found, key=lambda d: d[key])

        # Order requested pairs
        for key in reversed(inputs[0].keys()):
            inputs = sorted(inputs, key=lambda d: d[key])

        # Convert found-run to input-run to match CUIDADO:
        # converted = move_run(copy.deepcopy(found), pairs)

        # Match found to input CUIDADO:
        # ids, matched = match_dictionaries(converted, inputs)
        ids, matched = match_dictionaries(found, inputs)

        # Select from found files (+ add filename)
        selected = []
        for j in range(len(ids)):
            if ids[j]:
                dic = found[j]
                dic['filename'] = pairs_to_filename(dic)
                selected.append(dic)

        return selected

