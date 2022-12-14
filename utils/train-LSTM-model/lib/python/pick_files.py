import itertools
import re
from fun.move_run import move_run
from fun.pairs_to_filename import pairs_to_filename
from fun.match_dictionaries import match_dictionaries
import copy


def pick_files(files, pairs):

    # Combine input values
    comb = list(itertools.product(*pairs.values()))
    inputs = [dict(zip(pairs.keys(), comb[i])) for i in range(len(comb))]
    found = [dict(re.findall('([a-zA-Z]+)-([a-zA-Z0-9]+)', files[i])) for i in range(len(files))]

    # Convert values of strings of int to int
    for i in range(len(found)):
        for key in found[i].keys():
            if found[i][key].isdigit():
                found[i][key] = int(found[i][key])

    # Order found pairs
    for key in reversed(found[0].keys()):
        found = sorted(found, key=lambda d: d[key])

    # Order requested pairs
    for key in reversed(inputs[0].keys()):
        inputs = sorted(inputs, key=lambda d: d[key])

    # Convert found-run to input-run to match
    converted = move_run(copy.deepcopy(found), pairs)

    # Match found to input
    ids, matched = match_dictionaries(converted, inputs)

    # Select from found files (+ add filename)
    selected = []
    for i in range(len(ids)):
        if ids[i]:
            dic = found[i]
            dic['filename'] = pairs_to_filename(dic)
            selected.append(dic)

    return selected