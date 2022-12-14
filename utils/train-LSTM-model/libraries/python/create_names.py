import itertools


def create_names(temp, *words):
    comb = list(itertools.product(*words))
    files = []
    for c in comb:
        files.append(temp.format(*c))
    return files
