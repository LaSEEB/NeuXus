from fun.values_until_key import values_until_key


def move_run(found, pairs):

    pair_keys = list(pairs.keys())
    input_key_before_run = ''
    for i in range(len(pair_keys)):
        if pair_keys[i] == 'run':
            input_key_before_run = pair_keys[i-1]

    # Delete previous run
    for i in range(len(found)):
        if 'run' in found[i].keys():
            found[i].pop('run')

    run = 0
    previous_values = values_until_key(found[0], input_key_before_run)
    for i in range(len(found)):
        run += 1
        current_values = values_until_key(found[i], input_key_before_run)
        if current_values != previous_values:
            run = 1

        keys = list(found[i].keys())
        id_before_run = keys.index(input_key_before_run)
        keys.insert(id_before_run+1, 'run')
        values = list(found[i].values())
        values.insert(id_before_run+1, run)
        found[i] = dict(zip(keys,values))
        previous_values = current_values

    return found



