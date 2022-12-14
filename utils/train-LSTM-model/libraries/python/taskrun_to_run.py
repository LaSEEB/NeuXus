

def taskrun_to_run(inputs, found):

    # Order found pairs
    for key in reversed(found[0].keys()):
        found = sorted(found, key=lambda d: d[key])

    # Order requested pairs
    for key in reversed(inputs[0].keys()):
        inputs = sorted(inputs, key=lambda d: d[key])

    selected = []
    for i in range(len(found)):
        match = True
        found_keys = list(found[i].keys())
        for j in range(len(inputs)):
            input_keys = inputs[j].keys()
            for k in range(len(found_keys)):
                key = found_keys[k]
                if key in input_keys:

                    if key == 'run':
                        found_previous_key = found_keys[k-1]
                        input_previous_key = input_keys[k-1]


                    else:

                            if found[i][key] != inputs[j][key]:
                                match = False
                                break

            if match:
                selected.append(i)
                break







    # Turn found values strings of integers into integers
    for i in range(len(found)):
        for key in found[i].keys():
            if found[i][key].isdigit():
                found[i][key] = int(found[i][key])

    selected = []
    for i in range():
        pass

    lalal = 0
    last_sub = found[0].sub
    last_ses = found[0].ses
    last_mri = found[0].mri

    # for i in range(len(found)):
