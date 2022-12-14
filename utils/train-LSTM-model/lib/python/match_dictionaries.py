

def match_dictionaries(converted, inputs):

    # Select files from found
    ids = [False] * len(converted)
    selected = []
    for i in range(len(converted)):
        for j in range(len(inputs)):
            match = True
            for input_key in inputs[j].keys():
                if (input_key in converted[j].keys()) and (inputs[j][input_key] == converted[i][input_key]):
                    pass
                else:
                    match = False
                    break

            # for found_key in converted[i].keys():
            #     if found_key in inputs[j].keys():
            #         if converted[i][found_key] != inputs[j][found_key]:
            #             match = False
            #             break

            if match:
                selected.append(converted[i])
                ids[i] = True
                break

    return ids, selected