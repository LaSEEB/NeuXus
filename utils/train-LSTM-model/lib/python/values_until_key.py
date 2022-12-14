
def values_until_key(dic, key_before_run):

    values_before_run = []
    for key in dic.keys():
        values_before_run.append(dic[key])
        if key == key_before_run:
            break
    return values_before_run
