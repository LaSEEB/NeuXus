

def insert_in_dict(dic, key, value, pos):

    keys = list(dic.keys())
    keys.insert(pos, key)
    values = list(dic.values())
    values.insert(pos, value)
    return dict(zip(keys, values))
