

def pairs_to_filename(dic):

    names = list(dic.keys())
    values = list(dic.values())

    fpath = ''
    for i in range(len(names)):
        name = names[i]
        value = values[i]

        if name in ['set', 'fol']:
            continue

        if type(value) == int:
            value = '{0:02d}'.format(value)
        fpath = fpath + name + '-' + value + '_'

    fpath = fpath[:-1]  # To remove last '_'
    return fpath