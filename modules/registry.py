# logging storage parameter

# number of line to log per chunk received
NB_LINE = 5
# number of chunk iter to log
NB_ITER = 3


def get_chunk_first_value(chunk):
    return str(chunk.iloc[-NB_LINE - 1:-1, :])
