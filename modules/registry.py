# logging storage parameter
import pandas as pd

# number of line to log per chunk received
NB_LINE = 5
# number of chunk iter to log
NB_ITER = 3

pd.set_option("display.max_rows", 10)


def get_chunk_first_value(chunk):
    return chunk
