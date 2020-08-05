import pandas as pd
import random as rd

INDEX = [i * 0.2 for i in range(20)]
COLUMN = ['a', 'b', 'c', 'd']


def mock_df(index, column):
    value = []
    for i, _ in enumerate(index):
        value.append([])
        for col in column:
            value[i].append(rd.uniform(-1000, 1000))
    return pd.DataFrame(value, index=index, columns=column)


def is_df_equal(df1, df2):
    for val1, val2 in zip(df1.values, df2.values):
        for v1, v2 in zip(val1, val2):
            if v1 != v2:
                return False
    return True


def simulate_loop_and_verify(port, node, self, column=COLUMN, index=INDEX):
    for i in range(5):
        df = mock_df(index, column)
        df_copy = df.copy()
        port.set_from_df(df)
        node.update()
        self.assertTrue(is_df_equal(df, df_copy))
        port.clear()
    for i in range(5):
        df1 = mock_df(index, column)
        df1_copy = df1.copy()
        df2 = mock_df(index, column)
        df2_copy = df2.copy()
        df3 = mock_df(index, column)
        df3_copy = df3.copy()
        port.set_from_df(df1)
        port.set_from_df(df2)
        port.set_from_df(df3)
        node.update()
        self.assertTrue(is_df_equal(df1, df1_copy))
        self.assertTrue(is_df_equal(df2, df2_copy))
        self.assertTrue(is_df_equal(df3, df3_copy))
        port.clear()
