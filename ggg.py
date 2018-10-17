
from pandas import read_csv
from datetime import datetime
import numpy as np
from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot
import pandas as pd

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# val = [1,2,3,4,5,6]

#    var1(t-1)  var2(t-1)  var3(t-1)  var1(t)  var2(t)  var3(t)
# 0        NaN        NaN        NaN       61       68       64
# 1       61.0       68.0       64.0       60       54       72
# 2       60.0       54.0       72.0       32       56        3

df = pd.DataFrame(np.random.randint(0,100,size=(10, 3)), columns=list('ABC'))

print(df)

val = series_to_supervised(df, 1, 1, dropnan=False)

print(val)