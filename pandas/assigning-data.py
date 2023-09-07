import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option('display.max_rows', 5)
reviews['critic'] = 'everyone'
print(reviews['critic'])

print('\n')
reviews['index_backwards'] = range(len(reviews), 0, -1)
print(reviews['index_backwards'])
