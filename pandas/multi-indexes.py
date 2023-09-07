import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
print(countries_reviewed)

print('\n')
mi = countries_reviewed.index
print(type(mi))

print('\n')
print(countries_reviewed.reset_index())

