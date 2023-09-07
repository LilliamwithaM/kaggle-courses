import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option('display.max_rows', 5)
print(reviews.iloc[0])

print('\n')
print(reviews.iloc[:3, 0])

print('\n')
print(reviews.iloc[1:3, 0])

print('\n')
print(reviews.iloc[[0, 1, 2], 0])

print('\n')
print(reviews.iloc[-5:])

print('\n')
print(reviews.loc[0, 'country'])

print('\n')
print(reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']])

#2
first_description = reviews.description.iloc[0]

#3
first_row = reviews.iloc[0]

#4
first_descriptions = reviews.description.iloc[:10]

#5
indices = [1, 2, 3, 5, 8]
sample_reviews = reviews.iloc[indices]

#6
cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = reviews.loc[indices, cols]

#7
cols = ['country', 'variety']
df = reviews.loc[:99, cols]
