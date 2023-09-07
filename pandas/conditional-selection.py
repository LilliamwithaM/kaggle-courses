import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option('display.max_rows', 5)
print(reviews.country == 'Italy')

print('\n')
print(reviews.loc[reviews.country == 'Italy'])

print('\n')
print(reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)])

print('\n')
print(reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)])

print('\n')
print(reviews.loc[reviews.country.isin(['Italy', 'France'])])

print('\n')
print(reviews.loc[reviews.price.notnull()])

#8
italian_wines = reviews[reviews.country == 'Italy']

#9
top_oceania_wines = reviews.loc[
    (reviews.country.isin(['Australia', 'New Zealand']))
    & (reviews.points >= 95)
]