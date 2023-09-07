import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
print(reviews.groupby('points').points.count())

print('\n')
print(reviews.groupby('points').price.min())

print('\n')
print(reviews.groupby('winery').apply(lambda df: df.title.iloc[0]))

print('\n')
print(reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()]))

print('\n')
print(reviews.groupby(['country']).price.agg([len, min, max]))

#Ejercicios
#1
reviews_written = reviews.groupby('taster_twitter_handle').size()

#2
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()

#3
price_extremes = reviews.groupby('variety').price.agg([min, max])

#4
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

#5
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()

#6
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)