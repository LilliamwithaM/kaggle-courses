import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option('display.max_rows', 5)
print(reviews[pd.isnull(reviews.country)])

print('\n')
print(reviews.region_2.fillna("Unknown"))

print('\n')
print(reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino"))

#Ejercicios
#3
missing_price_reviews = reviews[reviews.price.isnull()]
n_missing_prices = len(missing_price_reviews)
# Cute alternative solution: if we sum a boolean series, True is treated as 1 and False as 0
n_missing_prices = reviews.price.isnull().sum()
# or equivalently:
n_missing_prices = pd.isnull(reviews.price).sum()

#4
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)