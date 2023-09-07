import pandas as pd
pd.set_option('display.max_rows', 5)
import numpy as np
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
review_points_mean = reviews.points.mean()
print(reviews.points.map(lambda p: p - review_points_mean))

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

print('\n')
print(reviews.apply(remean_points, axis='columns'))

print('\n')
print(reviews.head(1))

print('\n')
review_points_mean = reviews.points.mean()
print(reviews.points - review_points_mean)

print('\n')
print(reviews.country + " - " + reviews.region_1)

#Ejercicios
#4
centered_price = reviews.price - reviews.price.mean()

#5
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

#6
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

#7
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')