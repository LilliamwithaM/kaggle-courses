import pandas as pd
pd.set_option('display.max_rows', 5)
import numpy as np
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
print(reviews.points.describe())

print('\n')
print(reviews.points.mean())

print('\n')
print(reviews.taster_name.unique())

print('\n')
print(reviews.taster_name.value_counts())

#Ejercicios
#1
median_points = reviews.points.median()

#2
countries = reviews.country.unique()

#3
reviews_per_country = reviews.country.value_counts()