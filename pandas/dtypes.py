import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option('display.max_rows', 5)
print(reviews.price.dtype)

print('\n')
print(reviews.dtypes)

print('\n')
print(reviews.points.astype('float64'))

print('\n')
print(reviews.index.dtype)

#Ejercicios
#1
dtype = reviews.points.dtype

#2
point_strings = reviews.points.astype(str)