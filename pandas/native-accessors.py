import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option('display.max_rows', 5)
print(reviews)

print('\n')
print(reviews['country'])

print('\n')
print(reviews['country'][0])

#Ejercicios
#1
desc = reviews['description']


