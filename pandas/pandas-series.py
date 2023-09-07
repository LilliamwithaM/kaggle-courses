import pandas as pd

print('\n')
serie1 = pd.Series([1, 2, 3, 4, 5])
print(serie1)

print('\n')
serie2 = pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
print(serie2)

#3
ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
