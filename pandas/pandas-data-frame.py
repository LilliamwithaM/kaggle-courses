import pandas as pd

dataframe1 = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
print(dataframe1)

print('\n')
dataframe2 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
print(dataframe2)

print('\n')
dataframe3= pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
            'Sue': ['Pretty good.', 'Bland.']},
            index=['Product A', 'Product B'])
print(dataframe3)

#Ejercicios
#1
fruits = pd.DataFrame({'Apples': [30], 'Bananas': [21]})

#2
fruit_sales = pd.DataFrame({'Apples': [35, 41], 
            'Bananas': [21, 34]},
            index=['2017 Sales', '2018 Sales'])

#5
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
animals.to_csv('cows_and_goats.csv')
