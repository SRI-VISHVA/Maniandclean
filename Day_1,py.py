import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
# print(iris)

iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
print(iris_df.shape)
print(iris_df.head(5))
numarray = np.array(iris_df.sample(5)['sepal length (cm)'])
print(np.round(numarray.sum()))
example3 = pd.Series([0, np.nan, '', None])
print(example3.isnull)
example3 = example3.dropna()
print(example3)
example4 = pd.DataFrame([[1, np.nan, 7],
                         [2, 5, 8],
                         [np.nan, 6, 9]])
print(example4)
example4 = example4.dropna(axis='columns')
print(example4)
example4[3] = np.NaN
print(example4)
example4 = example4.dropna(axis='rows', thresh=0)
print(example4)
# ------------------------------------------------------------------------------------------------------------ #
example5 = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
print(example5)
example5 = example5.fillna(method='bfill')
print(example5)
example4 = example4.fillna(example4.mean())
print(example4)
# Removing duplicate data
example6 = pd.DataFrame({'letters': ['A', 'B'] * 2 + ['B'],
                         'numbers': [1, 2, 1, 3, 3]})
print(example6)
print(example6.duplicated())
example6 = example6.drop_duplicates('letters')
print(example6)
# Combining datasets: merge and join
df1 = pd.DataFrame({'employee': ['Gary', 'Stu', 'Mary', 'Sue'],
                    'group': ['Accounting', 'Marketing', 'Marketing', 'HR']})
df2 = pd.DataFrame({'employee': ['Mary', 'Stu', 'Gary', 'Sue'],
                    'hire_date': [2008, 2012, 2017, 2018]})
print(df1)
print(df2)
df3 = pd.merge(df1, df2)
print(df3)
df4 = pd.DataFrame({'group': ['Accounting', 'Marketing', 'HR'],
                    'supervisor': ['Carlos', 'Giada', 'Stephanie']})
print(df4)
print(pd.merge(df4, df3, on='group'))
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Marketing', 'Marketing', 'HR', 'HR'],
                    'core_skills': ['math', 'spreadsheets', 'writing', 'communication',
                                    'spreadsheets', 'organization']})
print(df5)
print(pd.merge(df4, df5, on='group'))
df6 = pd.DataFrame({'name': ['Gary', 'Stu', 'Mary', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(df6)
print(pd.merge(df1, df6, left_on='employee', right_on='name'))
df1a = df1.set_index('employee')
print(df1a)
df2a = df2.set_index('employee')
print(df2a)
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
df5 = pd.DataFrame({'group': ['Engineering', 'Marketing', 'Sales'],
                    'core_skills': ['math', 'writing', 'communication']})
print(df5)
print(pd.merge(df1, df5, on='group'))
print(pd.merge(df1, df5, on='group', how='outer'))
print(pd.merge(df1, df5, how='right'))
df7 = pd.DataFrame({'name': ['Gary', 'Stu', 'Mary', 'Sue'],
                    'rank': [1, 2, 3, 4]})
print(df7)
df8 = pd.DataFrame({'name': ['Gary', 'Stu', 'Mary', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(df8)
print(pd.merge(df7, df8, on='name'))
print(pd.merge(df7, df8, on='name', suffixes=('_left', '_right')))
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
con1 = np.concatenate([x, y, z], axis=0)
print(con1)
ser1 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
ser2 = pd.Series(['d', 'e', 'f'], index=[4, 5, 6])
print(pd.concat([ser1, ser2]))
df9 = pd.DataFrame({'A': ['a', 'c'],
                    'B': ['b', 'd']})
print(df9)
print(pd.concat([df9, df9], ignore_index=True))
print(pd.concat([df9, df9], axis=1))
df10 = pd.DataFrame({'A': ['a', 'd'],
                     'B': ['b', 'e'],
                     'C': ['c', 'f']})
df11 = pd.DataFrame({'B': ['u', 'x'],
                     'C': ['v', 'y'],
                     'D': ['w', 'z']})
print(pd.concat([df10, df11]))
print(pd.concat([df10, df11], join='inner'))
df12 = df9.append(df9)
print(df12)
df = pd.read_csv('housing_dataset.csv')
print(df.columns)
print(df.shape)
print(df.describe())
print(df['CRIM'].mean())
print(df.groupby(['AGE'])['MEDV'].mean())
df['AGE_50'] = df['AGE'].apply(lambda x: x > 50)
print(df.columns)
print(df['AGE_50'].value_counts())
groupby_twovar = df.groupby(['AGE_50', 'RAD', 'CHAS'])['MEDV'].mean()
print(groupby_twovar)
print(groupby_twovar.unstack())
print(df['CRIM'].nunique())
print(df['CRIM'].unique())
