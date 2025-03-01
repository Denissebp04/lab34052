import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/glopez21/ML-Data/main/diamonds.csv?authuser=0')

df.head(10)

df.info()

df.describe()

df.isnull().sum().sum()

sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

df['cut'].replace({'Fair':1}, inplace=True)
df['cut'].replace({'Good':2}, inplace=True)
df['cut'].replace({'Very Good':3}, inplace=True)
df['cut'].replace({'Premium':4}, inplace=True)
df['cut'].replace({'Ideal':5}, inplace=True)
df['clarity'].replace({'I1':0.125},inplace=True)
df['clarity'].replace({'SI2':0.25},inplace=True)
df['clarity'].replace({'SI1':0.375},inplace=True)
df['clarity'].replace({'VS2':0.5},inplace=True)
df['clarity'].replace({'VS1':0.625},inplace=True)
df['clarity'].replace({'VVS2':0.75}, inplace=True)
df['clarity'].replace({'VVS1':0.875},inplace=True)
df['clarity'].replace({'IF':1}, inplace=True)
df= df.drop(['color'], axis=1)
df.head(15)

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")

#cutvsprice
sns.boxplot(x='cut',y='price',data=df,hue='cut',palette='YlGnBu')
plt.legend(title='Cut', loc='upper right', labels=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])

#pricevscut
df.groupby('carat')['price'].mean().plot()
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Carat Vs Price')

