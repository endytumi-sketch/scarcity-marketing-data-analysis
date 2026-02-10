import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
popmart = pd.read_csv('data/popmart_resale.csv')
nike = pd.read_csv('data/nike_resale.csv')

# POP MART scatter plot
plt.figure()
plt.scatter(popmart['retail_price'], popmart['resale_price'])
plt.title('POP MART Resale vs Retail')
plt.xlabel('Retail Price')
plt.ylabel('Resale Price')
plt.show()

# Nike scatter plot
plt.figure()
plt.scatter(nike['retail_price'], nike['resale_price'])
plt.title('Nike Resale vs Retail')
plt.xlabel('Retail Price')
plt.ylabel('Resale Price')
plt.show()
