import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load datasets
popmart = pd.read_csv('data/popmart_resale.csv')
nike = pd.read_csv('data/nike_resale.csv')

# ---------- POP MART ----------
plt.figure()
plt.scatter(popmart['retail_price'], popmart['resale_price'])

X = popmart[['retail_price']]
y = popmart['resale_price']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# sort x to draw a clean line
x_sorted = np.sort(popmart['retail_price'])
y_line = model.predict(x_sorted.reshape(-1, 1))

r2 = r2_score(y, y_pred)

plt.plot(x_sorted, y_line)
plt.title(f'POP MART Resale vs Retail (R²={r2:.2f})')
plt.xlabel('Retail Price')
plt.ylabel('Resale Price')

plt.savefig("popmart_scatter.png", dpi=300)
plt.show()

# ---------- NIKE ----------
plt.figure()
plt.scatter(nike['retail_price'], nike['resale_price'])

X = nike[['retail_price']]
y = nike['resale_price']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

x_sorted = np.sort(nike['retail_price'])
y_line = model.predict(x_sorted.reshape(-1, 1))

r2 = r2_score(y, y_pred)

plt.plot(x_sorted, y_line)
plt.title(f'Nike Resale vs Retail (R²={r2:.2f})')
plt.xlabel('Retail Price')
plt.ylabel('Resale Price')

plt.savefig("nike_scatter.png", dpi=300)
plt.show()


