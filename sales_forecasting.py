import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Load CSVs (optimize by selecting only necessary columns)
orders = pd.read_csv('orders.csv', usecols=['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day'])
order_products = pd.read_csv('order_products__prior.csv')
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')

# Merge datasets
merged = order_products.merge(orders, on='order_id')
merged = merged.merge(products, on='product_id')
merged = merged.merge(aisles, on='aisle_id')
merged = merged.merge(departments, on='department_id')

### 1. Top-Selling Products
product_counts = merged['product_name'].value_counts().head(10)
print("\nTop 10 Most Sold Products:")
print(product_counts)

plt.figure(figsize=(10,5))
sns.barplot(x=product_counts.values, y=product_counts.index, palette='viridis')
plt.title("Top 10 Most Sold Products")
plt.xlabel("Units Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

### 2. Orders by Department
dept_counts = merged['department'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=dept_counts.values, y=dept_counts.index, palette='coolwarm')
plt.title("Top 10 Departments by Sales Volume")
plt.xlabel("Total Sales")
plt.ylabel("Department")
plt.tight_layout()
plt.show()

### 3. RFM Analysis & Clustering
last_order = orders.groupby('user_id')['order_number'].max().reset_index()
last_order.columns = ['user_id', 'LastOrder']
max_order = orders['order_number'].max()
last_order['Recency'] = max_order - last_order['LastOrder']

frequency = orders.groupby('user_id')['order_number'].count().reset_index()
frequency.columns = ['user_id', 'Frequency']

monetary = merged.groupby('user_id')['product_id'].count().reset_index()
monetary.columns = ['user_id', 'Monetary']

rfm = last_order[['user_id', 'Recency']].merge(frequency, on='user_id')
rfm = rfm.merge(monetary, on='user_id')
rfm_sample = rfm.sample(frac=0.05, random_state=42)

scaler = StandardScaler()
scaled = scaler.fit_transform(rfm_sample[['Recency', 'Frequency', 'Monetary']])
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_sample['Cluster'] = kmeans.fit_predict(scaled)

sns.pairplot(rfm_sample, hue='Cluster', palette='Set2', plot_kws={'alpha':0.5})
plt.suptitle("Customer Segmentation (RFM + KMeans)", y=1.02)
plt.show()

### 4. Forecasting Future Sales

from sklearn.linear_model import LinearRegression

sales_over_time = merged.groupby(['order_number', 'product_id']).size().reset_index(name='sales')
top_products = product_counts.head(3).index.tolist()
top_product_ids = products[products['product_name'].isin(top_products)]['product_id'].tolist()

for pid in top_product_ids:
    prod_sales = sales_over_time[sales_over_time['product_id'] == pid]
    prod_sales = prod_sales.sort_values('order_number')

    # Apply a moving average to smooth the trend
    prod_sales['sales_smooth'] = prod_sales['sales'].rolling(window=5, min_periods=1).mean()

    # Fit linear regression to smoothed values
    X = prod_sales[['order_number']]
    y = prod_sales['sales_smooth']
    model = LinearRegression()
    model.fit(X, y)

    # Forecast for next 30 order numbers
    future_range = np.arange(X['order_number'].max() + 1, X['order_number'].max() + 31)
    future_orders = pd.DataFrame({'order_number': future_range})
    future_predictions = model.predict(future_orders)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(prod_sales['order_number'], prod_sales['sales'], alpha=0.4, label='Original Sales')
    plt.plot(prod_sales['order_number'], prod_sales['sales_smooth'], label='Smoothed Sales', color='blue')
    plt.plot(future_orders['order_number'], future_predictions, '--', color='red', label='Forecast')
    
    product_name = products[products['product_id'] == pid]['product_name'].values[0]
    plt.title(f"Improved Forecast for Product: {product_name}")
    plt.xlabel("Order Number")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Choose a few high-selling products to forecast
top_products = product_counts.head(3).index.tolist()
top_product_ids = products[products['product_name'].isin(top_products)]['product_id'].tolist()

for pid in top_product_ids:
    prod_sales = sales_over_time[sales_over_time['product_id'] == pid]
    X = prod_sales[['order_number']]
    y = prod_sales['sales']
    model = LinearRegression()
    model.fit(X, y)
    future_orders = pd.DataFrame({'order_number': np.arange(X['order_number'].max()+1, X['order_number'].max()+6)})
    predictions = model.predict(future_orders)
    
    plt.figure(figsize=(8,4))
    plt.plot(X, y, label='Historical Sales')
    plt.plot(future_orders, predictions, '--', label='Forecast')
    product_name = products[products['product_id'] == pid]['product_name'].values[0]
    plt.title(f"Forecast for Product: {product_name}")
    plt.xlabel("Order Number")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()
    plt.show()

### 5. Best Sales Times (Day & Hour)
plt.figure(figsize=(10,4))
sns.countplot(x='order_dow', data=orders, palette='Set3')
plt.title("Order Distribution by Day of Week")
plt.xlabel("Day of Week (0 = Sunday)")
plt.ylabel("Total Orders")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x='order_hour_of_day', data=orders, palette='magma')
plt.title("Order Distribution by Hour")
plt.xlabel("Hour")
plt.ylabel("Total Orders")
plt.show()