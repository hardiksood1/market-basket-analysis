#!/usr/bin/env python
# coding: utf-8

# # Market Basket Analysis (Apriori Algorithm)

# In[1]:


from scipy import sparse
from scipy.sparse import csr_matrix
import array as arr


# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings('ignore')
#from category_encoders import HashingEncoder


# In[3]:


data = pd.read_csv(r"C:\Users\hp\Desktop\projects for business analytics\market basket analysis\Assignment-1_Data.csv")


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data['CustomerID'].fillna(99999, inplace=True)

data["SumPrice"]=data["Quantity"]*data["Price"]


# In[7]:


best_selling_items = data.groupby(['Country', 'Itemname']).agg({'Quantity': 'sum'}).reset_index()
best_selling_items = best_selling_items.groupby('Country').apply(lambda x: x[x['Quantity'] == x['Quantity'].max()]).reset_index(drop=True)
best_selling_items.sort_values("Quantity",ascending=False).reset_index()


# In[8]:


total_sales_country = data.groupby(['Country']).agg({'SumPrice': 'sum'}).reset_index()

total_sales_country = total_sales_country.sort_values('SumPrice', ascending=False).reset_index(drop=True)
print(total_sales_country)


# In[9]:


plt.bar(total_sales_country['Country'],total_sales_country['SumPrice'])
plt.figsize = (20,25)
plt.yscale('log')
plt.xticks(rotation=90)
plt.show()


# In[10]:


data = data.rename(columns={'Itemname': 'ItemName'})
data['ItemName'] = data['ItemName'].str.lower()
data['CustomerID'] = data['CustomerID'].astype('int')


# In[11]:


transactions_original = data.groupby(['BillNo', 'Date'])['ItemName'].apply(lambda x: ', '.join(str(x) for item in data)).reset_index()

transactions_original.drop(columns=['BillNo', 'Date'], inplace=True)

transactions_original.head()


# In[12]:


transactions = transactions_original.copy()
transactions.head()
transactions['ItemName'] = transactions['ItemName'].astype(str)


# In[13]:


def transform_and_validate_transactions(transactions_data, original_data):
# Split 'ItemName' into individual items
    items_data = transactions_data['ItemName'].str.split(', ', expand=True)
    
    # Calculate the number of unique ['BillNo', 'Date'] combinations in original_data
    unique_transactions_count = original_data.drop_duplicates(subset=['BillNo', 'Date']).shape[0]
    # Validate the number of rows
    assert items_data.shape[0] == unique_transactions_count, \
        f"Row count mismatch! Expected: {unique_transactions_count}, Got: {items_data.shape[0]}"
    
    # Calculate the number of unique items across all transactions
    all_items = set()
    transactions['ItemName'] = transactions['ItemName'].fillna('').astype(str)
    original_data['ItemName'] = original_data['ItemName'].fillna('').astype(str)
    original_data['ItemName'].str.split(', ').apply(lambda items: all_items.update(items if items else []))
    max_product_counts = transactions['ItemName'].str.split(', ').apply(len).max()
    # Validate the number of columns
    assert items_data.shape[1] == max_product_counts, \
        f"Column count mismatch! Expected: {max_product_counts}, Got: {items_data.shape[1]}"
    
    # Return the transformed and validated DataFrame
    return items_data

# Apply the function and validate the results
transformed_and_validated_transactions = transform_and_validate_transactions(transactions_original, data)
print(transformed_and_validated_transactions.head())


# In[ ]:


def process_chunk(chunk):
    """Processes a chunk of the DataFrame."""
    chunk_encoded = pd.get_dummies(
        chunk, 
        prefix='', 
        prefix_sep='',
        sparse=True 
    ).groupby(level=0, axis=1).max()
    return csr_matrix(chunk_encoded) 

chunk_size = 10000 # Adjust based on available memory
sparse_chunks = []
for i in range(0, len(transformed_and_validated_transactions), chunk_size):
    chunk = transformed_and_validated_transactions[i: i + chunk_size]
    sparse_chunks.append(process_chunk(chunk))

data_encoded_sparse = vstack(sparse_chunks)


# In[ ]:


def mine_association_rules(transactions_data, min_support=0.01, min_confidence=0.5):
    
    # Convert items to boolean columns
    data_encoded_sparse = pd.get_dummies(transactions_data, prefix='', prefix_sep='').groupby(level=0, axis=1).max()
    
    # Perform association rule mining
    frequent_itemsets = apriori(data_encoded_sparce, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    return rules

# Now, use the function and display the association rules
rules = mine_association_rules(transformed_and_validated_transactions)
print("Association Rules:")
print(rules.head())


# In[ ]:





# In[ ]:




