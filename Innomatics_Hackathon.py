#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("dataframe_.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna()


# In[7]:


plt.figure(figsize = (20,10))


# In[10]:


plt.scatter(df['input'],df['output'])
plt.show()


# In[11]:


df.describe()


# In[19]:


Q1 = df.output.quantile(0.25)
Q3 = df.output.quantile(0.75)
Q1,Q3


# In[20]:


IQR = Q3 -Q1
IQR


# In[21]:


lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit


# In[98]:


X= df['input']
X


# In[99]:


y = df['output']


# In[100]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[101]:



X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[102]:


X_train


# In[103]:


from sklearn.model_selection import cross_val_score
reg = LinearRegression()


# In[104]:


mse = cross_val_score(reg,X_train, y_train, scoring = 'neg_mean_squared_error', cv=10 )


# In[105]:


np.mean(mse)


# In[108]:


reg.fit(X_train,y_train)
reg_pred = reg.predict(X_test)


# In[58]:





# In[59]:


df


# In[60]:


df_no_outliers['input'].max()


# In[61]:


ML_reg.coef_


# In[62]:


ML_reg.intercept_


# In[63]:


y=0.4260*31.633331 -34.58


# In[64]:


y


# In[65]:


ML_reg.predict([[31.633331]])


# In[ ]:




