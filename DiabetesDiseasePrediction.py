#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error # accuracy evaluation


# In[7]:


# Import Dataset
disease = datasets.load_diabetes()
#print(disease)
disease_X = disease.data[:, np.newaxis, 2]


# In[8]:


# Split data into Train and Test
disease_X_train = disease_X[:-30]
disease_X_test = disease_X[-20:]
disease_Y_train = disease.target[:-30]
disease_Y_test = disease.target[-20:]


# In[9]:


# Generate Model
reg = linear_model.LinearRegression()
reg.fit(disease_X_train,disease_Y_train)
y_predict = reg.predict(disease_X_test)
accuracy = mean_squared_error(disease_Y_test,y_predict)
print(accuracy)
weights = reg.coef_
intercept = reg.intercept_
print(weights)
print(intercept)


# In[11]:


# Plot Graph
plt.scatter(disease_X_test, disease_Y_test)
plt.plot(disease_X_test, y_predict)
plt.show()


# In[ ]:




