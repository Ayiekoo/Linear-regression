#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[5]:


df = "C:/Users/Ayieko/Desktop/python/archive/Ecommerce Customers.csv"
df = pd.read_csv(df)
print(df)


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


###### Exploratory data analysis #####
#### use seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[12]:


###### more time on site means more expenditure
sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = df)


# In[14]:


sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent', data = df)


# In[15]:


sns.jointplot(x = 'Time on App', y = 'Length of Membership', kind = 'hex', data = df)


# In[16]:


##### exploring the relationship across the dtaset###

#### use pairplot ###
sns.pairplot(df)


# In[17]:


###### length of memmebership #####
##### Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
 ##########

    


# In[19]:


sns.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent', data = df)


# In[20]:


######### TRAINING AND TESTING THE DATA ###### 
y = df['Yearly Amount Spent']


# In[21]:


x  =df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]


# In[24]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


lm = LinearRegression()


# In[29]:


#### Train/fit the linear model on the training data ###


# In[30]:


lm.fit(X_train, y_train)


# In[31]:


#### The coefficients ###
print('coefficients: \n', lm.coef_)


# In[35]:


####### PREDICTING TEST DATA #######
### use the lm.preduct() to predict off the x_test set of data #####
predictions = lm.predict(x_test)


# In[36]:


##### create a scatterplot of the real test values vs. predicted values

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[37]:


###### EVALUATING THE MODEL #####
#####** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**#####


# In[38]:


#### CALCULATE THE METRICS MANUALLY ####

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[39]:


####### RESIDUALS #####
sns.displot((y_test-predictions), bins = 50)


# In[54]:


coeffecients = pd.DataFrame(lm.coef_, x.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:


#### Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
##### Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
####### Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
######### Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

