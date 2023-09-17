#!/usr/bin/env python
# coding: utf-8

# <h1>POLYNOMIAL REGRESSION</h1>
# 
# In this exercise, we look at the relationship between 18 cars and their speeds.
# The data was obtained from a booth recorting car speeds.
# 
# The speeds of the cars were recoreded at a given time of the day (Hour)

# In[3]:


import matplotlib.pyplot as plt

x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

plt.scatter(x, y)
plt.xlabel('Hour')
plt.ylabel('Speed')
plt.show()


# Now, we ca deliberate the relationship between the speed and the reported hour of the 18 vehicles.

# In[7]:


import numpy
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100) # Specifies how the line witl be draw.
"""
The line is drawn starting at position 1 and ends at position 22
"""

plt.scatter(x, y)
plt.plot(myline, mymodel(myline)) ## Draws the polinomial line
plt.show()


# We can establish the relationhip of the variables using the R-squared method.
# R-squared values range from 0 to 1
# 
#  - 0 means no relationship between the variables.
#  - 1 means 100% relationship between the variables.
# 
# The Sklearn module computes variable relationships based on the R-squared method.

# In[8]:


from sklearn.metrics import r2_score


# In[9]:


mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))


# In[10]:


## A value of 0.94 is printed.
### This suggests a very positive relationship between speed and hour in our dataset. 


# Now, we can predict future values based on the given data/outcomes
# 
# For example, let's predict the speed of the car passing at around 17:00
# 
# 

# In[11]:


mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))


# In[12]:


mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

speed = mymodel(17)
print(speed)


# We find that the car would run at a speed of 88.87km/hr at 17:00

# <h1>BAD FIT</h1>
# 
# We can have a bad fit.
# Bad fit makes the polynomial algorithm inappropriate for prediction.

# In[13]:


"""
Let's create a hypothetical data of variables a and b
We want to determine the relationship between the two variables.
Also, we shall determine the degree of the relationship between the variables.
"""


a = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20, 26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
b = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10, 26, 34, 90, 33, 38, 20, 56, 2, 47, 17]

mymodel = numpy.poly1d(numpy.polyfit(a, b, 3))

myline = numpy.linspace(2, 95, 100)
"""
The line begins at 2, 95 and ends at 100
"""

plt.scatter(a, b)
plt.plot(myline, mymodel(myline))
plt.show()


# In[14]:


"""
Now, the scatter plot is evenly distributed all over the plot.
We should get a low r-squared value
"""

mymodel = numpy.poly1d(numpy.polyfit(a, b, 3))

print(r2_score(b, mymodel(a)))


# In[ ]:


# Prints an r2_score of 0.0099

