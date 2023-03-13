#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("survey_results_public.csv")


# In[56]:


df.head()


# In[57]:


df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
df = df.rename({"ConvertedComp": "Salary"}, axis=1)
df.head()


# In[58]:


df = df[df["Salary"].notnull()]
df.head()


# In[59]:


df.info()


# In[60]:


df = df.dropna()
df.isnull().sum()


# In[61]:


df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()


# In[62]:


df['Country'].value_counts()


# In[63]:


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


# In[64]:


country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()


# In[65]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[66]:


df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 10000]
df = df[df['Country'] != 'Other']


# In[67]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[68]:


df["YearsCodePro"].unique()


# In[69]:


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)


# In[70]:


df["EdLevel"].unique()


# In[71]:


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_education)


# In[72]:


df["EdLevel"].unique()


# In[73]:


from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df["EdLevel"].unique()
#le.classes_


# In[74]:


le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df["Country"].unique()


# In[75]:


X = df.drop("Salary", axis=1)
y = df["Salary"]


# In[39]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y.values)


# In[40]:


y_pred = linear_reg.predict(X)


# In[41]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))


# In[42]:


error


# In[43]:


from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, y.values)


# In[44]:


y_pred = dec_tree_reg.predict(X)


# In[45]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[46]:


from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)


# In[47]:


y_pred = random_forest_reg.predict(X)


# In[48]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[49]:


from sklearn.model_selection import GridSearchCV

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)


# In[50]:


regressor = gs.best_estimator_

regressor.fit(X, y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[76]:


X


# In[77]:


# country, edlevel, yearscode
X = np.array([["United States", 'Master’s degree', 15 ]])
X


# In[78]:


X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)
X


# In[79]:


y_pred = regressor.predict(X)
y_pred


# In[80]:


import pickle


# In[81]:


data = {"model": regressor, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[82]:


with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


# In[83]:


y_pred = regressor_loaded.predict(X)
y_pred


# In[ ]:





# In[ ]:




