# # You must understand your data in order to get the best results. In this
# chapter you will discover 7 recipes that you can use in Python to better
# understand your machine learning data. After reading this lesson you will know
# how to:
# 1. Take a peek at your raw data.
# 2. Review the dimensions of your dataset.
# 3. Review the data types of attributes in your data.
# 4. Summarize the distribution of instances across classes in your dataset.
# 5. Summarize your data using descriptive statistics.
# 6. Understand the relationships in your data using correlations.
# 7. Review the skew of the distributions of each attribute.
# Each recipe is demonstrated by loading the Pima Indians Diabetes
# classification dataset from the UCI Machine Learning repository. Open your
# Python interactive environment and try each recipe out in turn. Let’s get
# started.

# 1. Peek at Your Data
# There is no substitute for looking at the raw data. Looking at the raw data
# can reveal insights that you cannot get any other way. It can also plant
# seeds that may later grow into ideas on how to better pre-process and handle
# the data for machine learning tasks. You can review the first 20 rows of your
# data using the head() function on the Pandas DataFrame.

# %%
# View first 20 rows
import pandas as pd

filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
peek = dataframe.head(20)
print(peek)

# 2. Dimensions of Your Data
# You must have a very good handle on how much data you have, both in terms of
# rows and columns.
# 􏰀- Too many rows and algorithms may take too long to train. Too few and
# perhaps you do not have enough data to train the algorithms.
# 􏰀- Too many features and some algorithms can be distracted or suffer poor
# performance due to the curse of dimensionality.
#
# You can review the shape and size of your dataset by printing the shape
# property on the Pandas DataFrame.

#%%
# Dimensions of your data
import pandas as pd

filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
shape = dataframe.shape
print(shape)

# 3. Data Type For Each Attribute
# The type of each attribute is important. Strings may need to be converted to
# floating point values or integers to represent categorical or ordinal
# values. You can get an idea of the types of attributes by peeking at the raw
# data, as above. You can also list the data types used by the DataFrame to
# characterize each attribute using the dtypes property.

# %%
# Data Types for Each Attribute
import pandas as pd
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
types = dataframe.dtypes
print(types)

# 4. Descriptive Statistics
# Descriptive statistics can give you great insight into the shape of each
# attribute. Often you can create more summaries than you have time to
# review. The describe() function on the Pandas DataFrame lists 8 statistical
# properties of each attribute. They are:
# 􏰀 1. Count.
# 􏰀 2. Mean.
# 􏰀 3. Standard Deviation.
#  4. Minimum Value.
# 􏰀 5. 25th Percentile.
# 􏰀 6. 50th Percentile (Median). 􏰀
#  7. 75th Percentile.
# 􏰀 8. Maximum Value.

# %%
# # Statistical Summary
import pandas as pd
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
description = dataframe.describe()
print(description)

# 5. Class Distribution (Classification Only)
# On classification problems you need to know how balanced the class values
# are. Highly imbalanced problems (a lot more observations for one class than
# another) are common and may need special handling in the data preparation
# stage of your project. You can quickly get an idea of the distribution of the
# class attribute in Pandas.

# %%
# Class Distribution
import pandas as pd
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
class_counts = dataframe.groupby('class').size() # how many class are there?
# and size
print(class_counts)

# 6. Correlations Between Attributes
# Correlation refers to the relationship between two variables and how they may
# or may not change together. The most common method for calculating
# correlation is Pearson’s Correlation Coefficient, that assumes a normal
# distribution of the attributes involved. A correlation of -1 or 1 shows a
# full negative or positive correlation respectively. Whereas a value of 0
# shows no correlation at all. Some machine learning algorithms like linear
# and logistic regression can suffer poor performance if there are highly
# correlated attributes in your dataset. As such, it is a good idea to review
# all of the pairwise correlations of the attributes in your dataset. You can
# use the corr() function on the Pandas DataFrame to calculate a correlation
# matrix.
#
# The matrix lists all attributes across the top and down the side, to give
# correlation between all pairs of attributes (twice, because the matrix is
# symmetrical). You can see the diagonal line through the matrix from the top
# left to bottom right corners of the matrix shows perfect correlation of each
# attribute with itself.

# %%
# Pairwise Pearson correlations
import pandas as pd
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
correlations = dataframe.corr(method='pearson')
print(correlations)

# 7. Skew of Univariate Distributions
# Skew refers to a distribution that is assumed Gaussian (normal or bell
# curve) that is shifted or squashed in one direction or another. Many machine
# learning algorithms assume a Gaussian distribution. Knowing that an
# attribute has a skew may allow you to perform data preparation to correct
# the skew and later improve the accuracy of your models. You can calculate the
# skew of each attribute using the skew() function on the Pandas DataFrame.
# The skew result show a positive (right) or negative (left) skew. Values
# closer to zero show less skew.

# %%
# Skew for each attribute
import pandas as pd
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
skew = data.skew()
print(skew)

# 8. Tips To Remember
# This section gives you some tips to remember when reviewing your data using
# summary statistics.
# 􏰀- Review the numbers. Generating the summary statistics is not enough. Take a
# moment to pause, read and really think about the numbers you are seeing.
# 􏰀- Ask why. Review your numbers and ask a lot of questions. How and why are
# you seeing specific numbers. Think about how the numbers relate to the
# problem domain in general and specific entities that observations relate to.
# 􏰀- Write down ideas. Write down your observations and ideas. Keep a small
# text file or note pad and jot down all of the ideas for how variables may
# relate, for what numbers mean, and ideas for techniques to try later. The
# things you write down now while the data is fresh will be very valuable later
# when you are trying to think up new things to try.