# You must be able to load your data before you can start your machine
# learning project. The most common format for machine learning data is CSV
# files. There are a number of ways to load a CSV file in Python. In this lesson
# you will learn three ways that you can use to load your CSV data in Python:
#   1. Load CSV Files with the Python Standard Library.
#   2. Load CSV Files with NumPy.
#   3. Load CSV Files with Pandas.

# 1. Loading CSV Data

# Common Format and MIME Type for Comma-Separated Values 1
# (CSV) Files  https://tools.ietf.org/html/rfc4180

# 􏰀 A. File Header.
# you should explicitly specify whether or not your CSV file had a file
# header when loading your data.

# 2. Comments
# Comments in a CSV file are indicated by a hash (#) at the start of a line.
# If you have comments in your file, depending on the method used to load your
# data, you may need to indicate whether or not to expect comments and the
# character to expect to signify a comment line.

# 3. Delimiter
# The standard delimiter that separates values in fields is the comma (,)
# character. Your file could use a different delimiter like tab or white
# space in which case you must specify it explicitly.

# 4. Quotes
# Sometimes field values can have spaces. In these CSV files the values are
# often quoted. The default quote character is the double quotation marks
# character. Other characters can be used, and you must specify the quote
# character used in your file.

#   B. Pima Indians Dataset
# The Pima Indians dataset is used to demonstrate data loading in this lesson.
# It will also be used in many of the lessons to come. This dataset describes
# the medical records for Pima Indians and whether or not each patient will
# have an onset of diabetes within five years.
# As such it is a classification problem. It is a good dataset for
# demonstration because all of the input attributes are numeric and the
# output variable to be predicted is binary (0 or 1). The data is freely
# available from the UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

#   C. Load CSV Files with the Python Standard Library
# The Python API provides the module CSV and the function reader() that can be
# used to load CSV files. Once loaded, you can convert the CSV data to a
# NumPy array and use it for machine learning.
#
# For example, you can download3 the Pima Indians dataset into your local
# directory with the filename pima-indians-diabetes.data.csv. All fields in this
# dataset are numeric and there is no header line.

# %%
# Load CSV Using Python Standard Library
import csv

import numpy as np

filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename) # raw_data = open(filename, 'rt') read and text
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')
print(data.shape)

#   D. Load CSV Files with NumPy
# You can load your CSV data using NumPy and the numpy.loadtxt() function. This
# function assumes no header row and all data has the same format. The example
# below assumes that the file pima-indians-diabetes.data.csv is in your current
# working directory.
# %%
# # Load CSV using NumPy
import numpy as np
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename) # raw_data = open(filename, 'rt') read and text
data = np.loadtxt(raw_data, delimiter=",")
print(data.shape)

# %%
# # This example can be modified to load the same dataset directly from a URL as
# follows:
import numpy as np
import urllib.request

url = 'http://nrvis.com/data/mldata/pima-indians-diabetes.csv'
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)

#   E. Load CSV Files with Pandas
# You can load your CSV data using Pandas and the pandas.read_csv() function.
# This function is very flexible and is perhaps my recommended approach for
# loading your machine learning data. The function returns a pandas.DataFrame
# that you can immediately start summarizing and plotting. The example below
# assumes that the pima-indians-diabetes.data.csv file is in the current
# working directory.

# %%
# # Load CSV using Pandas
import pandas as pd
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
print(dataframe.shape)