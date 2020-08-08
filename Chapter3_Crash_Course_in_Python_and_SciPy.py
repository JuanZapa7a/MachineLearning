# # 1. Python Crash Course

# When getting started in Python you need to know a few key details about the language syntax to be able to read and understand Python code. This includes:
# 􏰀 Assignment.
# 􏰀 Flow Control.
# 􏰀 Data Structures.
# 􏰀 Functions.
# We will cover each of these topics in turn with small standalone examples that you can type and run. Remember, whitespace has meaning in Python.

# 􏰀 A. Assignment.
# %%
# # Strings
data = 'hello world'
print(data[0])
print(data[-1])
print(len(data))
print(data)

# %%
# # Numbers
value = 123.1
print(value)
value = 10
print(value)

# %%
# # Boolean
a = True
b = False
print(a, b)

# %%
# # Multiple Assignment
a, b, c = 1, 2, 3
print(a, b, c)

# %%
# # No value
a = None
print(a)

#   B. Flow Control
# %%
# # If-Then-Else Conditional
value = 59
if value == 99: # notice the colon (:)
  print ('That is fast') # the tab intend the under block code
elif value > 200:
  print ('That is too fast')
else:
  print ('That is safe')

# %%
# # For-Loop
for i in range(10):
  print(i)

# %%
# # While-Loop
i=0
while i < 10:
  print (i)
  i += 1

#   C. Data Structures
# %%
# # Tuple -Tuples are read-only collections of items.
a = (1, 2, 3)
print(a)

# %%
# # List -Lists use the square bracket notation and can be index using array
# notation.
mylist = [1, 2, 3]
print(("Zeroth Value: %d") % mylist[0])
# or
print("Zeroth Value: {}".format(mylist[0]))
# add a new term in our case the number 4 in the 3 position
mylist.append(4)
print(("List Length: %d")  % len(mylist))
# or
print("List Length: {}".format(len(mylist)))
for value in mylist:
  print(value)

# %%
# # Dictionary -Dictionaries are mappings of names to values, like key-value
# pairs. Note the use of the curly bracket and colon notations when defining the
# dictionary.
mydict = {'a': 1, 'b': 2, 'c': 3}
print('A value: {}'.format(mydict['a']))
mydict['a'] = 11
print("A value: {}".format(mydict['a']))
print("Keys: {}".format(mydict.keys()))
print("Values: {}".format(mydict.values()))
for key in mydict.keys():
  print(mydict[key])

# %%
#   D. Functions
# # Functions -The biggest gotcha with Python is the whitespace. Ensure that
# you have an empty new line after indented code. The example below defines a
# new function to calculate the sum of two values and calls the function with
# two arguments.

# Sum function
def mysum(x,y):
  return x+y

# Test sum function
result = mysum(1, 3)
print(result)

# # 2. NumPy Crash Course
# # NumPy provides the foundation data structures and operations for SciPy.
# These are arrays (ndarrays) that are efficient to define and manipulate.

# %%
#   A. Create Array
# define an array
import numpy as np
mylist = [1, 2, 3]
mylist2 =[[1, 2, 3]]
myarray = np.array(mylist) # one vector with 3 objects
myarray2 = np.array(mylist2) # a matrix 1 row 3 cols
print(myarray)
print(myarray.shape)  # second dim is null, it is a vector
print(myarray2)
print(myarray2.shape) # second dim is no-null, it is a matrix
# %%
#   B. Access Data
# access values
import numpy as np
mylist = [[1, 2, 3], [3, 4, 5]]
myarray = np.array(mylist) # one array with two rows and three cols
print(myarray)
print(myarray.shape) # (rows,cols)
print("First row: {}".format(myarray[0])) # or myarray[0,:]
print("Last row: {}".format(myarray[-1])) # myarray[-1,:] -1 means last index
print("Specific row and col: {}".format(myarray[0, 2])) # row 1 col 3
print("Whole col: {}".format(myarray[:, 2])) # all rows from col 3

# %%
#   C. Arithmetic
# NumPy arrays can be used directly in arithmetic.
# arithmetic
import numpy as np
myarray1 = np.array([2, 2, 2])
myarray2 = np.array([3, 3, 3])
print("Addition: {}".format(myarray1 + myarray2)) # + elementwise
print("Multiplication: {}".format(myarray1 * myarray2)) # * elementwise

# # 3. Matplotlib Crash Course
# # Matplotlib can be used for creating plots and charts. The library is
# generally used as follows:
# 􏰀 Call a plotting function with some data (e.g. .plot()).
# 􏰀 Call many functions to setup the properties of the plot (e.g.labels,colors). 􏰀
#  Make the plot visible (e.g. .show()).

# %%
#   A. Line Plot
# The example below creates a simple line plot from one dimensional data.
# basic line plot
import matplotlib.pyplot as plt
import numpy as np
myarray = np.array([1, 2, 3])
plt.plot(myarray)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()

# %%
#   B. Scatter Plot
# Below is a simple example of creating a scatter plot from two dimensional
# data.
# basic scatter plot
import matplotlib.pyplot as plt
import numpy as np
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
plt.scatter(x,y)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()


# # 4. Pandas Crash Course
# Pandas provides data structures and functionality to quickly manipulate and
# analyze data. The key to understanding Pandas for machine learning is
# understanding the Series and DataFrame data structures.

# %%
#   A. Series
# A series is a one dimensional array (vector) where the rows and columns can
# be labeled.

# series
import numpy as np
import pandas as pd
myarray = np.array([1, 2, 3]) # [[1,2,3]] is a bidimensional array
print(myarray.shape)
rownames = ['a', 'b', 'c']
myseries = pd.Series(myarray, index=rownames)
print(myseries)

# You can access the data in a series like a NumPy array and like a dictionary,
# for example:
print(myseries[0])
print(myseries['a'])

# %%
#   B. DataFrame
# A data frame is a multi-dimensional array where the rows and the columns can
# be labeled.

# dataframe
import numpy as np
import pandas as pd
myarray = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3 array
print("Shape original array: {}".format(myarray.shape))
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pd.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)
print("Shape Dataframe: {}".format(mydataframe.shape))

# Data can be index using column names.

print("method 1:")
print("one column: {}".format(mydataframe['one']))
print("method 2:")
print("one column: {}".format(mydataframe.one))