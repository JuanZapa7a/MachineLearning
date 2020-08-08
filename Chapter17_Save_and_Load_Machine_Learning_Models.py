# Finding an accurate machine learning model is not the end of the project. In this chapter you will discover how to save and load your machine learning model in Python using scikit-learn. This allows you to save your model to file and load it later in order to make predictions. After completing this lesson you will know:
#   1. The importance of serializing models for reuse.
#   2. How to use pickle to serialize and deserialize machine learning models.
#   3. How to use Joblib to serialize and deserialize machine learning models.

# 1. Finalize Your Model with pickle
# Pickle is the standard way of serializing objects in Python. You can use the pickle  https://docs.python.org/2/library/pickle.html operation to serialize your machine learning algorithms and save the serialized format to a file. Later you can load this file to deserialize your model and use it to make new predictions. The example below demonstrates how you can train a logistic regression model on the Pima Indians onset of diabetes dataset, save the model to file and load it to make predictions on the unseen test set.

import pickle

# %%
# Save Model Using Pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=7)
print("Size of training set = train {}, labels {}".format(X_train.shape,
                                                          y_train.shape))
print("Size of test set = test {}, labels {}".format(X_test.shape,
                                                     y_test.shape))
# create a model
lgr_model = LogisticRegression(max_iter=1000)
lgr_X = lgr_model.fit(X_train, y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(lgr_model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print("Estimate of accuracy of the model on unseen data = {}".format(result))

# 2 Finalize Your Model with Joblib
# The Joblib https://pythonhosted.org/joblib/generated/joblib.dump.html library is part of the SciPy ecosystem and provides utilities for pipelining Python jobs. It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently . This can be useful for some machine learning algorithms that require a lot of parameters or store the entire dataset (e.g. k-Nearest Neighbors). The example below demonstrates how you can train a logistic regression model on the Pima Indians onset of diabetes dataset, save the model to file using Joblib and load it to make predictions on the unseen test set.

# %%
# Save Model Using joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=7)
print("Size of training set = train {}, labels {}".format(X_train.shape,
                                                          y_train.shape))
print("Size of test set = test {}, labels {}".format(X_test.shape,
                                                     y_test.shape))
# create a model
lgr_model = LogisticRegression(max_iter=1000)
lgr_X = lgr_model.fit(X_train, y_train)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(lgr_model, filename)
# some time later...
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print("Estimate of accuracy of the model on unseen data = {}".format(result))

# 3 Tips for Finalizing Your Model
# This section lists some important considerations when finalizing your machine learning models.

# - Python Version. Take note of the Python version. You almost certainly require the same major (and maybe minor) version of Python used to serialize the model when you later load it and deserialize it.

# - Library Versions. The version of all major libraries used in your machine learning project almost certainly need to be the same when deserializing a saved model. This is not limited to the version of NumPy and the version of scikit-learn.

# - Manual Serialization. You might like to manually output the parameters of your learned model so that you can use them directly in scikit-learn or another platform in the future. Often the techniques used internally by machine learning algorithms to make predictions are a lot simpler than those used to learn the parameters can may be easy to implement in custom code that you have control over.

# Take note of the version so that you can re-create the environment if for some reason you cannot reload your model on another machine or another platform at a later time.
