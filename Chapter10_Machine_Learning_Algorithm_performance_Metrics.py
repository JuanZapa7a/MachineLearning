# # The metrics that you choose to evaluate your machine learning algorithms are very important. Choice of metrics influences how the performance of machine learning algorithms is measured and compared. They influence how you weight the importance of different characteristics in the results and your ultimate choice of which algorithm to choose.

# In this chapter you will discover how to select and use different machine learning performance metrics in Python with scikit-learn. Let’s get started.

# # 1. Algorithm Evaluation Metrics

# In this lesson, various different algorithm evaluation metrics are demonstrated for both classification and regression type machine learning problems. In each recipe, the dataset is downloaded directly from the UCI Machine Learning repository.

# 􏰀 For classification metrics, the Pima Indians onset of diabetes dataset is used as demon- stration. This is a binary classification problem where all of the input variables are numeric.

# 􏰀 For regression metrics, the Boston House Price dataset is used as demonstration. this is a regression problem where all of the input variables are also numeric.

# All recipes evaluate the same algorithms, Logistic Regression for classification and Linear Regression for the regression problems. A 10-fold cross validation test harness is used to demonstrate each metric, because this is the most likely scenario you will use when employing different algorithm evaluation metrics.

# A caveat in these recipes is the cross validation.cross val score function used to report the performance in each recipe. http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html

# It does allow the use of different scoring metrics that will be discussed, but all scores are reported so that they can be sorted in ascending order (largest score is best). Some evaluation metrics (like mean squared error) are naturally descending scores (the smallest score is best) and as such are reported as negative by the cross validation.cross val score() function. This is important to note, because some scores will be reported as negative that by definition can never be negative. I will remind you about this caveat as we work through the lesson.

# You can learn more about machine learning algorithm performance metrics supported by scikit-learn on the page Model evaluation: quantifying the quality of predictions. Let’s get on with the evaluation metrics.

# # 2. Classification Metrics
# Classification problems are perhaps the most common type of machine learning problem and as such there are a myriad of metrics that can be used to evaluate predictions for these problems. In this section we will review how to use the following metrics:
# 􏰀 - Classification Accuracy.
# 􏰀 - Logarithmic Loss.
# 􏰀 - Area Under ROC Curve. 􏰀
#  - Confusion Matrix.
# 􏰀 - Classification Report.

# # A. Classification Accuracy
# Classification accuracy is the number of correct predictions made as a ratio of all predictions made.

# This is the most common evaluation metric for classification problems, it is also the most misused.

# It is really only suitable when
#   1. there are an equal number of observations in each class (which is rarely the case) and
#   2. that all predictions and prediction errors are equally important, which is often not the case.

#   Below is an example of calculating classification accuracy.

# %%
# Cross Validation Classification Accuracy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lgr_model = LogisticRegression(max_iter=1000)
# Display accuracy
result = cross_val_score(lgr_model, X, y, cv=kfold, scoring='accuracy')
print("Accuracy: mean = {:.3f}%, std = {:.3f}%".format(result.mean()*100.0,result.std()*100.0))

# # B. Logarithmic Loss
# Logarithmic loss (or logloss) is a performance metric for evaluating the predictions of probabilities of membership to a given class.

# The scalar probability between 0 and 1 can be seen as a measure of confidence for a prediction by an algorithm.

# Predictions that are correct or incorrect are rewarded or punished proportionally to the confidence of the prediction. Below is an example of calculating logloss for Logistic regression predictions on the Pima Indians onset of diabetes dataset.

# %%
# # Cross Validation Classification LogLoss
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lgr_model = LogisticRegression(max_iter=1000)
# Display accuracy
result = cross_val_score(lgr_model, X, y, cv=kfold, scoring='neg_log_loss')
print("Logloss: mean = {:.3f}, std = {:.3f}".format(result.mean(),
                                                    result.std()))

# Smaller logloss is better with 0 representing a perfect logloss. As mentioned above, the measure is inverted to be ascending when using the cross val score() function.

# # C. Area Under ROC Curve
# Area under ROC Curve (or AUC for short) is a performance metric for binary classification problems.

# The AUC represents a model’s ability to discriminate between positive and negative classes.
# - An area of 1.0 represents a model that made all predictions perfectly.
# - An area of 0.5 represents a model that is as good as random.

# ROC can be broken down into sensitivity and specificity. A binary classification problem is really a trade-off between sensitivity and specificity.

# 􏰀 - Sensitivity is the true positive rate also called the recall. It is the number of instances from the positive (first) class that actually predicted correctly.
#  - Specificity is also called the true negative rate. Is the number of instances from the negative (second) class that were actually predicted correctly.

#  The example below provides a demonstration of calculating AUC.

# %%
# Cross Validation Classification ROC AUC
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lgr_model = LogisticRegression(max_iter=1000)
# Display accuracy
result = cross_val_score(lgr_model, X, y, cv=kfold, scoring='roc_auc')
print("Logloss: mean = {:.3f}, std = {:.3f}".format(result.mean(),
                                                    result.std()))

# You can see the AUC is relatively close to 1 and greater than 0.5, suggesting some skill in the predictions

# # D. Confusion Matrix
# The confusion matrix is a handy presentation of the accuracy of a model with two or more classes.

# The table presents predictions on the x-axis and accuracy outcomes on the y-axis. The cells of the table are the number of predictions made by a machine learning algorithm.

# For example, a machine learning algorithm can predict 0 or 1 and each prediction may actually have been a 0 or 1. Predictions for 0 that were actually 0 appear in the cell for prediction = 0 and actual = 0, whereas predictions for 0 that were actually 1 appear in the cell for prediction = 0 and actual = 1. And so on.

# Below is an example of calculating a confusion matrix for a set of predictions by a Logistic Regression on the Pima Indians onset of diabetes dataset.

# %%
# Cross Validation Classification Confusion Matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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
print("Info...\n")
print("Size of training set = train {}, labels {}".format(X_train.shape,
                                                          y_train.shape))
print("Size of test set = test {}, labels {}".format(X_test.shape,
                                                     y_test.shape))
# create a model
lgr_model = LogisticRegression(max_iter=1000)
lgr_X = lgr_model.fit(X_train, y_train)
predicted = lgr_model.predict(X_test)
# Display report
confusion = confusion_matrix(y_test, predicted)
print(confusion)

# Although the array is printed without headings, you can see that the majority of the predictions fall on the diagonal line of the matrix (which are correct predictions).

# # D. Classification Report
# The scikit-learn library provides a convenience report when working on classification problems to give you a quick idea of the accuracy of a model using a number of measures.

# The classification report() function displays the precision, recall, F1-score and support for each class. The example below demonstrates the report on the binary classification problem.
# %%
# Cross Validation Classification Report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
print("Info...\n")
print("Size of training set = train {}, labels {}".format(X_train.shape,
                                                          y_train.shape))
print("Size of test set = test {}, labels {}".format(X_test.shape,
                                                     y_test.shape))
# create a model
lgr_model = LogisticRegression(max_iter=1000)
lgr_X = lgr_model.fit(X_train, y_train)
predicted = lgr_model.predict(X_test)
# Display report
report = classification_report(y_test, predicted, target_names = ['False',
                                                                  'True'])
print(report)

# # 3. Regression Metrics
# In this section will review 3 of the most common metrics for evaluating predictions on regression machine learning problems:
# 􏰀 Mean Absolute Error. 􏰀
#  Mean Squared Error.
#  R^2

# # A. Mean Absolute Error.
# 􏰀The Mean Absolute Error (or MAE) is the sum of the absolute differences between predictions and actual values. It gives an idea of how wrong the predictions were. The measure gives an idea of the magnitude of the error, but no idea of the direction (e.g. over or under predicting). The example below demonstrates calculating mean absolute error on the Boston house price dataset.
# %%
# Cross Validation Regression MAE
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lnr_model = LinearRegression()
# Display accuracy
result = cross_val_score(lnr_model, X, y, cv=kfold,
                         scoring='neg_mean_absolute_error')
print("Mean Absolute Error: mean = {:.3f}, std = {:.3f}".format(result.mean(),
                                                    result.std()))
# A value of 0 indicates no error or perfect predictions. Like logloss, this metric is inverted by the cross val score() function.




# # B. Mean Squared Error.
# The Mean Squared Error (or MSE) is much like the mean absolute error in that it provides a gross idea of the magnitude of error. Taking the square root of the mean squared error converts the units back to the original units of the output variable and can be meaningful for description and presentation. This is called the Root Mean Squared Error (or RMSE). The example below provides a demonstration of calculating mean squared error.
# %%
# # Cross Validation Regression MSE
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lnr_model = LinearRegression()
# Display accuracy
result = cross_val_score(lnr_model, X, y, cv=kfold,
                         scoring='neg_mean_squared_error')
print("Mean Squared Error: mean = {:.3f}, std = {:.3f}".format(result.mean(),
                                                                result.std()))
# This metric too is inverted so that the results are increasing. Remember to take the absolute value before taking the square root if you are interested in calculating the RMSE.

# # C. R^2 Metric
# The R2 (or R Squared) metric provides an indication of the goodness of fit of a set of predictions to the actual values. In statistical literature this measure is called the coefficient of determination. This is a value between 0 and 1 for no-fit and perfect fit respectively. The example below provides a demonstration of calculating the mean R2 for a set of predictions.
# %%
# Cross Validation Regression R^2
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lnr_model = LinearRegression()
# Display accuracy
result = cross_val_score(lnr_model, X, y, cv=kfold,
                         scoring='r2')
print("R^2: mean = {:.3f}, std = {:.3f}".format(result.mean(),
                                                               result.std()))

# You can see the predictions have a poor fit to the actual values with a value closer to zero and less than 0.5.

