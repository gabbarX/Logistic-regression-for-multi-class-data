from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def sigmoid(x):
    fn = 1 / (1 + np.exp(-x))
    return fn


def LogisticRegression(X, y, alpha, iterations, X_test):
    # Extracting rows and feature number from the independent matrix
    rows, cols = X.shape

    # initialising a matrix of zeroes for storing the values of theta.
    theta = np.zeros(cols)

    # initialising the bias
    bias = 0

    # implementing gradient descent to get value of theta and bias
    for i in range(iterations):
        # compute linear combination
        linearComb = np.dot(X, theta) + bias
        # apply sigmoid function
        y_pred = sigmoid(linearComb)
        # compute gradients
        dw = (1 / rows) * np.dot(X.T, (y_pred - y))
        db = (1 / rows) * np.sum(y_pred - y)
        # update parameters
        theta = theta - alpha * dw
        bias = bias - alpha * db

    # compute linear combination for the final iteration
    linearComb = np.dot(X_test, theta) + bias
    y_pred = sigmoid(linearComb)

    # Convert the continuos values to binary values.
    res = []
    for i in y_pred:
        if i > 0.5:
            res.append(1)
        else:
            res.append(0)

    return np.array(res)


# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
# print(y)

# Split the dataset into training and testing sets

learning_rate = 0.01
iterations = 100000
randomState = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=randomState
)

# Train three separate logistic regression models
LR1 = LogisticRegression(X_train, (y_train == 0), learning_rate, iterations, X_test)
LR2 = LogisticRegression(X_train, (y_train == 1), learning_rate, iterations, X_test)
LR3 = LogisticRegression(X_train, (y_train == 2), learning_rate, iterations, X_test)

# Combine the y_pred into a single matrix
y_pred = np.column_stack((LR1, LR2, LR3))
# print(y_pred)

# Choose the class with the highest probability for each instance
predicted_classes = np.argmax(y_pred, axis=1)

# Compute the accuracy of the y_pred
accuracy = accuracy(y_true=y_test, y_pred=predicted_classes)
print("Accuracy: ", accuracy)
