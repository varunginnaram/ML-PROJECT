import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset (for demonstration, replace this with your dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers with best parameters
perceptron = Perceptron(alpha=0.0038454011884736248, max_iter=570)
logistic_regression = LogisticRegression(C=0.017787658410143285, solver='newton-cg')
k_neighbors = KNeighborsClassifier(algorithm='auto', n_neighbors=7)
svc = SVC(C=4.329450186421157, gamma=0.030122914019804194, kernel='poly')
decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=8)
random_forest = RandomForestClassifier(max_depth=3, min_samples_split=11, n_estimators=69)

# Train the classifiers
perceptron.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)
k_neighbors.fit(X_train, y_train)
svc.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Evaluate the classifiers (optional)
perceptron_score = perceptron.score(X_test, y_test)
logistic_regression_score = logistic_regression.score(X_test, y_test)
k_neighbors_score = k_neighbors.score(X_test, y_test)
svc_score = svc.score(X_test, y_test)
decision_tree_score = decision_tree.score(X_test, y_test)
random_forest_score = random_forest.score(X_test, y_test)

# Print the evaluation scores (optional)
print("Perceptron Test Accuracy:", perceptron_score)
print("Logistic Regression Test Accuracy:", logistic_regression_score)
print("K Neighbors Test Accuracy:", k_neighbors_score)
print("SVC Test Accuracy:", svc_score)
print("Decision Tree Test Accuracy:", decision_tree_score)
print("Random Forest Test Accuracy:", random_forest_score)
