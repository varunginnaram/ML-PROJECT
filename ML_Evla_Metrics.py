import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load training and testing data
train_data = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")  # Replace with the path to your train dataset
test_data = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")    # Replace with the path to your test dataset

# Split train and test data into features and target
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']

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

# Predict using the classifiers
perceptron_pred = perceptron.predict(X_test)
logistic_regression_pred = logistic_regression.predict(X_test)
k_neighbors_pred = k_neighbors.predict(X_test)
svc_pred = svc.predict(X_test)
decision_tree_pred = decision_tree.predict(X_test)
random_forest_pred = random_forest.predict(X_test)

# Evaluate the classifiers
perceptron_accuracy = accuracy_score(y_test, perceptron_pred)
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_pred)
k_neighbors_accuracy = accuracy_score(y_test, k_neighbors_pred)
svc_accuracy = accuracy_score(y_test, svc_pred)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_pred)
random_forest_accuracy = accuracy_score(y_test, random_forest_pred)

perceptron_precision = precision_score(y_test, perceptron_pred, average='weighted')
logistic_regression_precision = precision_score(y_test, logistic_regression_pred, average='weighted')
k_neighbors_precision = precision_score(y_test, k_neighbors_pred, average='weighted')
svc_precision = precision_score(y_test, svc_pred, average='weighted')
decision_tree_precision = precision_score(y_test, decision_tree_pred, average='weighted')
random_forest_precision = precision_score(y_test, random_forest_pred, average='weighted')

perceptron_recall = recall_score(y_test, perceptron_pred, average='weighted')
logistic_regression_recall = recall_score(y_test, logistic_regression_pred, average='weighted')
k_neighbors_recall = recall_score(y_test, k_neighbors_pred, average='weighted')
svc_recall = recall_score(y_test, svc_pred, average='weighted')
decision_tree_recall = recall_score(y_test, decision_tree_pred, average='weighted')
random_forest_recall = recall_score(y_test, random_forest_pred, average='weighted')

perceptron_f1 = f1_score(y_test, perceptron_pred, average='weighted')
logistic_regression_f1 = f1_score(y_test, logistic_regression_pred, average='weighted')
k_neighbors_f1 = f1_score(y_test, k_neighbors_pred, average='weighted')
svc_f1 = f1_score(y_test, svc_pred, average='weighted')
decision_tree_f1 = f1_score(y_test, decision_tree_pred, average='weighted')
random_forest_f1 = f1_score(y_test, random_forest_pred, average='weighted')

'''perceptron_auc_roc = roc_auc_score(y_test, perceptron_pred)
logistic_regression_auc_roc = roc_auc_score(y_test, logistic_regression_pred)
k_neighbors_auc_roc = roc_auc_score(y_test, k_neighbors_pred)
svc_auc_roc = roc_auc_score(y_test, svc_pred)
decision_tree_auc_roc = roc_auc_score(y_test, decision_tree_pred)
random_forest_auc_roc = roc_auc_score(y_test, random_forest_pred)'''

# Print the evaluation metrics for each classifier
print("Perceptron:")
print("Accuracy:", perceptron_accuracy)
print("Precision:", perceptron_precision)
print("Recall:", perceptron_recall)
print("F1-score:", perceptron_f1)
#print("AUC-ROC:", perceptron_auc_roc)
print()

print("Logistic Regression:")
print("Accuracy:", logistic_regression_accuracy)
print("Precision:", logistic_regression_precision)
print("Recall:", logistic_regression_recall)
print("F1-score:", logistic_regression_f1)
#print("AUC-ROC:", logistic_regression_auc_roc)
print()

print("K-Nearest Neighbors:")
print("Accuracy:", k_neighbors_accuracy)
print("Precision:", k_neighbors_precision)
print("Recall:", k_neighbors_recall)
print("F1-score:", k_neighbors_f1)
#print("AUC-ROC:", k_neighbors_auc_roc)
print()

print("Support Vector Classifier:")
print("Accuracy:", svc_accuracy)
print("Precision:", svc_precision)
print("Recall:", svc_recall)
print("F1-score:", svc_f1)
#print("AUC-ROC:", svc_auc_roc)
print()

print("Decision Tree:")
print("Accuracy:", decision_tree_accuracy)
print("Precision:", decision_tree_precision)
print("Recall:", decision_tree_recall)
print("F1-score:", decision_tree_f1)
#print("AUC-ROC:", decision_tree_auc_roc)
print()

print("Random Forest:")
print("Accuracy:", random_forest_accuracy)
print("Precision:", random_forest_precision)
print("Recall:", random_forest_recall)
print("F1-score:", random_forest_f1)
#print("AUC-ROC:", random_forest_auc_roc)
