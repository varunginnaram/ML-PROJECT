from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your training and testing data
train_data = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")
test_data = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")

# Assuming the last column contains the target variable
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Initialize and train the MLPClassifier with best parameters
mlp = MLPClassifier(solver='adam', max_iter=500, learning_rate_init=0.001, hidden_layer_sizes=(50,), activation='relu')
mlp.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = mlp.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Initialize and train a normal MLPClassifier with default parameters
normal_mlp = MLPClassifier()
normal_mlp.fit(X_train, y_train)

# Make predictions on the testing data using the normal MLP
y_pred_normal = normal_mlp.predict(X_test)

# Calculate evaluation metrics for the normal MLP
accuracy_normal = accuracy_score(y_test, y_pred_normal)
precision_normal = precision_score(y_test, y_pred_normal, average='macro')
recall_normal = recall_score(y_test, y_pred_normal, average='macro')
f1_normal = f1_score(y_test, y_pred_normal, average='macro')

# Print evaluation metrics for the normal MLP
print("Accuracy (Normal MLP):", accuracy_normal)
print("Precision (Normal MLP):", precision_normal)
print("Recall (Normal MLP):", recall_normal)
print("F1 Score (Normal MLP):", f1_normal)
