import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

#Suppress the warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Load the training and testing datasets
train_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")  # Replace "train_dataset.xlsx" with the actual file path
test_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")  # Replace "test_dataset.xlsx" with the actual file path

# Assuming 'target_column' is the name of the column containing the target labels
X_train = train_df.drop(columns=['Label'])  # Features for training
y_train = train_df['Label']  # Target labels for training

X_test = test_df.drop(columns=['Label'])  # Features for testing
y_test = test_df['Label']  # Target labels for testing

# Initialize models
lr_model = LogisticRegression(random_state=42)
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)

# Train models
lr_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_lr = lr_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# Evaluate model performance
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_dt = accuracy_score(y_test, y_pred_dt)



# Make predictions on the training and testing data
y_pred_train_lr = lr_model.predict(X_train)
y_pred_train_nb = nb_model.predict(X_train)
y_pred_train_dt = dt_model.predict(X_train)

y_pred_test_lr = lr_model.predict(X_test)
y_pred_test_nb = nb_model.predict(X_test)
y_pred_test_dt = dt_model.predict(X_test)

# Calculate accuracy for training and testing sets
accuracy_train_lr = accuracy_score(y_train, y_pred_train_lr)
accuracy_train_nb = accuracy_score(y_train, y_pred_train_nb)
accuracy_train_dt = accuracy_score(y_train, y_pred_train_dt)

accuracy_test_lr = accuracy_score(y_test, y_pred_test_lr)
accuracy_test_nb = accuracy_score(y_test, y_pred_test_nb)
accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)

# Print accuracy scores
print("Training Accuracy - Logistic Regression:", accuracy_train_lr)
print("Training Accuracy - Naive Bayes:", accuracy_train_nb)
print("Training Accuracy - Decision Tree:", accuracy_train_dt)
print()
print("Testing Accuracy - Logistic Regression:", accuracy_test_lr)
print("Testing Accuracy - Naive Bayes:", accuracy_test_nb)
print("Testing Accuracy - Decision Tree:", accuracy_test_dt)




# Calculate metrics for Logistic Regression
precision_lr, recall_lr, f1_score_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='weighted')
auc_roc_lr = roc_auc_score(y_test, lr_model.predict_proba(X_test), multi_class='ovr')

# Calculate metrics for Naive Bayes
precision_nb, recall_nb, f1_score_nb, _ = precision_recall_fscore_support(y_test, y_pred_nb, average='weighted')
auc_roc_nb = roc_auc_score(y_test, nb_model.predict_proba(X_test), multi_class='ovr')

# Calculate metrics for Decision Tree
precision_dt, recall_dt, f1_score_dt, _ = precision_recall_fscore_support(y_test, y_pred_dt, average='weighted')
auc_roc_dt = roc_auc_score(y_test, dt_model.predict_proba(X_test), multi_class='ovr')

# Print metrics for each model
print("Logistic Regression Metrics:")
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1-score:", f1_score_lr)
print("AUC-ROC:", auc_roc_lr)
print()

print("Naive Bayes Metrics:")
print("Precision:", precision_nb)
print("Recall:", recall_nb)
print("F1-score:", f1_score_nb)
print("AUC-ROC:", auc_roc_nb)
print()

print("Decision Tree Metrics:")
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-score:", f1_score_dt)
print("AUC-ROC:", auc_roc_dt)
