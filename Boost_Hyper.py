import pandas as pd
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load training and testing data
train_data = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")  # Replace with your train dataset path
test_data = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")    # Replace with your test dataset path

# Split data into features and target
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']

# Define models
catboost = CatBoostClassifier(silent=True)
xgboost = XGBClassifier()
gradient_boosting = GradientBoostingClassifier()

# Define parameter grids for each model
catboost_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
}

xgboost_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
}

gradient_boosting_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
}
print("Hyper parameter tuning - CatBoost")
# Perform GridSearchCV for CatBoost
catboost_grid_search = GridSearchCV(catboost, catboost_param_grid, cv=3, scoring='accuracy')
catboost_grid_search.fit(X_train, y_train)

# Print the best parameters for CatBoost
print("CatBoost Best Parameters:", catboost_grid_search.best_params_)

print("Hyper parameter tuning - XGBoost")
# Perform GridSearchCV for XGBoost
xgboost_grid_search = GridSearchCV(xgboost, xgboost_param_grid, cv=3, scoring='accuracy')
xgboost_grid_search.fit(X_train, y_train)

# Print the best parameters for XGBoost
print("XGBoost Best Parameters:", xgboost_grid_search.best_params_)

print("Hyper parameter tuning - Gradient Boosting")
# Perform GridSearchCV for Gradient Boosting
gradient_boosting_grid_search = GridSearchCV(gradient_boosting, gradient_boosting_param_grid, cv=3, scoring='accuracy')
gradient_boosting_grid_search.fit(X_train, y_train)

# Print the best parameters for Gradient Boosting
print("Gradient Boosting Best Parameters:", gradient_boosting_grid_search.best_params_)

# Evaluate models on test data
catboost_test_pred = catboost_grid_search.predict(X_test)
xgboost_test_pred = xgboost_grid_search.predict(X_test)
gradient_boosting_test_pred = gradient_boosting_grid_search.predict(X_test)

# Calculate accuracy for each model
catboost_accuracy = accuracy_score(y_test, catboost_test_pred)
xgboost_accuracy = accuracy_score(y_test, xgboost_test_pred)
gradient_boosting_accuracy = accuracy_score(y_test, gradient_boosting_test_pred)

print("CatBoost Test Accuracy:", catboost_accuracy)
print("XGBoost Test Accuracy:", xgboost_accuracy)
print("Gradient Boosting Test Accuracy:", gradient_boosting_accuracy)

