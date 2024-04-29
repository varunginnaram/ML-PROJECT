import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, randint

# Load the training dataset
train_dataset_path = "C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx"  # Replace with the path to your training dataset
train_data = pd.read_excel(train_dataset_path)

# Load the testing dataset
test_dataset_path = "C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx"  # Replace with the path to your testing dataset
test_data = pd.read_excel(test_dataset_path)

# Separate features (X) and target (y) for training data
X_train = train_data.drop(columns=['Label'])  # Replace 'Label' with your target column name
y_train = train_data['Label']

# Separate features (X) and target (y) for testing data
X_test = test_data.drop(columns=['Label'])  # Replace 'Label' with your target column name
y_test = test_data['Label']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter grid for MLPClassifier
mlp_param_grid = {
    'max_iter': [500, 1000, 1500],            # Experiment with different maximum iterations
    'learning_rate_init': [0.001, 0.01],      # Try different learning rates
    'solver': ['adam', 'lbfgs'],              # Experiment with 'adam' and 'lbfgs' solvers
    'activation': ['relu', 'logistic'],       # Try 'relu' and 'logistic' activation functions
    'hidden_layer_sizes': [(50,), (100,), (150,)]  # Experiment with different hidden layer sizes
}

# Perform RandomizedSearchCV for MLPClassifier
mlp_clf = MLPClassifier()
mlp_random_search = RandomizedSearchCV(mlp_clf, mlp_param_grid, n_iter=10, cv=5, random_state=42)
mlp_random_search.fit(X_train_scaled, y_train)

# Print best parameters and best score for MLPClassifier
print("MLPClassifier - Best Parameters:", mlp_random_search.best_params_)
