#Class imbalance test on the dataset

import pandas as pd
import matplotlib.pyplot as plt



# Read the Excel file into a DataFrame
df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")

# 'df' is your DataFrame and 'Label' is the name of the target variable column
class_distribution = df["Label"].value_counts()

# Visualize class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

# Add frequencies on the plot
for i, freq in enumerate(class_distribution):
    plt.text(i, freq, str(freq), ha='center', va='bottom')

plt.show()

# Print frequencies of each class
print("Frequencies of each class:")
print(class_distribution)

# Calculate class imbalance ratio
class_ratios = class_distribution / class_distribution.sum()
minority_class_ratio = class_ratios.min()
majority_class_ratio = class_ratios.max()
class_imbalance_ratio = majority_class_ratio / minority_class_ratio

print("Class Imbalance Ratio:", class_imbalance_ratio)

# Print statement based on class imbalance ratio
if class_imbalance_ratio > 5:
    print("The class imbalance ratio is significantly high, indicating a severe class imbalance.")
elif class_imbalance_ratio > 1:
    print("The class imbalance ratio is moderately high, indicating a class imbalance.")
else:
    print("The class imbalance ratio is close to 1, indicating a balanced dataset.")
