import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

################### Loading Data

file_path = 'transaction_anomalies_dataset.csv'
data = pd.read_csv(file_path)

################### Printing Statistics of Whole Data

print("\n\nNull Values:\n\n", data.isnull().sum())
print("\nColumn Information:\n\nTransaction_ID:\t\tUnique identifier for each transaction.\nTransaction_Amount:\t\tThe monetary value of the transaction.\nTransaction_Volume:\t\tThe quantity or number of items/actions involved in the transaction.\nAverage_Transaction_Amount:\t\tThe historical average transaction amount for the account.\nFrequency_of_Transactions:\t\tHow often transactions are typically performed by the account.\nTime_Since_Last_Transaction:\t\tTime elapsed since the last transaction.\nDay_of_Week:\t\tThe day of the week when the transaction occurred.\nTime_of_Day:\t\tThe time of day when the transaction occurred.\nAge:\t\tAge of the account holder.\nGender:\t\tGender of the account holder.\nIncome:\t\tIncome of the account holder.\nAccount_Type:\t\tType of account (e.g., current, savings)\n\n")
print("Descriptive Statistics:\n\n", data.describe())

################### Cleaning Data

data2= data
data = data.drop(columns=['Transaction_ID'])
data = data.drop(columns=['Day_of_Week'])
data = data.drop(columns=['Time_of_Day'])
data = data.drop(columns=['Gender'])
data = data.drop(columns=['Account_Type'])
numpy_data=np.array(data.values)


###################### Graphs

def distribution_of_transactions(data):
    """
    Plot the distribution of transaction amounts.

    Parameters:
    - data (pd.DataFrame): Data containing transaction amounts.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Transaction_Amount'], bins=30, kde=False)
    plt.title('Distribution of Transaction Amount')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.show()


def transactions_amount_by_account_type(data):
    """
    Plot transaction amounts by account type.

    Parameters:
    - data (pd.DataFrame): Data containing transaction amounts and account types.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Account_Type', y='Transaction_Amount', data=data)
    plt.title('Transaction Amount by Account Type')
    plt.xlabel('Account Type')
    plt.ylabel('Transaction Amount')
    plt.savefig('Transaction Amount by Account Type.png')
    plt.show()


def average_transaction_amount_by_age(data):
    """
    Plot the average transaction amount by age.

    Parameters:
    - data (pd.DataFrame): Data containing transaction amounts and account types.
    """
    average_transaction_by_age = data.groupby('Age')['Transaction_Amount'].mean()
    plt.figure(figsize=(10, 6))
    plt.axhline(y=average_transaction_by_age.mean(), color='green', linestyle='--', label='Mean Transaction Amount')
    for account_type, color in zip(data['Account_Type'].unique(), ['blue', 'red']):
        subset_data = data[data['Account_Type'] == account_type]
        plt.scatter(subset_data['Age'], subset_data['Transaction_Amount'], color=color, label=account_type)

    plt.title('Average Transaction Amount by Age')
    plt.xlabel('Age')
    plt.ylabel('Average Transaction Amount')
    plt.legend()
    plt.savefig('Average Transaction Amount by Age.png')
    plt.show()


def count_of_transactions_by_day_of_week(data):
    """
    Plot the count of transactions by day of the week.

    Parameters:
    - data (pd.DataFrame): Data containing transaction amounts and days of the week.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Day_of_Week', data=data)
    plt.title('Count of Transactions by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    plt.savefig('Count of Transactions by Day of Week.png')
    plt.show()


def correlation(numpy_data, data):
    """
    Plot the correlation matrix.

    Parameters:
    - numpy_data (numpy.ndarray): NumPy array containing the data.
    - data (pd.DataFrame): Data containing column names.

    Note:
    The numpy_data array should be preprocessed to contain only numerical values.
    """
    numpy_data = numpy_data.astype(float)
    correlation_matrix = np.corrcoef(numpy_data, rowvar=False)
    plt.figure(figsize=(10, 6))
    colors = [(0, 0, 1), (0.5, 0, 0.5), (1, 1, 0)]  # Blue, Purple, Yellow
    cmap = LinearSegmentedColormap.from_list("Custom", colors, N=256)
    sns.heatmap(correlation_matrix, annot=False, cmap=cmap, cbar=True, xticklabels=data.columns, yticklabels=data.columns)
    plt.title('Correlation Matrix')
    plt.savefig('Correlation Matrix.png')
    plt.show()

def visualize_anomalies(data):
    """
    Visualize anomalies in the data.

    Parameters:
    - data (pd.DataFrame): Data containing transaction amounts and average transaction amounts.

    Note:
    Anomalies are detected based on a threshold value of 2000 for the transaction amount.
    """
    threshold = 2000
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Transaction_Amount', y='Average_Transaction_Amount', data=data, hue=data['Transaction_Amount'] > threshold, palette={False: 'blue', True: 'red'})
    plt.title('Average Transaction Amount vs Transaction Amount')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Average Transaction Amount')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='False', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='True', markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=legend_elements, title='Anomaly', loc='upper right')
    plt.savefig('Average Transaction Amount vs Transaction Amount.png')
    plt.show()

distribution_of_transactions(data2) 
transactions_amount_by_account_type(data2)
average_transaction_amount_by_age(data2)
count_of_transactions_by_day_of_week(data2)
correlation(numpy_data, data)
visualize_anomalies(data2)

################### Training

def evaluate(data):
    """
    Evaluate anomalies in the data using Isolation Forest.

    Parameters:
    - data (numpy.ndarray): Data to evaluate for anomalies.

    Returns:
    - sklearn.ensemble.IsolationForest: Trained Isolation Forest model.
    - float: Ratio of anomalies in the data.
    """
    model = IsolationForest(n_estimators=100, max_samples=256)
    model.fit(data)
    predictions = model.predict(data)
    predictions = np.where(predictions == 1, 1, 0)  # Convert to binary: 1 for normal, 0 for anomaly
    ratio_anomalies = np.sum(predictions == 0) / len(data)
    return model, ratio_anomalies

def extract_relevant_feature_data(data, relevant_features):
    """
    Extract relevant feature data from the given dataset.

    Parameters:
    - data (pd.DataFrame): Data containing relevant features.
    - relevant_features (list): List of relevant feature names.

    Returns:
    - numpy.ndarray: Extracted relevant feature data.
    """
    relevant_features_indexes = [data.columns.get_loc(feature) for feature in relevant_features]
    relevant_features_data = data.iloc[:, relevant_features_indexes].values
    return relevant_features_data

_, ratio_anomalies = evaluate(numpy_data)
print("Ratio of anomalies in total data:", ratio_anomalies)

relevant_features = ["Transaction_Amount", "Average_Transaction_Amount", "Frequency_of_Transactions"]
relevant_data = extract_relevant_feature_data(data, relevant_features)

model, ratio_anomalies = evaluate(relevant_data)
print("Ratio of anomalies in relevant features data:", ratio_anomalies)

###################### Testing

while True:
    query=[]
    for i in range(len(relevant_features)):
        userinput=input(f"(press e for exit) Enter the value for '{relevant_features[i]}': ")
        if userinput == "e":
            sys.exit()
        query.append(float(userinput))
    single_row = np.array([query])


    prediction = model.predict(single_row)
    if prediction == -1:
        print("Anomaly detected: This transaction is flagged as an anomaly.")
    else:
        print("No anomaly detected.")

