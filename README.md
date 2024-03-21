# Isolation Forest

## Introduction

Isolation Forest is an anomaly detection algorithm that efficiently isolates outliers by building random binary trees. It identifies anomalies as data points that are easier to separate from the rest of the dataset, making it effective for detecting outliers in high-dimensional datasets without relying on distance measures.

The following project uses Isolation Forest for detecting anomalies over the transaction_anomalies_dataset. A detailed description about the dataset will be described later.

## Requirements

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Dataset Description

The transaction_anomalies_dataset dataset 12 features, each of whose description is given as follows:

Transaction_ID:                 Unique identifier for each transaction.
Transaction_Amount:             The monetary value of the transaction.
Transaction_Volume:             The quantity or number of items/actions involved in the transaction.
Average_Transaction_Amount:     The historical average transaction amount for the account.
Frequency_of_Transactions:      How often transactions are typically performed by the account.
Time_Since_Last_Transaction:    Time elapsed since the last transaction.
Day_of_Week:                    The day of the week when the transaction occurred.
Time_of_Day:                    The time of day when the transaction occurred.
Age:                            Age of the account holder.
Gender:                         Gender of the account holder.
Income:                         Income of the account holder.
Account_Type:                   Type of account (e.g., current, savings)

Statistics of the dataset are also shown during program execution, as well as null values inside the data columns.

## Training and Testing

Dataset information composing of feature description, statistics, and null value counts are shown during program execution. After training, statistical graphs are presented, and its accuracy during training is also shown. Lastly testing phase prompts runs shortly for runtime detection of anomalies. 

The model trained over above dataset acheived bla bla bla accuracy,f1 score as fellows

## Graphical Results

