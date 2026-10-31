# Credit Card Fraud Detection API Documentation

## 1. Input Fields (POST /predict)

| Field    | Type   | Valid Range | Meaning                                                                |
| :------- | :----- | :---------- | :--------------------------------------------------------------------- |
| Time     | number | >= 0        | Seconds elapsed between this transaction and the first in the dataset. |
| V1 - V28 | number | Any float   | 28 PCA-transformed anonymized features from the original dataset.      |
| Amount   | number | >= 0        | Transaction amount.                                                    |

## 2. Response Fields

| Field             | Type         | Meaning                                                               |
| :---------------- | :----------- | :-------------------------------------------------------------------- |
| fraud_probability | number (0-1) | The model's estimated probability that the transaction is fraudulent. |
| decision          | text         | "Fraud" if probability >= 0.5, otherwise "Legitimate".                |

## 3. Limits

- This API is a decision-support tool. It does not automatically freeze cards.
- It should be used alongside human fraud analysts who review the "Fraud" flags.
- The model was trained on 2013 European data, so it may need retraining for modern Ethiopian transaction patterns.
