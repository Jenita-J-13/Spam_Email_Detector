# Spam Email Detection using Machine Learning

## Overview

This project uses Machine Learning to detect whether a message is Spam or Not Spam.

## Features

* Text preprocessing using TF-IDF
* Spam classification using Naive Bayes
* Interactive message prediction
* Accuracy around 96%

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

## Dataset

Dataset Used: SMS Spam Collection Dataset

Download Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?utm_source=chatgpt.com

Dataset:
SMS Spam Collection Dataset

Source:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Description:
The dataset contains 5,572 SMS messages labeled as either spam or ham (not spam).

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the project:

python spam_detector.py

## Example

Input:
Congratulations! You have won ₹50,000. Claim now.

Output:
SPAM MESSAGE
