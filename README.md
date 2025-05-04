# SMS_spam_classifier
# SMS Spam Detection

This repository contains a Python-based machine learning project for detecting spam messages using natural language processing (NLP) techniques. The dataset used is the publicly available **SMSSpamCollection** dataset, which contains labeled SMS messages categorized as either "spam" or "ham" (non-spam).

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Dataset](#dataset)
* [Project Workflow](#project-workflow)
* [Results](#results)
* [Usage](#usage)
* [Sample Predictions](#sample-predictions)
* [Future Improvements](#future-improvements)

---

## Overview

The goal of this project is to build a machine learning model that can classify SMS messages as spam or ham. The project includes steps for data preprocessing, feature extraction, model training, evaluation, and visualization of results. Two models, Multinomial Naive Bayes and Decision Tree Classifier, were implemented and compared for performance.

## Features

* Data visualization for spam vs. ham message distributions.
* Feature engineering, including:

  * Word count distribution.
  * Presence of currency symbols and numeric characters.
* Text preprocessing (removing special characters, lemmatization, and stopword removal).
* Spam classification using:

  * **Multinomial Naive Bayes**
  * **Decision Tree Classifier**
* Cross-validation for performance evaluation.
* Confusion matrix visualization.

## Technologies Used

* **Programming Language:** Python
* **Libraries:**

  * `numpy`
  * `pandas`
  * `nltk`
  * `scikit-learn`
  * `seaborn`
  * `matplotlib`
  * `imblearn`

## Dataset

* **Name:** SMSSpamCollection
* **Description:** A collection of 5,572 labeled SMS messages.
* **Format:** Tab-separated file with two columns:

  * `label`: Specifies if the message is "ham" (0) or "spam" (1).
  * `message`: The content of the SMS message.

## Project Workflow

1. **Data Loading:**

   * The dataset is loaded into a pandas DataFrame.

2. **Data Analysis and Visualization:**

   * Distribution of spam vs. ham messages.
   * Word count analysis for both categories.
   * Analysis of the presence of currency symbols and numeric characters.

3. **Data Preprocessing:**

   * Removal of non-alphabetic characters.
   * Conversion to lowercase.
   * Tokenization, stopword removal, and lemmatization.

4. **Feature Extraction:**

   * `TfidfVectorizer` used to transform text data into numeric vectors.

5. **Model Training and Evaluation:**

   * Models used:

     * Multinomial Naive Bayes
     * Decision Tree Classifier
   * Cross-validation for F1-score.
   * Performance evaluation using confusion matrix and classification report.

6. **Prediction Function:**

   * A custom function allows real-time prediction of SMS messages as spam or ham.

## Results

* **Multinomial Naive Bayes:**

  * Mean F1-score: \~0.95
  * Standard deviation: \~0.02
* **Decision Tree Classifier:**

  * Mean F1-score: \~0.92
  * Standard deviation: \~0.03

## Usage

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sms-spam-detector.git
   cd sms-spam-detector
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. Place the `SMSSpamCollection` dataset in the project directory.
2. Run the main script:

   ```bash
   python spam_detector.py
   ```

## Sample Predictions

| Message                                                                                 | Prediction |
| --------------------------------------------------------------------------------------- | ---------- |
| Had your mobile 9 months or more? Update to the latest colour mobiles with camera Free! | Spam       |
| Hey can we meet today at 6:30?                                                          | Ham        |

## Future Improvements

* Implement deep learning models (e.g., LSTM or BERT) for better accuracy.
* Fine-tune hyperparameters using `GridSearchCV`.
* Expand feature set to include sentiment analysis or other linguistic cues.
* Deploy the model as a web application using Flask/Django.



Feel free to contribute to the project by submitting issues or pull requests!
