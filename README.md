# Spam SMS Detection

Empowering Students, Elevating Careers - GrowthLink Machine Learning Assignment

## Task Objectives
Develop a robust machine learning model to classify SMS messages as spam or non-spam using the Kaggle SMS Spam Collection Dataset. This project implements text preprocessing, model training, and evaluation to achieve high accuracy, meeting GrowthLink’s standards for a month-long, remote, part-time assignment.

## Steps to Run the Project
1. **Setup Environment**:
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     venv\Scripts\activate  # Windows
     ./venv/bin/activate   # Mac/Linux
**Install dependencies:**

pip install -r requirements.txt

**Download Dataset:**

Download spam.csv from Kaggle SMS Spam Collection Dataset and place it in the project folder.

**Execute the Script:**

**Run:**

python smsdetection.py

Output includes data preprocessing confirmation, model training status, and evaluation metrics.

**Implementation Details**

**Preprocessing:**

Text is lowercased, stripped of punctuation and numbers, tokenized with NLTK, and stopwords are removed.
TF-IDF vectorization (max 3000 features) converts text into numerical features, optimizing for sparsity and relevance.

**Models:**

**Multinomial Naive Bayes:** A lightweight, probabilistic classifier ideal for text, leveraging word frequency independence.

**Support Vector Machine (SVM):** A linear kernel SVM for precise class separation, tuned for text classification.

**Evaluation:**

80/20 train-test split (random_state=42) ensures consistent results.
Metrics: accuracy, precision, recall, F1-score assess performance on spam (1) and ham (0) classes.

**Results**

**Dataset:**

5574 SMS messages (4825 ham, 747 spam).

**Naive Bayes:**

Accuracy: 0.9731
              precision    recall  f1-score   support
       ham (0)    0.97      1.00      0.98       965
      spam (1)    1.00      0.81      0.90       150
  accuracy                            0.97      1115
  
**SVM:**

Accuracy: 0.9794
              precision    recall  f1-score   support
       ham (0)    0.98      1.00      0.99       965
      spam (1)    0.99      0.85      0.91       150
  accuracy                            0.98      1115
  
**Insights**

**Model Comparison:** SVM outperforms Naive Bayes (97.94% vs. 97.31% accuracy), with a 4% higher spam recall (0.85 vs. 0.81), crucial for minimizing missed spam.

**Preprocessing Impact:** TF-IDF and stopword removal reduced noise, boosting model focus on key terms (e.g., "free," "win" for spam).

**Innovation:** Comparing two models (Naive Bayes for speed, SVM for precision) provides a balanced approach, with SVM’s slight edge justifying its computational cost.

**Challenges:** Dataset imbalance (87% ham) favors ham detection; future work could explore oversampling (e.g., SMOTE) for spam.

## Repository Contents

**smsdetection.py:** Core script with preprocessing, training, and evaluation.

**requirements.txt:** Dependency list for reproducibility.

**.gitignore:** Excludes venv/ to keep the repo clean.

**spam.csv:** Dataset (not uploaded; source linked above).

**Submission Notes**

**Submitted by:** Malloju Ramu

**GitHub:** https://github.com/MallojuRamu/sms-spam-detection
---
