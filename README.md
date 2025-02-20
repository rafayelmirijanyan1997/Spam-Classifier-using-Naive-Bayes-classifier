# Spam Classifier using Naïve Bayes

This Jupyter Notebook demonstrates a spam classifier using the Naïve Bayes algorithm. The classifier is trained on a dataset of SMS messages labeled as either "spam" or "ham" (non-spam).

## Table of Contents
1. Introduction
2. Dataset Exploration
3. Feature Encoding
4. Training and Testing the Classifier
5. Predicting Unseen Data
6. Conclusion

## Introduction
The dataset used is the SMSSpamCollection, containing SMS messages labeled as spam or ham.

## Dataset Exploration
The dataset is loaded and analyzed, revealing an imbalance with more ham messages than spam messages.

```python
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('SMSSpamCollection', sep='\t', header=None)
df.head()
```

## Feature Encoding
Messages are converted into feature vectors based on word presence.

```python
def textParse(bigString):
    import re
    return [tok.lower() for tok in re.split(r'[^A-Za-z]+', bigString) if len(tok) > 2]

def setOfWords2Vec(vocabList, inputSet):
    return [1 if word in inputSet else 0 for word in vocabList]
```

## Training and Testing the Classifier
A Multinomial Naïve Bayes classifier is trained and tested, achieving ~97.13% accuracy.

```python
X_train, X_test, Y_train, Y_test = train_test_split(instances, labels, test_size=0.25, random_state=24)

clf = MultinomialNB()
clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)
accuracy_score(Y_test, prediction)
```

## Predicting Unseen Data
Users can input a message to classify it as spam or ham.

```python
testset = input('Enter text message:')
returnVec = setOfWords2Vec(vocabList, textParse(testset))
testset = np.array(returnVec).reshape(1, -1)
predict(testset)
```

## Conclusion
The classifier effectively identifies spam messages with high accuracy. For full details, refer to the `spam_classifier.ipynb` notebook.
