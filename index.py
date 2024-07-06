#USING LOGISTIC REGRESSION MODEL (acc 98%)

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


#load dataset
df = pd.read_csv('emails.csv')

print(df.head())
print(df.shape)
print(df.info())

print(df['spam'].value_counts()) #mail = 4360 and spam = 1368

"""Under-sampling decreases the number of rows where ham is present and makes
 it equal to spam. Over-sampling is just the opposite"""

# divide the dataset into features and labels.
X = df['text']
y = df['spam']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

print(X_train.shape, X_test.shape) #((1120,), (374,))
print(y_train.shape, y_test.shape) #((1120,), (374,))

#TfidfVectorizer converts text to numbers.
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
new_X_train = vectorizer.fit_transform(X_train)
new_X_test = vectorizer.transform(X_test)

#Logistic Regression model
lr = LogisticRegression()
lr.fit(new_X_train, y_train)
lr_prediction = lr.predict(new_X_test)

#Accuracy of model
accuracy = accuracy_score(y_test, lr_prediction)
print(accuracy) #0.9860383944153578

#confusion matrix
confmat = confusion_matrix(y_test,lr_prediction)
print(confmat)
# Plotting the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()