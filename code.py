import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import nltk
import re

data = pd.read_csv("C:\projects\gender\gender-classifier-DFE-791531.csv", encoding="latin1")

data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis = 0, inplace = True) # we dropped the null rows


data.info()


import nltk
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

data.gender = [1 if gender == "female" else 0 for gender in data.gender]


import nltk
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]", " ", description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    lemma = nltk.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
    from sklearn.feature_extraction.text import CountVectorizer

max_features = 5000

cv = CountVectorizer(max_features=max_features, stop_words = "english")
sparce_matrix = cv.fit_transform(description_list).toarray()

print("top used {} words: {}".format(max_features, cv.get_feature_names()))


y = data.iloc[:, 0].values
x = sparce_matrix


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
# prediction
y_pred = rf.predict(x_test)
# Random Forest
accuracy = 100.0 * accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
lr = LogisticRegression(max_iter = 2000)
lr.fit(x_train,y_train)
# prediction
y_pred = lr.predict(x_test)
accuracy = 100.0 * accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
