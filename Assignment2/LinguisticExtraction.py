import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#test data
test = pd.read_csv('trainRE.csv')
test = test.dropna()
text_test = test.loc[:, "Text"]
sentiment_test = test.loc[:, "Sentiment"]
text_test = text_test.tolist()

#train data
train = pd.read_csv('trainRE.csv')
train = train.dropna()
df = train.sample(frac=.001, random_state=42)
text_train = df.loc[:, "Text"]
sentiment_train = df.loc[:, "Sentiment"]
text_train = text_train.tolist()


#Part 3: Linguistic FeatureExtraction
#Bag of Words
vectorizerBOW = CountVectorizer()
bagOfWords_train = vectorizerBOW.fit_transform(text_train)
bagOfWords_test = vectorizerBOW.transform(text_test)
#print(vectorizerBOW.get_feature_names_out())

#tf*idf
vectorizerTFIDF = TfidfVectorizer()
tfidf_train = vectorizerTFIDF.fit_transform(text_train)
tfidf_test = vectorizerTFIDF.transform(text_test)

#word2vec - WORK IN PROGRESS
#model = Word2Vec(sentences=text_train, vector_size=100, window=5, min_count=1, workers=4)
#model.save("word2vec.model")


#Part 4 and 5

#Part 4 - Logistic regression model using tfidf
X_test = tfidf_test
y_test = sentiment_test

X_train = tfidf_train
y_train = sentiment_train

lr_tfidf = LogisticRegression()
svc_tfidf = SVC(probability=True)
nbc_tfidf = GaussianNB()
rfc_tfidf = RandomForestClassifier()

lr_tfidf.fit(X_train, y_train)
svc_tfidf.fit(X_train, y_train)
nbc_tfidf.fit(X_train.toarray(), y_train)
rfc_tfidf.fit(X_train, y_train)

#Part 5 - Logistic regression model using tfidf
y_predict = lr_tfidf.predict(X_test)
#y_prob = lr_tfidf.predict_proba(X_test)[:, 1]
y_lr_tfidf_predicted = lr_tfidf.predict(X_test)

y_rfc_tfidf_predicted = rfc_tfidf.predict(X_test)

print("Logistic Regression: tf*idf")
print(classification_report(y_test, y_lr_tfidf_predicted))

print("Random Forest Classifier: tf*idf")
print(classification_report(y_test, y_rfc_tfidf_predicted))


#Part 4 - Logistic regression model using bag of words
X_test = bagOfWords_test
y_test = sentiment_test

X_train = bagOfWords_train
y_train = sentiment_train

lr_bagOfWords = LogisticRegression()
svc_bagOfWords = SVC(probability=True)
nbc_bagOfWords = GaussianNB()
rfc_bagOfWords = RandomForestClassifier()

lr_bagOfWords.fit(X_train, y_train)
svc_bagOfWords.fit(X_train, y_train)
nbc_bagOfWords.fit(X_train.toarray(), y_train)
rfc_bagOfWords.fit(X_train, y_train)

y_lr_bagOfWords_predicted = lr_bagOfWords.predict(X_test)
#y_lr_bagOfWords_pred_proba = lr_bagOfWords.predict_proba(X_test)

#y_svc_bagOfWords_predicted = svc_bagOfWords.predict(X_test)
#y_svc_bagOfWords_pred_proba = svc_bagOfWords.predict_proba(X_test)

#y_nbc_bagOfWords_predicted = nbc_bagOfWords.predict(X_test.toarray())
#y_nbc_bagOfWords_pred_proba = nbc_bagOfWords.predict_proba(X_test)

y_rfc_bagOfWords_predicted = rfc_bagOfWords.predict(X_test)
#y_rfc_bagOfWords_pred_proba = rfc_bagOfWords.predict_proba(X_test)

print("Logistic Regression: Bag of Words")
print(classification_report(y_test, y_lr_bagOfWords_predicted))

#print("SVC")
#print(classification_report(y_test, y_svc_bagOfWords_predicted))

#print("Gaussian NB")
#print(classification_report(y_test, y_nbc_bagOfWords_predicted))

print("Random Forest Classifier: Bag of Words")
print(classification_report(y_test, y_rfc_bagOfWords_predicted))


#Part 5 - Logistic regression model using bag of words
#y_predict = lr_bagOfWords.predict(X_test)
#y_prob = lr_bagOfWords.predict_proba(X_test)[:, 1]
#print(classification_report(y_test, y_predict))
