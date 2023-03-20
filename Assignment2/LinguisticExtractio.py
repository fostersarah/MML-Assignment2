import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
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
df = train.sample(frac=.01, random_state=42)
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
model = Word2Vec(sentences=text_train, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")


#Part 4

#Part 4 - Logistic regression model using tfidf

X_test = tfidf_test
y_test = sentiment_test

X_train = tfidf_train
y_train = sentiment_train

lr_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_tfidf.fit(X_train, y_train)

#Part 5 - Logistic regression model using tfidf

y_predict = lr_tfidf.predict(X_test)
y_prob = lr_tfidf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_predict))



#Part 4 - Logistic regression model using bag of words

X_test = bagOfWords_test
y_test = sentiment_test

X_train = bagOfWords_train
y_train = sentiment_train

lr_bagOfWords = LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_bagOfWords.fit(X_train, y_train)

#Part 5 - Logistic regression model using bag of words

y_predict = lr_bagOfWords.predict(X_test)
y_prob = lr_bagOfWords.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_predict))