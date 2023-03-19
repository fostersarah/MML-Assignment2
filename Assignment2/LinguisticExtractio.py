import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('trainRE.csv')
train = train.dropna()
df = train.sample(frac=.001, random_state=42)

text = df.loc[:, "Text"]
text = text.tolist()


#Bag of Words
vectorizer = CountVectorizer()
bagOfWords = vectorizer.fit_transform(text)
bagOfWordsArray = bagOfWords.toarray()

print(vectorizer.get_feature_names_out())


