import pandas as pd

#Step 1: Exploratory Data Analysis

#import data frames
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#know size

test_size = len(test.index)
train_size = len(train.index)

print("There are " + str(test_size) + " entries in the test.csv dataset.")

print("There are " + str(train_size) + " entries in the train.csv dataset.")

#know sentiment distribution
distribution_test = test['Sentiment'].sum()
distribution_train = train['Sentiment'].sum()

print("There are " + str(distribution_test) + " positive sentiment values in test.csv")
print("There are " + str(test_size - distribution_test) + " negative sentiment values in test.csv")

print("There are " + str(distribution_train) + " positive sentiment values in train.csv")
print("There are " + str(train_size - distribution_train) + " negative sentiment values in train.csv")

test_positive_percentage = distribution_test / test_size
test_negative_percentage = (test_size -distribution_test) / test_size
print(str(test_positive_percentage) + " Percent positive in test.csv")
print(str(test_negative_percentage) + " Percent negative in test.csv")

train_positive_percentage = distribution_train / train_size
train_negative_percentage = (train_size -distribution_train) / train_size
print(str(train_positive_percentage) + " Percent positive in train.csv")
print(str(train_negative_percentage) + " Percent negative in train.csv")

#are there missing values?
print(test.isna().sum())
print(train.isna().sum())
print("There are no missing values in either dataset")


#Step 2: Text Preprocessing

#convert into lowercase
#remove digital numbers
#remove special characters
