import pandas as pd
import re

def removeUpperCase(message):
    upperCase = "[A-Z]"
    return re.sub(upperCase, "", message)

def removeUserHandel(message):
    newString = ""
    atHandel = False
    for count in range(len(message)):
        if (message[count] == '@'):
            atHandel = True
        if (atHandel == False):
            newString = newString + message[count]
        if (message[count] == ' '):
            atHandel = False
    return newString

def removeLinks(message):
    newString = ""
    atHttp = False
    for count in range(len(message)):
        if (message[count] == ':' and count > 4):
            if (message[count - 1] == 'p' and message[count - 2] == 't' and message[count - 3] == 't' and message[count - 4] == 'h'):
                atHttp = True
                newString = newString[0:count - 4]
        if (atHttp == False):
            newString = newString + message[count]
        if (message[count] == ' '):
            atHttp = False
    return newString

def removeNumbers(message):
    numbers = "[0-9]"
    return re.sub(numbers, "", message)

def removeSpecialCharacters(message):
    special = "[^a-zA-Z ]"
    return re.sub(special, "", message)

def runExploratoryDataAnalysis():
    # know size
    test_size = len(test.index)
    train_size = len(train.index)

    print("There are " + str(test_size) + " entries in the test.csv dataset.")

    print("There are " + str(train_size) + " entries in the train.csv dataset.")

    # know sentiment distribution
    distribution_test = test['Sentiment'].sum()
    distribution_train = train['Sentiment'].sum()

    print("There are " + str(distribution_test) + " positive sentiment values in test.csv")
    print("There are " + str(test_size - distribution_test) + " negative sentiment values in test.csv")

    print("There are " + str(distribution_train) + " positive sentiment values in train.csv")
    print("There are " + str(train_size - distribution_train) + " negative sentiment values in train.csv")

    test_positive_percentage = distribution_test / test_size
    test_negative_percentage = (test_size - distribution_test) / test_size
    print(str(test_positive_percentage) + " Percent positive in test.csv")
    print(str(test_negative_percentage) + " Percent negative in test.csv")

    train_positive_percentage = distribution_train / train_size
    train_negative_percentage = (train_size - distribution_train) / train_size
    print(str(train_positive_percentage) + " Percent positive in train.csv")
    print(str(train_negative_percentage) + " Percent negative in train.csv")

    # are there missing values?
    print(test.isna().sum())
    print(train.isna().sum())
    print("There are no missing values in either dataset")


#import data frames
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

runExploratoryDataAnalysis()

#Step 2: Text Preprocessing

for count in range(len(test.index)):
    tempString = test.iloc[count]['Text']
    tempString = tempString.lower()
    tempString = removeNumbers(tempString)
    tempString = removeUserHandel(tempString)
    tempString = removeLinks(tempString)
    tempString = removeSpecialCharacters(tempString)
    test.loc[count, 'Text'] = tempString

test.to_csv("testRE.csv")

for count2 in range(len(train.index)):
    tempString2 = train.iloc[count2]['Text']
    tempString2 = tempString2.lower()
    tempString2 = removeNumbers(tempString2)
    tempString2 = removeUserHandel(tempString2)
    tempString2 = removeLinks(tempString2)
    tempString2 = removeSpecialCharacters(tempString2)
    train.loc[count2, 'Text'] = tempString2

train.to_csv("trainRE.csv")
