# # Problem 2

import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
import numpy as np

# Classify names into four categories: Greek, Arabic, American, Japanese
datapath = '/content/sample_data/'
classDocs = ['arabic.txt', 'greek.txt', 'japan.txt', 'us.txt']

# This function returns all of the names split by " " and segmented by class
def load_file(datapath):
    classLines = []
    for classDoc in classDocs:
        classDocLines = []
        with open(datapath + classDoc, 'r') as document:
            for line in document:
                classDocLines.append(line.lower().split())
        classLines.append(classDocLines)
    return classLines

# This function takes in a list of the lines in each class and outputs training and test splits for each of the class sets
def splits(classLines):
    # initializing empty training and testing sets
    trainingSets,testSet = [],[]

    for i in range(len(classLines)):
        train, test = train_test_split(classLines[i], test_size = 0.3, random_state = 42)
        trainingSets.append(list(zip(train, [i] * len(train))))
        testSet += list(zip(test, [i] * len(test)))
    random.shuffle(testSet)
    
    return trainingSets, testSet

# This function takes in a training set which is partitioned by class and tells you which set belongs where
def vocabularySets(trainingData):
    vocabularySets = []
    for trainingSet in trainingData:
        vocab = defaultdict(int)
        for line, cat in trainingSet:
            for word in line:
                vocab[word] += 1
        vocabularySets.append(vocab)
    return vocabularySets

# getting classlines
classLines = load_file(datapath)

# splitting it into training and testing data
training, test = splits(classLines)

# Assigning vocabulary set from training set
vocabSets = vocabularySets(training)

# This takes in a line and the vocabulary set for each class, calculates the log odds of each class and then selects randomly among the best options

def prediction(line, vocabSets):
    logOdds = [0] * len(vocabSets)
    for word in line:
        occurences = [vocabSets[i][word] for i in range(len(vocabSets))]
        for i in range(len(logOdds)):
            logOdds[i] += np.log((vocabSets[i][word] + 1) / (occurences[i] + len(vocabSets)))
    maxOdds = max(logOdds)
    bestChoices = []
    for i in range(len(logOdds)):
        if logOdds[i] == maxOdds:
            bestChoices.append(i)
    return random.choice(bestChoices)

# This function takes in training and test sets and returns a list of predictions on then test data
def classifier(trainingData, testData):
    vocabSets = vocabularySets(trainingData)
    predictions = []
    for line, cat in testData:
        predictions.append(prediction(line, vocabSets))
    return predictions

# predicting on the data
predictions = classifier(training, test)

# Calculating Error
def relativeError(predictions, test):
    E = 0
    for i in range(len(test)):
        if predictions[i] != test[i][-1]:
            E += 1
    return E / len(test)

print(f'Relative error : {round(relativeError(predictions, test) *100 ,3)} %')

# Calculating Accuracy
def relativeAccuracy(predictions, test):
    Accu = 0
    for i in range(len(test)):
        if predictions[i] == test[i][-1]:
            Accu += 1
    return Accu / len(test)

print(f'Relative accuracy : {round(relativeAccuracy(predictions, test) *100 ,3)} %')

# How many words don't show up in any of the lists
count = 0
for line, cat in test:
    inlist = False
    for word in line:
        for vocabSet in vocabSets:
            if word in vocabSet:
                inlist = True
    if not inlist:
        count += 1
print(f'{(count / len(test))*100} %')