#!/usr/bin/env python

import sys
import json
import numpy as np
import math
import random
from collections import Counter
import nltk
from nltk.collocations import *

from utils import getStemmedDocuments

unwanted_words = {'.', '...', '?', '!', ',', ';', "'s", '(', ')', ':', '``', "''", "'"}

def read_json(file, sample, raw_data=True, adv_features=False):
    data=[]
    count = 0;
    with open(file) as f:
        for json_data in f:
           review = []
           element = json.loads(json_data)  # load from current line as string
           rating = int(element["stars"])
           if raw_data:
               review = element["text"].split()
           else:
               if adv_features:
                   #review = [a+" "+b for a,b in zip(element["text"].split(" ")[:-1], element["text"].split(" ")[1:])]
                   review = getStemmedDocuments(element["text"], stem=False)
                   bigram_measures = nltk.collocations.BigramAssocMeasures()
                   #review2 = element["text"].split()
                   review2 = [e for e in review if e not in unwanted_words]
                   finder = BigramCollocationFinder.from_words(review2)
                   review = review + [a+" "+b for a,b in finder.nbest(bigram_measures.pmi, 10)]
                   #review =  review + [a+" "+b for a,b in nltk.bigrams(review)]
                   #review = [e for e in review if e not in unwanted_words]
               else:
                  review = getStemmedDocuments(element["text"]) 
           data.append((rating, review))
           #count = count +1
           #if count == sample:
           #    print("****")
           #    count=0
           #    break
    return data


def train_on_data(input_data):
    data = {}
    for rating, review in input_data:
        if rating not in data:
            data[rating] = {"words": list(review), "num_of_samples": 1}
        else:
            data[rating]["words"] += review
            data[rating]["num_of_samples"] += 1

    for rating in data:
        data[rating]["num_of_words"] = len(data[rating]["words"])
        data[rating]["words"] = Counter(data[rating]["words"])

    return data

def predict(train_data, test_data, adv_features=False):
    max_prob = -1e100

    #print("-----------------------")
    #print(test_data[0])
    for i in range(len(train_data)):
        sum = 0
        rating = i+1 
        c = train_data[rating]["words"]
        for j in range(len(test_data[1])):
            word = test_data[1][j]
            if adv_features and (j==0 or j==1 or j==2):
                    sum = sum + 2*math.log(c[word]+1)
            else: 
                sum = sum + math.log(c[word]+1)
        sum = sum - len(test_data[1]) * math.log(train_data[rating]["num_of_words"] + len(v))
        sum = sum + math.log(train_data[rating]["num_of_samples"])

        #print(sum)
        if sum>max_prob:
            max_prob = sum
            class_for_max_sum = rating
    return class_for_max_sum

def adv_test_on_data(train_data, test_data):
    count=0
    for i in range(len(test_data)):
        prediction = predict(train_data, test_data[i], adv_features=True)
        actual_class = test_data[i][0]
        confusion_mat[prediction-1][actual_class-1]+=1
        if actual_class == prediction:
            count += 1
    return count/len(test_data)


def test_on_data(train_data, test_data, algo="NB", confusion=False):
    count=0
    for i in range(len(test_data)):
        if algo == "NB":
            prediction = predict(train_data, test_data[i])
        if algo == "Random":
            prediction = random.randint(1, 5)
        if algo == "Majority":
            prediction = majority_class
        actual_class = test_data[i][0]
        if actual_class == prediction:
            count += 1
        if confusion:
            confusion_mat[prediction-1][actual_class-1]+=1
    return count/len(test_data)

def get_vocab(data):
    v = Counter([])
    for rating in data:
        v += data[rating]["words"]
    return v

def calculate_F1_score():
    sum = 0
    sum_row = np.sum(confusion_mat, axis=1)
    sum_col = np.sum(confusion_mat, axis=0)
    for i in range(5):
        p = float(confusion_mat[i][i]/sum_row[i])
        r = float(confusion_mat[i][i]/sum_col[i])
        F1 = 2*p*r/(p+r)
        sum += F1
        print("class %d precision=%f recall=%f F1=%f" % (i+1, p,r, F1))
    print("Average F1_score = %f" % (sum/5))

    #Grab command line input
if len(sys.argv[1:]) < 3:
    print("Usage: <path_of_train_data> <path_of_test_data>  <part_num>")
    sys.exit(1)
trainFile = sys.argv[1]
testFile = sys.argv[2]
part_num = sys.argv[3]

train_sample_size = 100000
test_sample_size = 10000
if part_num=='a' or part_num=='b' or part_num=='c':
    training_data = read_json(trainFile, train_sample_size)
    trained_data = train_on_data(training_data)
    v = get_vocab(trained_data)
    test_data = read_json(testFile,test_sample_size)

if part_num=='a':
    print("Executing Prediction on Training data")
    train_accuracy = test_on_data(trained_data, training_data)
    print("Training Accuracy: %f\n" % (train_accuracy))

    print("Executing Prediction on Testing data")
    test_accuracy = test_on_data(trained_data, test_data)
    print("Test Accuracy: %f\n" % (test_accuracy))

if part_num=='b':
    print("Executing Random Prediction on Test Data")
    test_accuracy = test_on_data(trained_data, test_data, algo="Random")
    print("Random Accuracy: %f\n" % (test_accuracy))

    print("Executing Majority Prediction on Test Data")
    majority_class = 1
    majority_class_sample = trained_data[1]["num_of_samples"]
    for i in range(len(trained_data)-1):
        if trained_data[i+2]["num_of_samples"] > majority_class_sample:
            majority_class = i+2
            majority_class_sample = trained_data[i+2]["num_of_samples"]
    test_accuracy = test_on_data(trained_data, test_data, algo="Majority")
    print("Majority Accuracy: %f\n" % (test_accuracy))

if part_num=='c':
    confusion_mat = np.zeros([5,5])
    test_on_data(trained_data, test_data, confusion=True)
    print("Confusion Matrix is")
    print(confusion_mat)
    print("\n")

if part_num=='d':
    print("Executing stem/stop Prediction on Test data")
    training_data = read_json(trainFile, train_sample_size, raw_data=False)
    print("!!!!!!!!!!!")
    trained_data = train_on_data(training_data)
    print("@@@@@@@@@@@@@@@@@@@@")
    test_data = read_json(testFile,test_sample_size, raw_data=False)
    print("###########################")
    v = get_vocab(trained_data)
    print("$$$$$$$$$$$$$$$$$$$$$$")
    test_accuracy = test_on_data(trained_data, test_data)
    print("Test stem/stop Accuracy: %f\n" % (test_accuracy))

if part_num=='e' or part_num=='f':
    print("Executing feature engg. Prediction on Test data")
    training_data = read_json(trainFile, train_sample_size, raw_data=False, adv_features=True)
    trained_data = train_on_data(training_data)
    test_data = read_json(testFile,test_sample_size, raw_data=False, adv_features=True)
    v = get_vocab(trained_data)   
    #test_accuracy = test_on_data(trained_data, test_data)
    #print("Test feature engg.1 Accuracy: %f\n" % (test_accuracy))
    confusion_mat = np.zeros([5,5])
    test_accuracy = adv_test_on_data(trained_data, test_data)
    print("Test feature engg. Accuracy: %f\n" % (test_accuracy))
    print("\n")
if part_num=='f':
    calculate_F1_score() 

if part_num=='g':
    print("Executing feature engg. Prediction on Full Test data")
    training_data = read_json(trainFile, train_sample_size, raw_data=False, adv_features=True)
    print("training on data")
    trained_data = train_on_data(training_data)
    print("reading test data")
    test_data = read_json(testFile,test_sample_size, raw_data=False, adv_features=True)
    v = get_vocab(trained_data)
    confusion_mat = np.zeros([5,5])
    print("testing on data")
    test_accuracy = test_on_data(trained_data, test_data)
    print("Test feature engg.1 Accuracy: %f\n" % (test_accuracy))
    test_accuracy = adv_test_on_data(trained_data, test_data)
    print("Test feature engg.2 Accuracy: %f\n" % (test_accuracy))
    calculate_F1_score()
