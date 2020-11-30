# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:17:40 2020

@author: Edoardo
"""


import numpy as np
import pandas as pd
import jsonlines
import json
import random

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import normalize, scale, QuantileTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import time
import networkx as nx
from networkx import json_graph


classesFunctions = ["math", "encryption", "string", "sort"]
instructionsx64 = {
        "movement": ["mov", "push", "pop", "cwtl", "cltq", "cqto"], 
        "unary": ["inc", "neg", "dec", "not"],
        "binary": ["leaq", "add", "sub", "imul", "xor", "or", "and"],
        "shift": ["sal", "shl", "sar", "shr"],
        "specialarithmetic": ["imulq", "mulq", "idivq", "divq"],
        "comparison": ["cmp", "test"],
        "set": ["sete", "setz", "setne", "setnz", "sets", "setns", "setg", "setnle", "setge", "setnl", "setl", "setnge", "seta", "setae", "setnb", "setb", "setnae", "setbe", "setna"],
        "jump": ["jmp", "je", "jz", "jne", "js", "jns", "jg", "jnle", "jge", "jnl", "jl", "jnge", "jle", "jng", "ja", "jnbe", "jae", "jnb", "jb", "jnae", "jbe", "jna"],
        "conditionalmovement": ["cmove", "cmovz", "cmovne", "cmovnz", "cmovs", "cmovns", "cmovg", "cmovnle", "cmovge", "cmovnl", "cmovl", "cmovnge", "cmovle", "cmovng", "cmova", "cmovnbe", "cmovae", "cmovnb", "cmovb", "cmovnae", "cmovbe", "cmovna"],
        "call": ["call", "leave", "ret"]
    }


instructionssemplifed = {
    "movement": ["mov", "push", "pop", "leaq"],
    "math": ["xor", "or", "add", "sub", "mul", "and", "not", "neg"],
    "shift": ["sal", "shl", "sar", "shr"],
    "comp": ["cmp", "test"],
    "call": ["call", "leave", "ret"],
    "mem": ["xmm", "0x"]
}

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    
    print(unique_labels(y_true, y_pred))
    classes = ["math", "encryption", "sort", "string"]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def parseDataset(dataset, outputDataset="cleaned_dataset.json"):
    cleaned = []
    with jsonlines.open(dataset) as reader:
        for line in reader:
            cleaned.append(line)
    
    with open(outputDataset, mode='w') as f:
        json.dump(cleaned, f)
    
    return "File creato"


def buildNewDataset(dataset, outputDB="builded_dataset.json"):
    output = [{"data": [], "target": []}]
    
    with open(dataset, mode='r') as f:
        dataset = json.load(f)
        
    for line in dataset:
        output[0]["data"].append(line["lista_asm"])
        output[0]["target"].append(line["semantic"])
    
    with open(outputDB, mode='w') as f:
        json.dump(output, f)
        
    return output



def preprocessingDataset(dataset, is_dataset=True, instructionToCheck=["movement", "binary", "comparison", "call", "shift", "unary"], normalize2=False, is_scale=False):
    """
    Parameters
    ----------
    dataset : the dataset
    is_dataset : if it is loading a dataset
    instructionToCheck : list of categories to check
        DESCRIPTION. The default is ["movement", "binary", "comparison", "call", "shift", "unary"].
    normalize2 : if we want normalize
    is_scale : if we want scale
    Returns
    -------
    X_All attributes np array.
    """
    attributes_list = []
    if is_dataset:
        for func in dataset:
            length_asm = len(func["lista_asm"])
            asm_list = func["lista_asm"][1:length_asm - 1].split("', ")
            nx_graph = json_graph.adjacency_graph(func['cfg'])
            instruction = []
            #print(type(len(nx_graph.edges())))
            css = nx.components.number_strongly_connected_components(nx_graph)
            #print(css)
            complexity = len(nx_graph.edges()) - len(nx_graph.nodes()) - css
            
            """
            for key in range(0, len(instructionToCheck)):
                instruction.append(0)
                
            for command in asm_list:
                for i, clss in enumerate(instructionToCheck):
                    count = instruction[i]
                    for cmd in range(0, len(instructionsx64[clss])):
                        count += command.count(instructionsx64[clss][cmd])
                    instruction[i] = count
            """

            # count of each command category
            
            for key in range(0, len(instructionsx64.keys())):
                instruction.append(0)
            reg_used_count = 0
            for command in asm_list:
                reg_used_count += command.count("0x")
                for i, key in enumerate(instructionsx64.keys()):
                    count = instruction[i]
                    for j in range(0, len(instructionsx64[key])):
                        count += command.count(instructionsx64[key][j])
                    instruction[i] = count 
                    
            

            # for vectorizer
            """
            #instruction.append(len(asm_list))
            for command in asm_list:
                command = command.replace("'", '')
                instruction.append(command) # ["push rbp", "push r15"]
                
            attributes_list.append(' '.join(instruction))
            """
            instruction.append(len(nx_graph.nodes()))
            instruction.append(len(nx_graph.edges()))
            instruction.append(complexity)
            instruction.append(reg_used_count)
            attributes_list.append(instruction)
        if normalize2:
            attributes_list = normalize(attributes_list, norm='l2')
        if is_scale:
            attributes_list = scale(attributes_list)
            
    elif True:
        length_asm = len(dataset["lista_asm"])
        asm_list = dataset["lista_asm"][1:length_asm - 1].split("', ")
        
        nx_graph = json_graph.adjacency_graph(dataset['cfg'])
        instruction = []
        #print(type(len(nx_graph.edges())))
        css = nx.components.number_strongly_connected_components(nx_graph)
        #print(css)
        complexity = len(nx_graph.edges()) - len(nx_graph.nodes()) - css
        for key in range(0, len(instructionsx64.keys())):
            instruction.append(0)
        reg_used_count = 0
        for command in asm_list:
            reg_used_count += command.count("0x")
            for i, key in enumerate(instructionsx64.keys()):
                count = instruction[i]
                for j in range(0, len(instructionsx64[key])):
                    count += command.count(instructionsx64[key][j])
                instruction[i] = count

        """
        for key in range(0, len(instructionToCheck)):
                instruction.append(0)
         
        
        for command in asm_list:
            for i, clss in enumerate(instructionToCheck):
                count = instruction[i]
                for cmd in range(0, len(instructionsx64[clss])):
                    count += command.count(instructionsx64[clss][cmd])
                instruction[i] = count
        """
        
        instruction.append(len(nx_graph.nodes()))
        instruction.append(len(nx_graph.edges()))
        instruction.append(complexity)
        instruction.append(reg_used_count)
        attributes_list.append(instruction)
        if normalize2:
            attributes_list = normalize(attributes_list, norm='l2')
    
    return np.array(attributes_list)
        

def doVectorizer(typevec):
    if typevec == "hashing":
        vectorizer = HashingVectorizer()
    elif typevec == "count":
        vectorizer = CountVectorizer()
    elif typevec == "tfid":
        vectorizer = TfidfVectorizer()
    
    return vectorizer

def doModel(modeltype, xtrain, ytrain):
    if modeltype == "bernoulli":
        return BernoulliNB().fit(xtrain, ytrain)
    elif modeltype == "multinomial":
        return MultinomialNB().fit(xtrain, ytrain)
    elif modeltype == "trees":
        return tree.DecisionTreeClassifier().fit(xtrain, ytrain)
    elif modeltype == "svm":
        return svm.SVC(kernel='linear', C=1).fit(xtrain, ytrain)
    elif modeltype == "gaussian":
        return GaussianNB().fit(xtrain, ytrain)
    elif modeltype == "regression":
        return LogisticRegression().fit(xtrain, ytrain)
    return 
    

def predictAll(X_all, y_all, vect, model, test_size=0.2, random_state=15, is_string=False, blind_test_also=False, testfile="nodupblindtest.json"):
    """
    Parameters
    ----------
    X_all : np array attributes
    y_all : np array target classes
    vect : vectorizer type, if use vector ["hashing", "cont", "tfid"]
    model : the model we'll use to train ["bernoulli", "multinomial", "trees", "svm", "gaussian", "regression"]
    test_size : number optional size of the test, the default is 0.2.
    random_state : integer, random seed
    is_string : if we're using text or not
    blind_test_also : run also the blindtest
    testfile : string, The default is "nodupblindtest.json".
    Returns
    -------
    None.
    """
    
    modelstr = model
    if is_string:
        vectorizer = doVectorizer(vect)
        X_all = vectorizer.fit_transform(X_all)
        
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=random_state)
    
    print("Train: {} - Test: {}".format(X_train.shape[0], X_test.shape[0]))

    model = doModel(model, X_train, y_train)
    
    
    y_predict = model.predict(X_test)
    print(classification_report(y_test, y_predict))
    
    cm = confusion_matrix(y_test, y_predict, labels=None, sample_weight=None)
    #print(y_predict, " ahahuah")
    plot_confusion_matrix(y_test, y_predict, classes=[0, 1, 2, 3], normalize=True)
    
    cv = ShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)
    scores = cross_val_score(model, X_all, y_all, cv=cv)
    print(scores)
    print("Accuracy: {} (+/- {})".format(scores.mean(),scores.std() * 2))
    
    if blind_test_also:
        filename = int(time.time())
        f = open("results/" + modelstr + "_" + str(filename) + ".txt","w+")
        with jsonlines.open(testfile) as reader:
            for line in reader:
                if is_string:
                    xnew = vectorizer.transform(np.array([line["lista_asm"]]))
                else:
                    xnew = preprocessingDataset(line, is_dataset=False)
                newpr = model.predict(xnew)
                f.write(newpr[0] + "\n")
        f.close()
    
    return
    
if __name__ == "__main__":
    #pdbnodup = parseDataset("noduplicatedataset.json", "cleaned_nodupdataset.json")
    #out = buildNewDataset("cleaned_nodupdataset.json", "builded_nodupdataset.json")
    c18 = True
    ds = "cleaned_nodupdataset.json" if c18 else "cleaned_dataset.json"
    ds2 = "builded_nodupdataset.json" if c18 else "builded_dataset.json"
    
    with open(ds, mode='r') as f:  
        dataset = json.load(f)
    preprocessed = preprocessingDataset(dataset, normalize2=False, is_scale=False) 
    
    with open(ds2, mode='r') as f:
        y_all = json.load(f)
        
    y_all = y_all[0]["target"]    
    predictAll(preprocessed, y_all, "count", "regression", test_size=0.33, random_state=15, is_string=False, blind_test_also=True, testfile="blindtest.json")
            
    