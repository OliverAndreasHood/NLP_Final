# Analiza algorytmami klasycznymi

import numpy as np
import nltk
import random
import sys

#1.Naive Bayes + 15 najważniejszych słów; TrS=.8 TsS=0.2

def NaivB(revs, BoW, k=3000, TrSet=0.8, outacc=False, mostif=False, n=10):
    """
    revs => type==list, parsed input data as [([words], statement), ([...],.), ...]
    Bow => type==probability.FreqDist, BagOfWords representation of revs
    k => type==int
    TrSet => type==float default 0.8, size of Trening_Set. Testing_set = 1 - TrSet    
    out => type==bool default = False, to print out NB_acc.
    mostif => type==bool default False, print n most informative futures from Naive Bayes Classifier 
    n => type==int default 10, number of most informative features to print, have to be greater than 0.
    
    """
    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features
    
    word_features = list(BoW.keys())[:k]
    random.shuffle(revs)
    TrSet = int(len(revs)*TrSet)
    
    featuresets = [(find_features(rev),category) for (rev,category) in revs]
    
    training_set = featuresets[:TrSet]
    testing_set = featuresets[TrSet:]
    
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    acc = (nltk.classify.accuracy(classifier,testing_set))*100
    
    if outacc: print(f"Dokładność metody Naive Bayes do problemu klasfyikacji na zbiorze testowym wynosi: {acc:.2f}") 
    if mostif and n>0: classifier.show_most_informative_features(n)
    
    return acc

#2. Regresja logistyczna
    
def LogRegr(revs):

    return acc

#3. SVM ???

def SvmF(revs):
    
    return acc

#4. Powyższe trzy razem ??? Póki co nie mam pomysło jak się za to wziąć :(

def AllF(revs):
    
    return acc

    
#Średnia dokładnosć każdej z metod:
#10 powtórzeń budowania modelu

def Repeat(l, BoW, n=2, a=False, ls_acc=False):
    """
    l => type==list, input data as [([words], statement), ([...],.), ...]
    Bow => type==probability.FreqDist, BagOfWords representation of l, 
    n => type==int default 0, repeats number of choosen method
    a => type==bool default False, repeat all methods autmoaticly
    ls_acc => type==bool default False, list acc of each one repeat.
    """
    whole_accs = []
    if a:
        print("Repeats automaticly for 4 methods.")
        func = "AUTO"
    else:
        func = input("#Runing method repeats#\nChoose one following methods:\n1 => Naive Bayes\n2 => Logic Regresion\n3 => SVM\n4 => all above\n> ")
        methods = ['1','2','3','4']
        if func in methods:
            if func == "1": func = "NB"
            elif func == "2": func = "LR"
            elif func == "3": func = "SVM"
            elif func == "4": func = "ALL"
        else:
            print("Wrong input!")
            return False
    
    
    if func == "NB" or func == "AUTO":
        NBacc_list = []
        i=n
        print(f"\nRuning Naive Bayes {n} repeats..")
        while i:
            NBacc_list.append(NaivB(l, BoW))
            if ls_acc: print(f'Run No.{n-i+1} acc: {NBacc_list[n-i]:.2f}')
            i-=1
        NBacc = np.mean(NBacc_list)
        
        print(f'\nNaive Bayes method mean accuracy in {n} repeats: {NBacc:.2f}\n')
    
    if func == "LR" or func == "AUTO":
        LRacc_list = []
        i=n
        while i:
            LRacc_list.append(LogRegr(l))
            if ls_acc: print(f'Run No.{n-i+1} acc: {NBacc_list[n-i]:.2f}')
            i-=1
            
        LRacc = np.mean(LRacc_list)
        print(f'Logic Regresion method mean accuracy in {n} repeats: {LRacc}')
    
    if func == "SVM" or func == "AUTO":
        SVMacc_list = []
        i=n
        while i:
            SVMacc_list.append(SvmF(l))
            if ls_acc: print(f'Run No.{n-i+1} acc: {NBacc_list[n-i]:.2f}')
            i-=1

        SVMacc = np.mean(SVMacc_list)
        print(f'SVM method mean accuracy in {n} repeats: {SVMacc}')
    
    if func == "ALL" or func == "AUTO":
        ALLacc_list = []
        i=n
        while i:
            ALLacc_list.append(AllF(l))
            if ls_acc: print(f'Run No.{n-i+1} acc: {ALLacc_list[n-i]:.2f}')
            i-=1

        ALLacc = np.mean(ALLacc_list)
        print(f'All method mean accuracy in {n} repeats: {ALLacc}')
    
    
    return whole_accs
















