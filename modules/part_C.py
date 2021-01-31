# Analiza algorytmami klasycznymi
import numpy as np
import random
import nltk
import sys
import time
from modules.ff import find_features
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode

##########################1.Naive Bayes + 15 most informative words ###########

def NaivB1(revs, BoW, lim=3000, TrSet=0.8, outacc=False, mostif=False, mif=10):
    """
    revs => type==list, parsed input data as [([words], statement), ([...],.), ...]
    Bow => type==probability.FreqDist, BagOfWords representation of l, 
    lim => type==int. word_features cut-off 
    TrSet => type==float default 0.8, size of Trening_Set. Testing_set = 1 - TrSet    
    outacc => type==bool default = False, print out Naive Bayes acc.
    mostif => type==bool default False, print n most informative futures from Naive Bayes Classifier 
    mif => type==int default 10, number of most informative features to print, have to be greater than 0.
    """
    start = time.time()
    print('Classic Naive Bayes algorythm\n')
   
    sys.stdout.write("\r=> Preparing.. ")
    wf = list(BoW.keys())[:lim]
    random.shuffle(revs)
    TrSet = int(len(revs)*TrSet)
    
    featuresets = [(find_features(rev, wf),category) for (rev,category) in revs]
    
    training_set = featuresets[:TrSet]
    testing_set = featuresets[TrSet:]
    sys.stdout.write(f"\rPreparing time {time.time()-start:.2f} s\n")
    
    start = time.time()
    sys.stdout.write("\r=> Training..  ")
    NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
    sys.stdout.write(f"\rTraining time  {time.time()-start:.2f} s\n")
    
    start = time.time()
    sys.stdout.write("\r=> Testing..  ")
    acc = (nltk.classify.accuracy(NB_classifier,testing_set))*100
    sys.stdout.write(f"\rTesting time  {time.time()-start:.2f} s\n")
    time.sleep(1)    
    
    if outacc: print(f"\n> The accuracy of the MNB Naive Bayes method to the classification problem on the test set is: {acc:.2f}") 
    time.sleep(2)    
    if mostif and mif>0: NB_classifier.show_most_informative_features(mif)
    return 

def NaivB2(training_set, testing_set):
      
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    acc = (nltk.classify.accuracy(MNB_classifier,testing_set))*100
        
    return acc

##########################2. Logistic Regression ##############################

def LogRegr(training_set, testing_set):
    
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='lbfgs', max_iter=300))
    LogisticRegression_classifier.train(training_set)
    acc = (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100

    return acc

##########################3. SVM ##############################################

def SvmF(training_set, testing_set):
     
    LinearSVC_classifier = SklearnClassifier(LinearSVC(max_iter=2000))
    LinearSVC_classifier.train(training_set)
    acc = (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100
        
    return acc

##########################4. Aggregated all above #############################

class AggClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

def AllF(training_set, testing_set):

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='lbfgs', max_iter=300))
    LogisticRegression_classifier.train(training_set)

    LinearSVC_classifier = SklearnClassifier(LinearSVC(max_iter=2000))
    LinearSVC_classifier.train(training_set)
    
    agg_classifier = AggClassifier(MNB_classifier,
                                   LogisticRegression_classifier,
                                   LinearSVC_classifier)
    
    acc = (nltk.classify.accuracy(agg_classifier, testing_set))*100
        
    return acc

    
#Średnia dokładnosć każdej z metod:
#10 powtórzeń budowania modelu

def Cmain_f(revs, BoW, lim=3000, TrSet=0.8, repeats=2, func=None, auto=False, ls_acc=True):
    """
    revs => type==list, parsed input data as [([words], statement), ([...],.), ...]
    Bow => type==probability.FreqDist, BagOfWords representation of l, 
    lim => type==int. word_features cut-off 
    TrSet => type==float default 0.8, size of Trening_Set. Testing_set = 1 - TrSet    
    repeats => type==int default 0, repeats number of choosen method
    func => type==str default None, type of algorythm to run automaticly ["NB","LR","SVM","ALL"]
    auto => type==bool default False, repeat all methods autmoaticly
    ls_acc => type==bool default True, list acc of each one repeat.
    """
    print("\nC_main reading input config...")
    time.sleep(1)
    
    #końcowa tabela zbiorcza
    accs = []
    
    if auto:
        print("Proces will repeat automaticly for all methods.")
        func = "AUTO"
        time.sleep(1)
    else:
        if func==None:
            x = input("#Runing method repeats#\nChoose one following methods:\n1 => Naive Bayes\n2 => Logistic Regresion\n3 => Linear SVM\n4 => Aggregated all above\n> ")
            methods = ['1','2','3','4']
            if x in methods:
                if x == "1": func = "NB"
                elif x == "2": func = "LR"
                elif x == "3": func = "SVM"
                elif x == "4": func = "ALL"
            else:
                print("Wrong input!")
                return False    
    
    sys.stdout.write("\rCreating Data Sets => Preparing.. ")
    wf = list(BoW.keys())[:lim]
    TrSet = int(len(revs)*TrSet)
    random.shuffle(revs)
    
    sys.stdout.write("\rCreating Data Sets => Indexing..  ")
    featuresets = [(find_features(rev,wf),category) for (rev,category) in revs]
    sys.stdout.write("\rCreating Data Sets => Done         \n")
    
    if func == "NB" or func == "AUTO":
        NBacc_list = []
        i=repeats
        print(f'\nRuning MNB Naive Bayes {repeats} repeats..')
        while i:
            random.shuffle(featuresets)
            training_set = featuresets[:TrSet]
            testing_set = featuresets[TrSet:]
            sys.stdout.write(f"\rRun No.{repeats-i+1} ")
            
            NBacc_list.append(NaivB2(training_set, testing_set))
            if ls_acc: print(f'acc: {NBacc_list[repeats-i]:.2f}')
            i-=1
                
        NBacc = np.mean(NBacc_list)
        print(f'MNB Naive Bayes method mean accuracy in {repeats} repeats: {NBacc:.4f}\n')
        accs.append(("NB", NBacc, NBacc_list))
        time.sleep(1)
        
    if func == "LR" or func == "AUTO":
        LRacc_list = []
        i=repeats
        print(f"\nRuning Logistic Regression {repeats} repeats..")
        while i:
            random.shuffle(featuresets)
            training_set = featuresets[:TrSet]
            testing_set = featuresets[TrSet:]
            sys.stdout.write(f"\rRun No.{repeats-i+1} ")
            
            LRacc_list.append(LogRegr(training_set, testing_set))
            if ls_acc: print(f'acc: {LRacc_list[repeats-i]:.2f}')
            i-=1
            
        LRacc = np.mean(LRacc_list)
        print(f'Logistic Regression method mean accuracy in {repeats} repeats: {LRacc:.4f}\n')
        accs.append(("LR", LRacc, LRacc_list))
        time.sleep(1)

    if func == "SVM" or func == "AUTO":
        SVMacc_list = []
        i=repeats
        print(f"\nRuning Linear SVM {repeats} repeats..")
        while i:
            random.shuffle(featuresets)
            training_set = featuresets[:TrSet]
            testing_set = featuresets[TrSet:]
            sys.stdout.write(f"\rRun No.{repeats-i+1} ")
            
            SVMacc_list.append(SvmF(training_set, testing_set))
            if ls_acc: print(f'acc: {SVMacc_list[repeats-i]:.2f}')
            i-=1

        SVMacc = np.mean(SVMacc_list)
        print(f'Linear SVM method mean accuracy in {repeats} repeats: {SVMacc:.4f}\n')
        accs.append(("SVM", SVMacc, SVMacc_list))
        time.sleep(1)

    if func == "ALL" or func == "AUTO":
        ALLacc_list = []
        i=repeats
        print(f"\nRuning Aggregated classifier {repeats} repeats..")
        while i:
            random.shuffle(featuresets)
            training_set = featuresets[:TrSet]
            testing_set = featuresets[TrSet:]
            sys.stdout.write(f"\rRun No.{repeats-i+1} ")
            
            ALLacc_list.append(AllF(training_set, testing_set))
            if ls_acc: print(f'acc: {ALLacc_list[repeats-i]:.2f}')
            i-=1

        ALLacc = np.mean(ALLacc_list)
        print(f'Aggregated classifier method mean accuracy in {repeats} repeats: {ALLacc:.4f}\n')
        accs.append(("ALL", ALLacc, ALLacc_list))
        time.sleep(1)



    return accs
















