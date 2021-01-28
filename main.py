##########################################################################
#                                                                        #
#                             NLP-FINAL                                  #
#       which is the last winter semester project of the NLP course,     #
#         student history from 2020 and 2021 in four python parts.       #
#   Authors:                                                             #
#       Piotr Szulc                                                      #
#       Magdalena Lipka                                                  #
#                                                                        #
##########################################################################


############################## PART_A ####################################
from modules.part_A import load_csv2 as lcsv

# tworzę liste krotek "a" i liste wszystkich słów "allwords"
a, p_len, n_len, allwords = lcsv("movies_data.csv", lim = 1018) 
# daję 1018 żeby później operować na ok.500 pos i 500 neg. Finalnie ustawi się 0 => całosć

print(f"\n\nNumber of reviews:\nPositive: {p_len}\nNegative: {n_len}")
if p_len == n_len: print("The data is "+"\x1b[6;30;42m"+"balanced.\n"+'\x1b[0m')
else: print("The data is "+'\x1b[7;30;41m'+"not balanced!\n"+"\x1b[0m")

############################## PART_B ####################################
from modules.part_B import word_counter2 as wc

# tworzę BagOfWords i wypusuję wartosci Bow,Sow + plot 40 most_common
bow = wc(allwords, cl = 40, out=False)

############################## PART_C ####################################
from modules.part_C import NaivB1 as NB1
from modules.part_C import Cmain_f as Cf

# odpalam raz Naive Bayes do wypisania najważniejszych słów
NB1(a, bow, outacc=True, mostif=True, mif=15)

# odpalam serię wszystkich metod po 10 powtórzeń, zwraca listę wyników.
accs = Cf(a, bow, repeats=10, auto=True)

############################## PART_D ####################################
from modules.part_D import bow_and_web, get_vectors, train_network, get_accuracy

#odpalam przejście analizy opartej o BoW
BoWmodel_params = bow_and_web(a, TrSet=0.8, lr=0.1, n_iters=100)
print(f'\n> {BoWmodel_params[0]}\n> {BoWmodel_params[1]}')

import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=200)
train, valid, test = get_vectors(glove, a) #przygotowuje sobie dane w oparciu o gotowe embeddingi z glove

import torch
#za kazdym razem będzie bral 200 rekordow (przy trenowaniu) i co epoke tasujemy (shuffle = True)
train_loader = torch.utils.data.DataLoader(train, batch_size=200, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=200, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=200, shuffle=True)

import torch.nn as nn
siec = nn.Sequential(nn.Linear(200, 50),  #przekształcenie liniowe R^200 ---> R^30
                        nn.ReLU(),          #przekształcenie ReLU
                        nn.Linear(50, 10),  #kolejne przekształcenie liniowe R^50--->R^10
                        nn.ReLU(),          #przekształcenie ReLU
                        nn.Linear(10, 2))   # przekształcenie liniowe, efekt: 2 liczby
                        
train_network(siec, train_loader, valid_loader, test_loader, num_epochs=100, learning_rate=0.0001)

print("Final test accuracy:", get_accuracy(siec, test_loader)) #dokladnosc na zbiorze testowym
