##########################################################################
#                                                                        #
#                             NLP-FINAL                                  #
#       which is the last winter semester project of the NLP course,     #
#        student history from 2020 and 2021 in seven python parts.       #
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
accs = Cf(a, bow, repeats=5, auto=False, func="NB")

############################### PART_D ####################################
from modules.part_D1 import bow_and_web

            ##################### 1 #####################
#odpalam przejście analizy opartej o BoW (1xLinear)
BoWmodel_params = bow_and_web(a, TrSet=0.8, lr=0.1, n_iters=10)
print(f'\n> {BoWmodel_params[0]}\n> {BoWmodel_params[1]}')

            ##################### 2 #####################
from modules.part_D2 import get_vectors, train_network
from modules.ff import get_accuracy
import torch.nn as nn
import torchtext

#przygotowujemy baze embeddingów GloVe
glove = torchtext.vocab.GloVe(name="6B", dim=200)

#przygotowuje sobie dane w oparciu o gotowe embeddingi z glove
train_l, valid_l, test_l = get_vectors(glove, a, batch_size=200, shuffle=True) 

#definiujemy strukturę modelu
siec = nn.Sequential(nn.Linear(200, 50),
                        nn.ReLU(),
                        nn.Linear(50, 10),
                        nn.ReLU(),
                        nn.Linear(10, 2))
                   
#odpalam przejcie analizy opartej o embeddingi GloVe 200D     
train_network(siec, train_l, valid_l, test_l, num_epochs=100, learning_rate=0.0001, pltout=False)
print("Final test accuracy:", get_accuracy(siec, test_l)) #dokladnosc na zbiorze testowym

            ##################### 3 #####################





















