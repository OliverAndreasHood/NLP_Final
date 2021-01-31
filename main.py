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
a, p_len, n_len, allwords = lcsv("movies_data.csv", lim = 15000) 
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
accs = Cf(a, bow, repeats=10, auto=False, func="NB")

############################### PART_D ####################################
from modules.part_D1 import bow_and_web

            ##################### 1 #####################
#odpalam przejście analizy opartej o BoW (1xLinear)
BoWmodel_params = bow_and_web(a, TrSet=0.8, lr=0.1, n_iters=10)
print(f'\n> {BoWmodel_params[0]}\n> {BoWmodel_params[1]}\n')

            ##################### 2 #####################
from modules.part_D2 import get_vectors
from modules.part_D2 import train_network
import torch.nn as nn
import torchtext

#przygotowujemy baze embeddingów GloVe
glove = torchtext.vocab.GloVe(name="6B", dim=200)

#przygotowuje sobie dane w oparciu o gotowe embeddingi z glove
train_l, valid_l, test_l = get_vectors(glove, a, batch_size=312, shuffle=True) 

#definiujemy strukturę modelu
siec = nn.Sequential(nn.Linear(200, 20),
                        nn.ReLU(),
                        nn.Linear(20, 2))
                   
#odpalam przejcie analizy opartej o embeddingi GloVe 200D     
train_network(siec, train_l, valid_l, test_l, num_epochs=100, learning_rate=2e-4, pltout=True)


            #####################  3  ######################
from modules.part_D3 import get_indx
from modules.part_D3 import TBatcher

train, valid, test = get_indx(glove, a)

batch_size = 32
train_loader = TBatcher(train, batch_size=batch_size, drop_last=True)  #dane treningowe z batchem
valid_loader = TBatcher(valid, batch_size=batch_size, drop_last=False)  #dane walidacyjne z batchem
test_loader = TBatcher(test, batch_size=batch_size, drop_last=False)  #dane testowe z batchem

                ################# LSTM #################
from modules.part_D3 import T_LSTM
from modules.part_D3 import md_train

print('\nMaking LSTM model')
model_lstm = T_LSTM(200, 5, 2)
md_train(model_lstm, train_loader, valid_loader, test_loader, num_epochs=15, learning_rate=1e-5, pltout=True)

                ################# GRU #################
from modules.part_D3 import T_GRU

print('\nMaking GRU model')
model_gru = T_GRU(200, 5, 2)
md_train(model_gru, train_loader, valid_loader, test_loader, num_epochs=15, learning_rate=3e-5, pltout=True) 
