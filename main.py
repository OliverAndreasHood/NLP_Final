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
accs = Cf(a, bow, repeats=10, auto=True, ls_acc=True)

############################## PART_D ####################################
