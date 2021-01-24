from modules.part_A import load_csv2 as lcsv
from modules.part_B import word_counter2 as wc
from modules.part_C import NaivB as NB
from modules.part_C import Repeat as Rpt

############################## PART_A ####################################

# tworzę liste krotek "a" i liste wszystkich słów "allwords"
a, p_len, n_len, allwords = lcsv("movies_data.csv")#, lim = 1018) 
# daję 1018 żeby później operować na ok.500 pos i 500 neg. Finalnie ustawi się 0 => całosć

print(f"\n\nIlosć recenzji:\nPozytywnych: {p_len}\nNegatywnych: {n_len}")
if p_len == n_len: print("Dane są zbalansowane.\n")
else: print("Dane nie są zbalansowane!\n")

############################## PART_B ####################################

# tworzę BagOfWords i wypusuję wartosci Bow,Sow + plot 40 most_common
bow = wc(allwords, cl = 40, out=False)

############################## PART_C ####################################

NBacc = NB(a, bow, k=3000, outacc=True, n=15)
Rpt(a, bow, n=10, ls_acc=True)
