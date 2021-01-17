from modules.part_A import load_csv2 as lcsv
from modules.part_B import word_counter2 as wc

############################## PART_A ####################################
a, p_len, n_len, allword = lcsv("movies_data.csv", lim = 1020) 
# daję 1020 żeby później operować na 500 pos i 500 neg. Finalnie ustawi się 0 => całosć

print(f"\n\nIlosć recenzji:\nPozytywnych: {p_len}\nNegatywnych: {n_len}")
if p_len == n_len: print("Dane są zbalansowane.\n")
else: print("Dane nie są zbalansowane!\n")

############################## PART_B ####################################

bow = wc(allword, cl = 40)

############################## PART_C ####################################




