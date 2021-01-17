from modules.part_A import load_csv2 as lcsv

a = []
a, p_len, n_len = lcsv("movies_data.csv", a, lim = 1020) 
# daję 1020 żeby później operować na 500 pos i 500 neg. Finalnie ustawi się 0 => całosć

print(f"\nIlosć recenzji:\nPozytywnych: {p_len}\nNegatywnych: {n_len}")
if p_len == n_len: print("Dane są zbalansowane.")
else: print("Dane nie są zbalansowane.")
