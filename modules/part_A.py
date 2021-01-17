import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
import string 

def load_csv2(file_path, l, lim = 0):
    """
    type(file_path) = str()
    type(l) = list()
    lim = 0 (default) means parse whole file
    csv file with 2 columns separated with comma
    Loading function. Appending data to l as [row[0],row[1]]
    """
    #ładowanie pliku z podziałem na słowa i wykluczeniem stopwords, digit, punctuation
    with open(file_path, "r", encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        sw =  set(stopwords.words("english"))
        line_count = 0
        pos, neg = [], []
        
        for row in csv_reader:   
            if line_count == 0:
                print(f'Column names are {", ".join(row)}\n')
                line_count += 1
            else:
                row[0] = row[0].replace("<br /><br />", " ")
                row[0] = word_tokenize(row[0])
                    
                filtred = []
                for word in row[0]:
                    if word not in sw and word not in string.punctuation:
                        if word.isalpha() == True:
                            filtred.append(word.rstrip().lower())
            
                if int(row[1]) == 0:
                    neg.append([filtred, int(row[1])])
                else:
                    pos.append([filtred, int(row[1])])
                line_count += 1
        
            
            # Ogranicznik parsowania
            if lim != 0:
                percent = (line_count)/lim*100
                if percent%5 == 0 and line_count != 0:
                    print(f'Parsed {percent:.1f}% ({line_count}) rows of {file_path}')
            
                if line_count == lim+1:
                    break
            else:
                percent = (line_count)/50000*100
                if percent%5 == 0 and line_count != 0:
                    print(f'Parsed {percent:.1f}% ({line_count}) rows of {file_path}')

                
        l = pos + neg
    print('Done\n')
    return l

def pn_compare(l):
    """
    Returns True if positive reviews amount equals negative rewievs amount
    else returns False
    """
    pos, neg = 0, 0
    for rev in l:
        if rev[1] == 0:
            neg += 1
        else:
            pos += 1
    print(f"Ilosć recenzji:\nPozytywnych: {pos}\nNegatywnych: {neg}")
    if pos == neg:
        print("Dane są zbalansowane.")
        return True
    else:
        print("Dane nie są zbalansowane.")
        return False



###############
# inna opcja ładowania danych ale ma problem ze znakami \ :
#
#reviews = []
#f = open("movies_data.csv", "r", encoding='utf-8')
#f.readline()
#for line in f:
#    line = line.replace("<br /><br />", " ")
#    line = line.replace("\n", "")
#    line = line.replace("\\'", "'")            # ale to nie działa :(
#    reviews.append([line[:-2],line[-1]])
#f.close()