import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
import string 
import sys
import time

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
        percent = 0
        
        for row in csv_reader:
            sys.stdout.write('\rloading data' + f'\t{percent:.1f}%\t ({line_count})')
            sys.stdout.flush()
               
            if line_count == 0:
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
            if lim == 0:
                percent = (line_count)/50000*100
            else:
                percent = (line_count)/lim*100
            
            if line_count == lim+1:
                break
            time.sleep(0.0001)
            
        l = pos + neg
        sys.stdout.write('\rDone!       ')
    return l, len(pos), len(neg)

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