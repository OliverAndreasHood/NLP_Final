import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
import sys
import time

def load_csv2(file_path, l = [], lim = 0):
    """
    type(file_path) = str()
    type(l) = list() (l = [] as default)
    lim = 0 (default) means parse whole file
    csv file with 2 columns separated with comma
    Loading function. Appending data to l as [row[0],row[1]]
    returns:
        l = [[row[0], row[1]], ...]
        len(pos) and len(neg)
        alw -> filtered all_words list
        
    """
    #ładowanie pliku z podziałem na słowa i wykluczeniem stopwords, digit, punctuation
    with open(file_path, "r", encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        sw =  set(stopwords.words("english"))
        line_count = 0
        pos, neg, alw = [], [], []
        percent = 0
        
        for row in csv_reader:
            sys.stdout.write(f'\rloading data' + f'\t{percent:.1f}%\t ({line_count})')
            sys.stdout.flush()
               
            if line_count == 0:
                line_count += 1
            else:
                row[0] = row[0].replace("<br /><br />", " ")
                row[0] = word_tokenize(row[0])
            
                filtred = []
                for word in row[0]:
                    word = word.rstrip().lower()
                    if word not in sw:
                        if word.isalpha() == True:
                            filtred.append(word)
                            alw.append(word)
                            
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
            
            if lim != 0 and line_count == lim+1:
                break
            time.sleep(0.001)
            
        l = pos + neg
        sys.stdout.write('\rDone!       ')
        time.sleep(1)
    return l, len(pos), len(neg), alw
