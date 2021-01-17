from nltk.probability import FreqDist
import sys
import time

def word_counter2(l, cl = 10):
    """
    type(l) = list()
    type(cl) = int() (default = 10)
    
    printing [cl] most_common words and returning BoW, SoW dictionaries
    """
    BoW, SoW = {}, {}
    all_words = []
    wc = 0

    time.sleep(1)
    for rev, sentiment in l:
        sys.stdout.write(f'\rIlosc wszystkich występujących słów:\t{wc}')
        sys.stdout.flush()
        for word in rev:
            wc += 1
            all_words.append(word)
    
    BoW = FreqDist(all_words)
    SoW = set(BoW)
    print(f'\nIlosc unikatowych słów:\t{len(SoW)}\n')
    
    time.sleep(1)
    BoW.plot(cl,title=f'Rozkład występowania {cl} najpopularniejszych słów')
    
    return all_words, BoW, SoW
    
    
    
    
    
    
    
    