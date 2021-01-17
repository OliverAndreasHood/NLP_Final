from nltk.probability import FreqDist
import time

def word_counter2(alw, cl = 10):
    """
    type(l) = list()
    type(cl) = int() (default = 10)
    
    printing SetOfWords [set(BoW)]
    ploting [cl] most_common words 
    returning all_words list and BoW probability.FreqDist dict
    """
    BoW = FreqDist(alw)
    print(f'Ilosc wszystkich słów:\t{len(alw)}')
    SoW = set(BoW)
    print(f'Ilosc unikatowych słów:\t{len(SoW)}\n')
    
    time.sleep(1)
    BoW.plot(cl,title=f'Rozkład występowania {cl} najpopularniejszych słów')
    
    return BoW
    


    
    
    
    
    
    