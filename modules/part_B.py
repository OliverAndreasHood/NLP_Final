from nltk.probability import FreqDist
import time

def word_counter2(alw, out=False, cl = 10):
    """
    alw => type==list, list of words to count
    out => type==bool default = False, to print out BoW, SoW and most_common_words plot.
    cl => type==int default = 10, number of most_common_words to plot.
    
    if out:
        printing BagOfWords lenght
        printing SetOfWords lenght
        ploting [cl] most_common words 
    returning BoW dict, type==probability.FreqDist 
    """
    if out: print(f'Ilosc wszystkich słów:\t{len(alw)}')
    BoW = FreqDist(alw)
    if out: print(f'Ilosc unikatowych słów:\t{len(set(BoW))}\n')
    
    time.sleep(1)
    if out: BoW.plot(cl,title=f'Rozkład występowania {cl} najpopularniejszych słów')
    
    return BoW
    


    
    
    
    
    
    