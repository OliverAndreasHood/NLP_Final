# NLP_Final

Mamy do zrobienia:

    A. Preprocesing: (DONE!) ?Lemmatyzacja?
        1. Przygotowanie i oczyszczenie
        2. Zbalansowanie danych (jesli Neg != Pos)
    
    B. Analiza wstępna tekstu: (DONE!)
        1. Ilosć słów (Bag of Words)
        2. Ilosć unikatowych (Set of Words)
        3. 40 najpopularniejszych słów (most_common)
    
    C. Analiza algorytmami klasycznymi (Trening_set[80%], Test_set[20%]): (DONE!)
        1. Naive Bayes:
            + 15 najważniejszych słów
        2. Regresja logistyczna
        3. SVM
        4. Powyższe trzy razem
        +
        Średnia dokładnosć każdej z metod: (DONE!)
            10 powtórzeń budowania modelu
    
    D. Analiza o neuronowe:
        "zbuduj prostą sieć xD":
        1. Reprezentacja (Bag of Words) i jedno przekształcenie liniowe
            Trening_set = 80%
            Test_set = 20%
        2. Embeddingi 200D {200 wymiarów badabum plask blee}:
            Każda recenzja jako suma/srednia aryt. jej embeddingów.
        3. Embeddingi 200D + LSTM(?) lub/oraz GRU(?)
        
        Ad 2 i 3:
            Trening_set = 70%
            Valid_set = 15%
            Test_set = 15%
            Różne:
                przekształcenia
                f. aktywacji
                epoch
                lrng_rate
                optimizer (jesli trzeba)
                batch [!] - ta wielkosć podzbioru wybieranego z review_setów
            + Zaprezentować graficznie
