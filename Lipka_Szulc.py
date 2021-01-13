'''
No siema :D
Ogólnie rzecz biorąc jest trochę do roboty więc ja bym nie zwlekał tylko leciał z tematem.
Mamy do zrobienia:
    Preprocesing:
        Przygotowanie i oczyszczenie
        Zbalansowanie danych (jesli Neg != Pos)
    
    Analiza wstępna tekstu:
        Ilosć słów (Bag of Words)
        Ilosć unikatowych (Set of Words)
        40 najpopularniejszych słów (most_common)
    
    Analiza algorytmami klasycznymi (Trening_set[80%], Test_set[20%]):
        Naive Bayes:
            + 15 najważniejszych słów
        Regresja logistyczna
        SVM ???
        Powyższe trzy razem ???
        +
        Średnia dokładnosć każdej z metod:
            10 powtórzeń budowania modelu
    
    Analiza o neuronowe:
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
            
            Zaprezentować graficznie (pewnie te dwa wykresy co ostatnio)
            
PS.: Z tego niewiele póki co rozumiem ale wkleje żeby było :D
~ Adi napisał:
"
Przy wykorzystaniu LSTM/GRU proszę pamiętać o kwestii batchowania w sieciach rekurencyjnych
(recenzje zazwyczaj są różnej długości, a z drugiej strony wymagany jest tensor o zadanej długości).
Możliwe rozwiązania: pad_sequence, TBatcher, zastosowanie BucketIterator. 
Ten ostatni ma następującą składnie:
    
train_iter = torchtext.data.BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.txt), sort_within_batch=True, repeat=False).
"


Organizacyjnie:
    plik z embeddigami przeniosłem sobie [żeby nie kopiować GB!] do folderu z plikiem Lipka_Szulc.py
    i jako work_directory mam też folder z tym plikiem czyli:
        ...\Desktop\7semestr\nlp\projekt
    Mam nadzieję, że Ci to będzie odpowiadać, jesli nie to dawaj znać, to cos ogarniemy ^^
    
    Pluuus... co powiesz na współpracę przez gita? Możnaby wtedy działać równoczesnie ale to 
    trochę ogarniania jest wiec jak nie chcesz to jakos damy rade zwykłym ctrl+c, ctrl+v :D


'''
























































