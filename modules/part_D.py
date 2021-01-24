#D. Analiza o neuronowe:
#    "zbuduj prostą sieć xD":


#    3. Embeddingi 200D + LSTM(?) lub/oraz GRU(?)
#    
#    Ad 2 i 3:
#        Trening_set = 70%
#        Valid_set = 15%
#        Test_set = 15%
#        Różne:
#            przekształcenia
#            f. aktywacji
#            epoch
#            lrng_rate
#            optimizer (jesli trzeba)
#            batch [!] - ta wielkosć podzbioru wybieranego z review_setów
#        + Zaprezentować graficznie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchtext


#    1. Reprezentacja (Bag of Words) i jedno przekształcenie liniowe
#        Trening_set = 80%
#        Test_set = 20%


#do cwiczenia

revs = [ (['me', 'gusta', 'comer', 'en', 'la', 'cafeteria'], 0),
         (['Give', 'it', 'to', 'me'], 1),
         (['No', 'creo', 'que', 'sea', 'una', 'buena', 'idea'], 0),
         (['No', 'it', 'is', 'not', 'a', 'good', 'idea', 'to', 'get', 'lost', 'at', 'sea'], 1),
         (['Yo', 'creo', 'que', 'si'], 0),
         (['it', 'is', 'lost', 'for', 'me'], 1)]

'''
def bow_and_web(revs, TrSet=0.8):
    TrSet = int(len(revs)*TrSet)
    training_set=revs[:TrSet]
    test_set=revs[TrSet:]
    
    label_to_ix = {0: 0, 1: 1 }
    
    word_to_ix = {} #zbior slow z indywidualna liczba
    for sent, _ in training_set + test_set:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    VOCAB_SIZE = len(word_to_ix)  # ile wszysktich slow
    NUM_LABELS = len(label_to_ix) # ile kategorii

    #pomocnicza funkcja1

    def make_bow_vector(sentence, word_to_ix):
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            vec[word_to_ix[word]] += 1
            return vec.view(1, -1)
        
        #pomocnicza funkcja2

    def make_target(label, label_to_ix):
        return torch.LongTensor([label_to_ix[label]])
    
    #Model

    class BoWClassifier(nn.Module):  
        def __init__(self, num_labels, vocab_size):
            super().__init__()
            self.linear = nn.Linear(vocab_size, num_labels)
        
        def forward(self, bow_vec):
            return F.log_softmax(self.linear(bow_vec), dim = 1)
        
    model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    n_iters = 100
    for epoch in range(n_iters):
        for instance, label in training_set:     
            bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
            target = autograd.Variable(make_target(label, label_to_ix))
    
            #forward
            log_probs = model(bow_vec)
            loss = loss_function(log_probs, target)
        
            #backward
            loss.backward()
            optimizer.step()
        
            #zerujemy gradient
            optimizer.zero_grad()
    for instance, label in test_set:
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vec)

    return (list(model.parameters()))
'''
#nie jestem pewna jeszcze co i jak ladnie zwrocic



#    2. Embeddingi 200D {200 wymiarów }:
#        Każda recenzja jako suma/srednia aryt. jej embeddingów.


def embed(revs):
    embd=[]
    emd=[]
    em=[]
    glove = torchtext.vocab.GloVe(name="6B", dim=200)
    for item in revs:
        emd.append(item[0])
        for it in emd[0]:
    
            x=glove[it]

            em.append(x)
            em.append(it)
        embd.append(em)
        em=[]
        emd=[]

    print(embd)
            
            

