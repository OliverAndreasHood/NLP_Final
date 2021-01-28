#D. Analiza o neuronowe:

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

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt

#    1. Reprezentacja (Bag of Words) i jedno przekształcenie liniowe
#        Trening_set = 80%
#        Test_set = 20%


#do cwiczenia

revs = [ (['me', 'gusta', 'comer', 'en', 'la', 'cafeteria'], 0),
         (['Give', 'it', 'to', 'me'], 1),
         (['No', 'creo', 'que', 'sea', 'una', 'buena', 'idea'], 0),
         (['No', 'it', 'is', 'not', 'a', 'good', 'idea', 'to', 'get', 'lost', 'at', 'sea'], 1),
         (['Yo', 'creo', 'que', 'si'], 0),
         (['it', 'is', 'lost', 'for', 'me'], 1),
         (['it', 'is', 'lost', 'for', 'me'], 1),
         (['it', 'is', 'lost', 'for', 'me'], 1),
         (['it', 'is', 'lost', 'for', 'me'], 1),
         (['it', 'is', 'lost', 'for', 'me'], 1)]


def bow_and_web(revs, TrSet=0.8, lr=0.1, n_iters=100):
    """
    revs => type==list, parsed input data as [([words], statement), ([...],.), ...]
    TrSet => type==float default 0.8, size of Trening_Set. Testing_set = 1 - TrSet 
    lr => type==float default 0.1, model learning rate
    n_iters => type==int default 100, number of training iterations
    """
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

    print('Neural BoW analyse\n')
    label_to_ix = {0: 0, 1: 1 }
    word_to_ix = {} #zbior slow z indywidualna liczba

    sys.stdout.write('\r=> Preparing..')
    start = time.time()
    TrSet = int(len(revs)*TrSet)
    training_set=revs[:TrSet]
    test_set=revs[TrSet:]
    sys.stdout.write(f"\rPreparing time {time.time()-start:.2f} s\n")


    sys.stdout.write('\r=> Indexing..')
    start = time.time()
    for sent, _ in training_set + test_set:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    sys.stdout.write(f"\rIndexing time {time.time()-start:.2f} s\n")

    VOCAB_SIZE = len(word_to_ix)  # ile wszysktich slow
    NUM_LABELS = len(label_to_ix) # ile kategorii
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Category amount: {NUM_LABELS}\n")

    #parametry modelu, funkcji kosztu i optymalizator
    model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    print(f'Model parameters:\n> Training_set size: {len(training_set)} | Testing_set size: {len(test_set)}\n> Learning Rate: {lr} | Iterations: {n_iters}')

    print('\nTraining BoWClassifier model..')

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

        if epoch % int(n_iters/5) == 0:
            print(f'Epoch number: {epoch+1} \t| Loss value: {loss}')
    print('> Done')

    for instance, label in test_set:
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vec)

    return (list(model.parameters()))


#    2. Embeddingi 200D {200 wymiarów }:
#        Każda recenzja jako suma/srednia aryt. jej embeddingów.

def get_vectors(glove_vector, revs):
    train, valid, test = [], [], [] #tworze trzy listy na dane train, valid i test
    for i, line in enumerate(revs): #przechodze dane 
        rev = line[0]                 
    
        rev_emb = sum(glove_vector[w] for w in rev) 
        label = torch.tensor(int(line[1] == 1)).long() 
            
        #dzielimy dane na trzy kategorie
        if i % len(revs) < int(0.7*len(revs)):     
            train.append((rev_emb, label)) 
        elif i == int(0.7*len(revs)):
            valid.append((rev_emb, label))
        elif i > int(0.7*len(revs)) and i < int(0.85*len(revs)):
            valid.append((rev_emb, label))
        else:            
            test.append((rev_emb, label)) 
            
    return train, valid, test


#trenowanie
def train_network(model, train_loader, valid_loader, test_loader, num_epochs=5, learning_rate=1e-5, pltout=True):
    print('Neural Glove analyse\n')
    criterion = nn.CrossEntropyLoss()  #funkcja kosztu
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    
    print(f'Model parameters:\n> Training_set size: {len(train_loader.dataset)} | Valid_set size: {len(valid_loader.dataset)} | Testing_set size: {len(test_loader.dataset)}\
\n> Learning Rate: {learning_rate} | Iterations: {num_epochs}')
    time.sleep(1)

    for epoch in range(num_epochs):          #dla kazdej epoki
        for tweets, labels in train_loader:  #przechodze dane treningowe
            optimizer.zero_grad()
            pred = model(tweets)         
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))           #zapisuje wartosc funkcji kosztu
        
        
        if epoch % int(num_epochs/5) == 0:                   
            epochs.append(epoch)             
            train_acc.append(get_accuracy(model, train_loader))   #dokladnosc na zbiorze treningowym
            valid_acc.append(get_accuracy(model, valid_loader))   #dokladnosc na zbiorze walidacyjnym
            print(f'Epoch number: {epoch+1} | Loss value: {loss} | Train accuracy: {round(train_acc[-1],3)} | Valid accuracy: {round(valid_acc[-1],3)}')
  
    #Wykresy
    if pltout:
      plt.title("Training Curve")
      plt.plot(losses, label="Train dataset")
      plt.xlabel("Epoch number")
      plt.ylabel("Loss value")
      plt.show()

      plt.title("Training Curve")
      plt.plot(epochs, train_acc, label="Train dataset")
      plt.plot(epochs, valid_acc, label="Validation dataset")
      plt.xlabel("Epoch number")
      plt.ylabel("Accuracy")
      plt.legend(loc='best')
      plt.show()

#Funkcja wyznaczająca dokładność predykcji:
def get_accuracy(model, data_loader):
    correct, total = 0, 0  #ile ok, ile wszystkich
    for revs, labels in data_loader: #przechodzi dane
        output = model(revs)         #jak dziala model
        pred = output.max(1, keepdim=True)[1]  #ktora kategoria
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total
