#D. Analiza o neuronowe:
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

############# 1. Reprezentacja BoW i jedno przekształcenie liniowe ###########

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
    start = time.time()
    for epoch in range(n_iters):
        for instance, label in training_set:
            sys.stdout.write(f'\rEpoch number: {epoch+1}/{n_iters}')
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

        sys.stdout.write(f'\rEpoch number: {epoch+1} \t| Loss value: {loss:.4f}\n')
    sys.stdout.write(f'\r> Done ({time.time()-start:.2f} s)                  \n')

    for instance, label in test_set:
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vec)

    return (list(model.parameters()))


