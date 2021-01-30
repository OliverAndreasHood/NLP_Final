#D. Analiza o neuronowe

import time
import sys
import torch
import random
import torchtext
import torch.nn as nn
import matplotlib.pyplot as plt
from modules.ff import get_accuracy

################ 3. Embeddingi 200D + LSTM lub/oraz GRU #######################

glove = torchtext.vocab.GloVe(name="6B", dim=200)

def get_indx(glove_vector, revs):
    sys.stdout.write('\rGetting index loaders..')    
    train, valid, test = [], [], [] 
    for i, line in enumerate(revs): 
        rev = line[0]                 

        idx = [glove_vector.stoi[w] for w in rev if w in glove_vector.stoi]
        if not idx:
          continue
        idx = torch.tensor(idx)

        label = torch.tensor(int(line[0] == 1)).long()

        #dzielimy dane na trzy kategorie
        if i % len(revs) < int(0.7*len(revs)):     
            train.append((idx, label)) 
        elif i == int(0.7*len(revs)):
            valid.append((idx, label))
        elif i > int(0.7*len(revs)) and i < int(0.85*len(revs)):
            valid.append((idx, label))
        else:            
            test.append((idx, label)) 
    
    sys.stdout.write('\rGetting index loaders => Done\n')
    return train, valid, test

############################## LSTM ###############################

class T_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors) #embeddingi
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) #LSTM
        self.fc = nn.Linear(hidden_size, num_classes)  #przeksztalcenie liniowe
    
    def forward(self, x):
        x = self.emb(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size) #początkowy  h0
        c0 = torch.zeros(1, x.size(0), self.hidden_size) #początkowy c0
        out, _ = self.lstm(x, (h0, c0))  #LSTM
        out = self.fc(out[:, -1, :]) #liniowe przekształcenie ostatniego outputu
        return out

############################## GRU ###############################

class T_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors) #embeddingi
        self.hidden_size = hidden_size 
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)  #GRU
        self.fc = nn.Linear(hidden_size, num_classes)   #przeksztalcenie liniowe
    
    def forward(self, x):
        x = self.emb(x)  #embeddingi
        h0 = torch.zeros(1, x.size(0), self.hidden_size) #początkowy stan ukryty
        out, _ = self.gru(x, h0)   #GRU
        out = self.fc(out[:, -1, :]) #ostatni output przeksztalcamy liniowo jeszcze
        return out

class TBatcher:
    def __init__(self, revs, batch_size=32, drop_last=False):
        self.revs_by_length = {} 
        for words, label in revs:
            wlen = words.shape[0] 
            
            if wlen not in self.revs_by_length: 
                self.revs_by_length[wlen] = []  
                
            self.revs_by_length[wlen].append((words, label),) 

        #DataLoader dla kazdego zbioru tweetow o tej samej dlugosci
        self.loaders = {wlen : torch.utils.data.DataLoader(revs, batch_size=batch_size, shuffle=True, drop_last=drop_last) for wlen, revs in self.revs_by_length.items()}
    
    
    def __iter__(self): 
        iters = [iter(loader) for loader in self.loaders.values()] 
        while iters:
            im = random.choice(iters) 
            try:
                yield next(im)      
            except StopIteration:
                iters.remove(im)

############################## TRAIN ###############################

def md_train(model, train_loader, valid_loader, test_loader, num_epochs=5, learning_rate=1e-5, pltout=True):
    print('Trening recurential network..\n')
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    losses, train_acc, valid_acc, epochs = [], [], [], []  
    
    time.sleep(1)
    start = time.time()
    for epoch in range(num_epochs):          #dla kazdej epoki
        for tweets, labels in train_loader:  #przechodze dane treningowe
            if epoch != 0:
                sys.stdout.write(f'\rEpoch number: {epoch}/{num_epochs}')
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
            sys.stdout.write(f'\rEpoch number: {epoch+1}      | Loss value: {loss:.4f} | Train accuracy: {round(train_acc[-1],3)} | Valid accuracy: {round(valid_acc[-1],3)}\n')
    sys.stdout.write(f'\r> Done ({time.time()-start:.2f} s)                                                                                                   \n')
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
    else:
        print("plotout == False\n")    
    print('Dokładność na zbiorze testowym wynosi : {:.4f}'.format(get_accuracy(model, test_loader)))


    












