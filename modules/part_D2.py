#D. Analiza o neuronowe

import sys
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from modules.ff import get_accuracy

######################## 2. Embeddingi 200D ##################################

def get_vectors(glove_vector, revs, batch_size=200, shuffle=True):
    sys.stdout.write('\rGetting vectors loaders..')
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
        
    #za kazdym razem będzie bral 200 rekordow (przy trenowaniu) i co epoke tasujemy (shuffle = True)
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, shuffle) 
    test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle)
    
    sys.stdout.write('\rGetting vectors loaders => Done\n ')
    return train_loader, valid_loader, test_loader

############################## TRAIN ###############################

def train_network(model, train_loader, valid_loader, test_loader, num_epochs=5, learning_rate=1e-5, pltout=True):
    print('Neural Glove analyse\n')
    criterion = nn.CrossEntropyLoss()  #funkcja kosztu
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    losses, train_acc, valid_acc, epochs = [], [], [], []
    
    print(f'Model parameters:\n> Training_set size: {len(train_loader.dataset)} | Valid_set size: {len(valid_loader.dataset)} | Testing_set size: {len(test_loader.dataset)}\
\n> Learning Rate: {learning_rate} | Iterations: {num_epochs}\n')
    time.sleep(1)
    start = time.time()
    for epoch in range(num_epochs):          #dla kazdej epoki
        for tweets, labels in train_loader:  #przechodze dane treningowe
            sys.stdout.write(f'\rEpoch number: {epoch+1}/{num_epochs}')
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
        print("plotout == False")
    
    acc = get_accuracy(model, test_loader)
    print(f"Final test accuracy: {acc:.2f}\n")