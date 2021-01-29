#Funkcje pomocnicze do projektu

def find_features(document, word_features):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features

#Funkcja wyznaczająca dokładność predykcji:
def get_accuracy(model, data_loader):
    correct, total = 0, 0  #ile ok, ile wszystkich
    for revs, labels in data_loader: #przechodzi dane
        output = model(revs)         #jak dziala model
        pred = output.max(1, keepdim=True)[1]  #ktora kategoria
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total
