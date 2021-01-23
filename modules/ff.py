def find_features(document, word_features):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features
