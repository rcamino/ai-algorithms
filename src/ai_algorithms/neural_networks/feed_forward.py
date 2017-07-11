class FeedForwardNetwork(object):

    def __init__(self, layers):
        self.layers = layers

    def score(self, features):
        last_layer_output = features
        for layer in self.layers:
            last_layer_output = layer.activate(last_layer_output)
        return last_layer_output

    def predict(self, features):
        scores = self.score(features)
        predictions = range(len(scores))
        return sorted(predictions, reverse=True, key=lambda prediction: scores[prediction])[0]
