from ai_algorithms.perceptrons.perceptron import create_empty_perceptron, update_perceptron


class MultiClassPerceptron(object):

    def __init__(self, perceptron_per_class):
        self.perceptron_per_class = perceptron_per_class

    def score(self, features):
        return [perceptron.score(features) for perceptron in self.perceptron_per_class.values()]

    def predict(self, features):
        predictions = self.perceptron_per_class.keys()
        scores = map(lambda prediction: self.perceptron_per_class[prediction].score(features), predictions)
        return sorted(predictions, reverse=True, key=lambda prediction: scores[prediction])[0]


def multiclass_perceptron_learning(samples, labels, possible_labels=None, iterations=1000):
    n_features = len(samples[0])
    if possible_labels is None:
        possible_labels = set(labels)
    perceptron_per_class = {}
    for label in possible_labels:
        perceptron_per_class[label] = create_empty_perceptron(n_features)
    multiclass_perceptron = MultiClassPerceptron(perceptron_per_class)
    for iteration in xrange(iterations):
        change = False
        for features, label in zip(samples, labels):
            prediction = multiclass_perceptron.predict(features)
            if prediction != label:
                change = True
                update_perceptron(perceptron_per_class[label], features, 1)
                update_perceptron(perceptron_per_class[prediction], features, -1)
        if not change:
            break
    return multiclass_perceptron
