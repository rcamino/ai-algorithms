class MultiClassPerceptron(object):

    def __init__(self, perceptron_per_class):
        self.perceptron_per_class = perceptron_per_class

    def score(self, features):
        return [perceptron.emit(features) for perceptron in self.perceptron_per_class.values()]

    def predict(self, features):
        pairs = [(prediction, perceptron.emit(features))
                 for prediction, perceptron in self.perceptron_per_class.items()]
        sorted_pairs = sorted(pairs, reverse=True, key=lambda pair: pair[1])
        best_pair = sorted_pairs[0]
        return best_pair[0]


class Perceptron(object):

    def __init__(self, weights):
        self.weights = weights

    def score(self, features):
        prediction = 0.0
        for feature_i, weight_i in zip(features, self.weights):
            prediction += feature_i * weight_i
        return prediction

    def predict(self, features):
        return 1 if self.prediction(features) > 0 else 0


def update_perceptron(perceptron, features, direction):
    for i, feature_i in enumerate(features):
        perceptron.weights[i] += feature_i * direction


def perceptron_learning(samples, labels, positive_label=1, iterations=1000):
    multiclass_perceptron = multiclass_perceptron_learning(samples, labels, iterations)
    return multiclass_perceptron.perceptron_per_class[positive_label]


def multiclass_perceptron_learning(samples, labels, iterations=1000):
    n_features = len(samples[0])
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


def create_empty_perceptron(size):
    return Perceptron([0.0 for _ in xrange(size)])


def add_bias_to_features(features):
    return [1.0] + features


def add_bias_to_samples(samples):
    return map(add_bias_to_features, samples)
