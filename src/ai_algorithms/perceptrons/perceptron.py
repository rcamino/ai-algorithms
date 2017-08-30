class Perceptron(object):

    def __init__(self, weights):
        self.weights = weights

    def score(self, features):
        prediction = 0.0
        for feature_i, weight_i in zip(features, self.weights):
            prediction += feature_i * weight_i
        return prediction

    def predict(self, features):
        return 1 if self.score(features) > 0 else 0


def perceptron_learning(samples, labels, positive_label=1, iterations=1000):
    n_features = len(samples[0])
    perceptron = create_empty_perceptron(n_features)
    for iteration in xrange(iterations):
        change = False
        for features, label in zip(samples, labels):
            prediction = perceptron.predict(features)
            if prediction != label:
                change = True
                if label == positive_label:
                    update_perceptron(perceptron, features, 1)
                else:
                    update_perceptron(perceptron, features, -1)
        if not change:
            break
    return perceptron


def create_empty_perceptron(size):
    return Perceptron([0.0 for _ in xrange(size)])


def update_perceptron(perceptron, features, direction):
    for i, feature_i in enumerate(features):
        perceptron.weights[i] += feature_i * direction
