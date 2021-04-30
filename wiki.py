"""
Wikipedia language detector
    This program classifies two languages Italian and Dutch
    based on 20 word sentences
    Models implemented: DecisionTree, AdaBoost
author - Abhishek Inamdar (ai2363@rit.edu)
"""
import math
import pickle  # serialization-deserialization utility
import sys

from data_line import getLines
from data_line import getPredictionLines
from decisionTree import dTree

TRAINING_FILE = "./data/train_data.txt"
VALIDATION_FILE = "data/validation_data.txt"
D_TREE_OBJECT_FILE = "./objects/dTree.obj"
A_BOOST_OBJECT_FILE = "./objects/aBoost.obj"


class DecisionTreeModel:
    """
    Decision tree Model
    """

    __slots__ = "data", "out_file", "tree"

    def __init__(self, train_file, out_file):
        """
        Constructor method
        :param train_file: training data
        :param out_file: object output file
        """
        lines = getLines(train_file)

        self.data = {"train": lines[0]}
        self.out_file = out_file
        self.tree = None

    def train(self, depth):
        """
        training method
        persisting decision tree object into file
        :param depth: depth of decision tree
        :return: None
        """
        examples = self.data["train"]
        features = set(examples[0].features.keys())
        self.tree = dTree(examples, features, [], depth)

        f = open(self.out_file, "wb")
        pickle.dump(self, f)
        f.close()

    def test(self, test_file):
        """
        test method
        :param test_file: test data file
        :return: results
        """
        if not self.tree:
            raise FileNotFoundError("Training is required")
        if not test_file:
            raise ValueError("Test file is required")

        examples = getLines(test_file)
        results = []

        for ex in examples[0]:
            decision = self.tree.decide(ex)
            results.append({"value": ex.value, "goal": ex.goal, "decision": decision})

        return results

    def predict(self, predict_file):
        """
        predict method
        :param predict_file: prediction data file
        :return: results
        """
        if not self.tree:
            raise FileNotFoundError("Training is required")
        if not predict_file:
            raise ValueError("Prediction file is required")

        examples = getPredictionLines(predict_file)
        results = []

        for ex in examples[0]:
            decision = self.tree.decide(ex)
            results.append({"value": ex.value, "decision": decision})

        return results


class WeightedSample:
    """
    weighted sample representation
    """

    __slots__ = "data", "sum", "dist_sum"

    def __init__(self, dataLines):
        """
        constructor method
        :param dataLines: list of instances.
        """
        self.data = dataLines
        self.sum = 0

        for dataLine in self.data:
            dataLine.weight = 1
            self.sum += dataLine.weight

        self.dist_sum = self.sum

    def normalize(self):
        """
        normalize method
        :return: None
        """
        z = self.dist_sum / self.sum
        self.sum = 0

        for dataLine in self.data:
            dataLine.weight *= z
            self.sum += dataLine.weight

    def changeWeight(self, i, new_weight):
        """
        Change the weight of dataLine in the sample
        :param i: index of the dataLine
        :param new_weight: new weight
        :return: None
        """
        self.sum -= self.data[i].weight
        self.data[i].weight = new_weight
        self.sum += new_weight


class AdaBoostModel:
    """
    Ada Boost Model
    """

    __slots__ = "data", "out_file", "stumps", "dTree"

    def __init__(self, train_file, out_file):
        """
        constructor method
        :param train_file: training data
        :param out_file: object output data
        """
        lines = getLines(train_file)

        self.data = {"train": lines[0]}
        self.out_file = out_file
        self.stumps = []
        self.dTree = None

    def train(self, no_of_stumps):
        """
        training method
        persisting AdaBoost object into file
        :param no_of_stumps: no of stumps to be used
        :return: None
        """
        examples = self.data["train"]
        features = set(examples[0].features.keys())

        # AdaBoost Algorithm
        sample = WeightedSample(examples)
        self.stumps = []
        for i in range(no_of_stumps):
            # decision tree with depth 1
            stump = dTree(examples, features, [], 1)
            error = 0

            for example in examples:
                decision = stump.decide(example)

                if decision != example.goal:
                    error += example.weight

            for j in range(len(examples)):
                example = examples[j]
                decision = stump.decide(example)

                if decision == example.goal:
                    new_weight = example.weight * error / (sample.dist_sum - error)
                    sample.changeWeight(j, new_weight)

            sample.normalize()
            stump.weight = math.log(sample.dist_sum - error) / error
            self.stumps.append(stump)

        f = open(self.out_file, "wb")
        pickle.dump(self, f)
        f.close()

    def test(self, test_file):
        """
        Test method
        :param test_file: test data
        :return: None
        """
        if not self.stumps:
            raise FileNotFoundError("Training is required")

        if not test_file:
            raise ValueError("Test file is required")
        examples = getLines(test_file)
        results = []

        for ex in examples[0]:
            decision = self.vote(ex)
            results.append({"value": ex.value, "goal": ex.goal, "decision": decision})

        return results

    def predict(self, predict_file):
        """
        predict method
        :param predict_file: prediction data file
        :return: results
        """
        if not self.stumps:
            raise FileNotFoundError("Training is required")

        if not predict_file:
            raise ValueError("Prediction file is required")
        examples = getPredictionLines(predict_file)
        results = []

        for ex in examples[0]:
            decision = self.vote(ex)
            results.append({"value": ex.value, "decision": decision})

        return results

    def vote(self, dataLine):
        """
        classifies dataLine based on vote
        :param dataLine: line of data to classify
        :return: classification
        """
        count = {}
        max_count = 0
        result = None

        for stump in self.stumps:
            decision = stump.decide(dataLine)

            if decision in count:
                count[decision] += stump.weight
            else:
                count[decision] = stump.weight

            if count[decision] > max_count:
                max_count = count[decision]
                result = decision

        return result


def showUsageMessage(showBothMessages):
    """
    helper method for showing Usage messages
    :param showBothMessages: to show messages or not
    :return: None
    """
    if showBothMessages:
        print("Usage: python3 wiki.py train <training-data-file>")
        print("Usage: python3 wiki.py predict <DT|AB|BS> <predict-data-file>")
    else:
        print("Incorrect Arguments.")
        print("Usage: python3 wiki.py predict <DT|AB|BS> <predict-data-file>")
    exit(1)


def printResults(results, modelName):
    """
    Print results based on model's results and given examples
    :param results: list of results
    :param modelName: Name of the Model
    """
    correct = 0
    total = 0
    for res in results:
        total += 1
        if res["decision"] == res["goal"]:
            correct += 1
    accuracyPercentage = round((correct / total) * 100, 2)
    print(modelName + " Model Accuracy: " + str(accuracyPercentage) + "%")


def printPredictionResults(predictions):
    """
    Print results based on model's prediction and given examples
    :param predictions: list of predictions
    """
    total = 0
    print()
    for res in predictions:
        total += 1
        print("Segment:", res["value"][:30], "| Prediction:", res["decision"])
    print()


def train(examples):
    """
    train method
    :param examples: examples to train model,
        each example will be in the following format:
        <IT or DU><|><20 word sentence>
        "IT" denotes Italian, "DU" denotes Dutch,
        <|> is delimiter between identifier and sentence
    :return: None
    """
    dt_model = DecisionTreeModel(examples, D_TREE_OBJECT_FILE)
    # depth of the decision tree
    dt_model.train(5)

    ab_model = AdaBoostModel(examples, A_BOOST_OBJECT_FILE)
    # number of stumps
    ab_model.train(10)


def validate(validate_file):
    """
    validates trained models
    :param validate_file: validation set file
        each line will be in the following format:
        <IT or DU><|><20 word sentence>
        "IT" denotes Italian, "DU" denotes Dutch,
        <|> is delimiter between identifier and sentence
    :return: None
    """
    objFile = D_TREE_OBJECT_FILE
    objFile = open(objFile, "rb")
    model = pickle.load(objFile)
    objFile.close()
    results = model.test(validate_file)
    printResults(results, "Decision Tree")

    objFile = A_BOOST_OBJECT_FILE
    objFile = open(objFile, "rb")
    model = pickle.load(objFile)
    objFile.close()
    results = model.test(validate_file)
    printResults(results, "AdaBoost")


def predict(model, predict_file):
    """
    predicts given examples based on the given trained model
    each example should be in the following format:
    <20 word sentence>
    :param model: which model to use?
    :param predict_file: prediction file containing test examples
    :return: None
    """
    if model == "DT":
        objFile = D_TREE_OBJECT_FILE
    else:
        objFile = A_BOOST_OBJECT_FILE
    objFile = open(objFile, "rb")
    model = pickle.load(objFile)
    objFile.close()
    predictions = model.predict(predict_file)
    printPredictionResults(predictions)


def main():
    """
    Main method
    :return: None
    """
    if len(sys.argv) < 3:
        showUsageMessage(True)

    action = sys.argv[1]

    if action == "train":
        examples = sys.argv[2]
        train(examples)
        print("Training Done!!")
        validate(VALIDATION_FILE)
        print("Validation Done!!")

    elif action == "predict":
        if len(sys.argv) < 4:
            showUsageMessage(False)
        model = sys.argv[2]
        if model not in {"DT", "AB", "BS"}:
            showUsageMessage(False)
        predict_file = sys.argv[3]
        print("Prediction Starts...")
        predict(model, predict_file)
        print("Prediction End!")


if __name__ == '__main__':
    main()
