import math


def dTree(examples, features, parent_examples, depth):
    """
    decision tree builder method (recursive)
    :param examples: current examples
    :param features: set of features
    :param parent_examples: parent's examples
    :param depth: maximum depth
    :return: decision tree Root
    """
    if not examples:
        return DTNode(majorityOutput(parent_examples), True)
    if not features:
        return DTNode(majorityOutput(examples), True)
    if isGoalSame(examples):
        return DTNode(examples[0].goal, True)

    feature, children = getMaximumGain(examples, features)
    root = DTNode(feature)

    if depth < 1:
        depth = 1

    for value in children:
        exs = children[value]
        if depth == 1:
            subtree = DTNode(majorityOutput(exs), True)
            root.add(value, subtree)
        else:
            subtree = dTree(exs, features.difference({feature}), examples, depth - 1)
            root.add(value, subtree)

    return root


class DTNode:
    """
    Node class for decision tree
    """

    __slots__ = "is_leaf", "value", "children", "weight"

    def __init__(self, value, is_leaf=False):
        """
        constructor method
        :param value: the node's value
        :param is_leaf: is this a leaf node?
        """
        self.is_leaf = is_leaf
        self.value = value
        self.children = {}
        self.weight = None

    def add(self, label, d_node):
        """
        Adds a child to the node.
        :param label: the branch label
        :param d_node: the new node
        """
        self.children[label] = d_node

    def decide(self, dataLine):
        """
        Classify given of data.
        :param dataLine: line of Data
        :return: result
        """
        node = self

        while node:
            if node.is_leaf:
                return node.value
            branch = dataLine.features[node.value]

            if branch in node.children:
                node = node.children[branch]
            else:
                return vote(node)

        return None


def vote(node):
    """
    Get the majority classification from node's children
    :param node: the node to be voted on
    :return: majority classification
    """
    if node.is_leaf:
        return node.value

    if not node.children:
        return None

    count = {}
    max_count = -1
    max_val = None

    for branch in node.children:
        value = vote(node.children[branch])

        if not value:
            continue

        count[value] = count[value] + 1 if value in count else 1

        if count[value] > max_count:
            max_count = count[value]
            max_val = value

    return max_val


def count_goals(examples):
    """
    Counts the number of examples for
    each classification.
    Considering weights
    :param examples: list of examples
    :return: count of every classification
    """
    count = {}

    for example in examples:
        weight = example.weight if example.weight else 1

        if example.goal in count:
            count[example.goal] += weight
        else:
            count[example.goal] = weight

    return count


def majorityOutput(examples):
    """
    Gets the majority classification from a
    list of examples.
    :param examples: list of examples.
    :return: majority classification
    """
    value = None
    max_weight = -1
    count = count_goals(examples)

    for ex in examples:
        if count[ex.goal] > max_weight:
            max_weight = count[ex.goal]
            value = ex.goal

    return value


def isGoalSame(examples):
    """
    Checks if every example in the list
    has the same classification.
    :param examples: list of examples
    :return: True or False
    """
    goal = examples[0].goal
    for i in range(1, len(examples)):
        if examples[i] != goal:
            return False
    return True


def getEntropy(examples):
    """
    :param examples: list of examples
    :return: entropy of the list
    """
    count = count_goals(examples)
    total = 0

    for key in count.keys():
        p = count[key] / len(examples)
        total += -p * math.log(p, 2)

    return total


def getMaximumGain(examples, features):
    """
    calculates Maximum gain
    :param examples: list of examples
    :param features: set of features
    :return: feature and split with max gain.
    """
    e = getEntropy(examples)
    max_gain = -1
    max_feature = None
    children = None

    for feature in features:
        gain, c = calculateGain(examples, feature, e)

        if gain > max_gain:
            max_gain = gain
            max_feature = feature
            children = c

    return max_feature, children


def calculateGain(examples, feature, e):
    """
    Calculates the information gain after
    splitting examples on a feature.
    :param examples: list of examples
    :param feature: feature to split on
    :param e: current entropy
    :return: information gain
    """
    children = split(examples, feature)
    total = 0

    for c in children:
        labelled_examples = children[c]
        total += (len(labelled_examples) / len(examples)) * getEntropy(labelled_examples)

    gain = e - total
    return gain, children


def split(examples, feature):
    """
    Splits a list of examples on a feature.
    :param examples: list of examples
    :param feature: feature to split on.
    :return: table representing the split.
    """
    result = {}

    for example in examples:
        value = example.features[feature]

        if value in result:
            result[value].append(example)
        else:
            result[value] = [example]

    return result
