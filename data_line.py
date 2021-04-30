class DataLine:
    """
    representation of a single line of data.
    """

    __slots__ = "goal", "value", "features", "weight"

    def __init__(self, line, isPredict):
        """
        constructor method
        :param line: The input line
        """
        if isPredict:
            self.goal = None
            self.value = line
        else:
            self.goal = line[:2]
            self.value = line[3:]
        self.features = get_features(line)
        self.weight = None


def getLines(file):
    """
    returns objects of DataLines from a file
    :param file: file of the data
    :return: List of DataLine objects
    """
    lines = [[]]
    for line in open(file):
        if not len(line.strip()) > 3:
            continue
        lines[0].append(DataLine(line, False))
    return lines


def getPredictionLines(file):
    """
    returns objects of DataLines from a prediction file
    :param file: file of the data
    :return: List of DataLine objects
    """
    lines = [[]]
    for line in open(file):
        if not len(line.strip()) > 0:
            continue
        lines[0].append(DataLine(line, True))
    return lines


def get_features(line):
    """
    Gets the features of a line as a Map/Dictionary.
    All features are boolean, can be expanded further
    :param line: The line to be operated on
    :return: map of features
    """
    words = set(line.split())

    return {
        # Italian features
        "containsWord_di": "di" in words,
        "containsWord_e": "e" in words,
        "containsWord_il": "il" in words,

        # Dutch features
        "containsWord_het": "het" in words,
        "containsWord_de": "de" in words,
        "containsWord_dat": "dat" in words,
    }


def endsWith(suffix, line):
    """
    Checks if there exists a word which ends with a given suffix
    :param suffix: the suffix
    :param line: the line
    :return: True if word exists, False otherwise
    """
    line = line.split()
    for word in line:
        if len(word) < len(suffix):
            continue
        i = len(word) - len(suffix)
        val = True
        for ch in suffix:
            if word[i] != ch:
                val = False
                break
            i += 1
        if val:
            return val
    return False
