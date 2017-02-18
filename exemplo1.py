class NaiveBaseClass:
    def calculate_relative_occurences(self, list1):
        no_examples = len(list1)
        ro_dict = dict(Counter(list1))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(no_examples)
        return ro_dict

    def get_max_value_key(self, d1):
        values = d1.values()
        keys = d1.keys()
        max_value_index = values.index(max(values))
        max_key = keys[max_value_index]
        return max_key

    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)

class NaiveBayesText(NaiveBaseClass):
    """"
    When the goal is classifying text, it is better to give the input X in the form of a list of lists containing words.
    X = [
    ['this', 'is', 'a',...],
    (...)
    ]
    Y still is a 1D array / list containing the labels of each entry

    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = []

    def train(self, X, Y):
        self.class_probabilities = self.calculate_relative_occurences(Y)
        self.labels = np.unique(Y)
        self.no_examples = len(Y)
        self.initialize_nb_dict()
        for ii in range(0,len(Y)):
            label = Y[ii]
            self.nb_dict[label] += X[ii]
        #transform the list with all occurences to a dict with relative occurences
        for label in self.labels:
            self.nb_dict[label] = self.calculate_relative_occurences(self.nb_dict[label])
