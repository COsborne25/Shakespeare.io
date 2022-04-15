import random;
import math;
import numpy as np;
import re;

import constants;

# random.seed(1);

class NeuralNetwork:

    def __init__(self, layer_sizes, learning_weight):

        self.weights = []
        for i in range(0, len(layer_sizes) - 1):
            layer_weights = [];
            for j in range(0, layer_sizes[i + 1]):
                layer_weights.append([ random.uniform(-1, 1) for k in range (0, layer_sizes[i]) ]);
            self.weights.append(np.array(layer_weights));
        self.size = len(self.weights);

        self.biases = [];
        for i in range(0, len(layer_sizes) - 1):
            self.biases.append(np.array([ [random.random()] for j in range(0, layer_sizes[i + 1]) ]));

        self.learning_weight = learning_weight;
    
    # Returns: z_layers
    def ForwardPropogate(self, input):
        """ Forward propogate the neural network for a single input. """

        # output = n(weights * inputs + biases)
        z_layers = [input];
        for layer in range(0, len(self.weights)):
            z = np.matmul(self.weights[layer], input) + self.biases[layer];
            input = self.Normalization(z);
            z_layers.append(z);
        return z_layers;    

    def Cost(self, input):
        """ Calculate the cost of the function for a single input. """

        outputs = self.ForwardPropogate(input);
        guess = self.Normalization(outputs[self.size]);
        solution = Solution(input);

        return np.dot(np.transpose(guess - solution)[0], np.transpose(guess - solution)[0]);

    def AverageCost(self, input_data):
        """ Calculate the average cost for a set of input data. """
        average_cost = 0;
        for set in input_data:
            average_cost = self.Cost(set);
        return average_cost / len(input_data);

    def BackwardPropgate(self, input):
        """ Adjust the weights and biases for a single input. """

        # cost = (output - solution)^2
        outputs = self.ForwardPropogate(input);
        guess = self.Normalization(outputs[self.size]);
        solution = Solution(input);

        # C = (y^n - solution)^2
        # dC/dy^n = 2 * (y^n - solution)
        # delta^n = dC/dy^n * n'(z^n)
        delta = 2 * (outputs[len(self.weights)] - solution) * self.dNormalization(outputs[len(self.weights)]);

        dWeights = [];
        dBiases = [];

        for i in range(0, len(self.weights)):
            dWeights.append(delta * np.transpose(self.Normalization(outputs[self.size - i - 1])));
            dBiases.append(delta);
            delta = np.matmul(np.transpose(self.weights[self.size - i - 1]), delta) * self.dNormalization(outputs[self.size - i - 1]);

        dWeights.reverse();
        dBiases.reverse();

        for layer in range(0, self.size):
            self.weights[layer] -= self.learning_weight * dWeights[layer];
            self.biases[layer] -= self.learning_weight * dBiases[layer];
    
    def Train(self, training_set):
        """ Run back propogation on a training set of data. """
        for input in training_set:
            self.BackwardPropgate(input);

    def Normalization(self, input):
        return 1.0 / (1 + np.exp(-input));
    
    def dNormalization(self, input):
        return np.exp(-input) / np.square(1 + np.exp(-input));


"""
Read the quotes from the input files.

"""

def ToText(arr):
    """ Converts the numpy array back into raw text. """
    ret = "";
    for element in arr:
        c = '';
        if(element[0] == 0.0):
            c = ' ';
        else:
            c = chr(int(element * 26 + 64));
        ret += c;
    return ret;

def ToArr(text):
    """ Converts the raw text back into an array. """
    arr = [];
    for c in text:
        if(c == " "):
            arr.append([0.0]);
        else:
            arr.append([(ord(c) - 64) / 26.0]);
    return np.array(arr);

def FormatQuote(quote):
    """ Formats and numerates the characters in the quote. """
    quote = re.sub(r'[^\w\s]', '', quote);
    quote = quote.upper();
    
    while(len(quote) < constants.ENTRY_LENGTH):
        quote += " ";

    return ToArr(quote[:constants.ENTRY_LENGTH]);

tempest_quotes = {};
""" Dictionary with keys as formatted tempest quotes and values as their speaker. """

lines_file = open("lines.txt", "r");
characters_file = open("characters.txt", "r");

lines = lines_file.readlines();
characters = characters_file.readlines();

for i in range(0, len(lines)):
    formatted_quote = FormatQuote(lines[i][:-1]);
    tempest_quotes[ToText(formatted_quote)] = characters[i][:-1];

lines_file.close();
characters_file.close();

"""
Create the word dictionary.

"""

def QuoteToWordsArray(line):
    words = line.split(" ");
    while('' in words):
        words.remove('');
    return words;

words_dictionary = {};
for line in tempest_quotes.keys():
    words = QuoteToWordsArray(line);
    for word in words:
        if word not in words_dictionary.values():
            words_dictionary[len(words_dictionary)] = word;
words_lookup = {v: k for k, v in words_dictionary.items()}

def WordToArray(word):
    arr = np.array([ [0] for i in range(0, len(words_dictionary)) ]);
    arr[words_lookup.get(word)][0] = 1;
    return arr;

"""
Organize the quotes into training sets.

"""

word_associations = {};
current_word = "";
for line in tempest_quotes.keys():
    words = QuoteToWordsArray(line);
    for word in words:
        if(current_word == ""):
            current_word = word;
            continue;
        word_associations[current_word] = word;
        current_word = word;
print(word_associations);


training_data = [];
testing_data = [];

for i in range(0, len(tempest_quotes.keys())):
    if(i % 2 == 0):
        training_data.append(ToArr(list(tempest_quotes.keys())[i]));
    else:
        testing_data.append(ToArr(list(tempest_quotes.keys())[i]));

speakers_set = set(tempest_quotes.values());
speakers_dict = {};
i = 0;
for speaker in speakers_set:
    speakers_dict[speaker] = i;
    i += 1;
speakers_num = {v: k for k, v in speakers_dict.items()}

def Solution(input):
    """ Returns the solution to a given input. """
    sol = [[0] for i in range(0, len(speakers_dict))];
    sol[speakers_dict.get(tempest_quotes.get(ToText(input)))] = [1];
    return np.array(sol);

"""
Train the neural network to read the quotes.

"""

net = NeuralNetwork([2, 1000, 1000, len(speakers_dict)], constants.TRAINING_RATE);

round = 0;
while(True):
    print("Round #" + str(round) + " | Training Cost " + str(net.AverageCost(training_data)) + " | Testing Cost " + str(net.AverageCost(testing_data)));
    net.Train(training_data);
    round += 1;
