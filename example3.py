"""
This is an example code which uses the class Neurolnetwork to create and train
a network. It is richly commented, s. th. you can let yourself be guided through
the programm.
"""

# Print the intro message
print(' ')
print('********************************')
print('Starting the example program:')
print('********************************')
print(' ')

# import the network as a class
from network import NeuralNetwork as NN

# import the network handler
import network_handler as NH

# import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# As every neural network, it needs data to be trained, which we
# have to provide some. To this end define a classification rule
# as a kind of game and see whether the network finds the pattern.
# The input is _ _ _ _ _ with X or O each. The rule is
# At the configuration OxOxO or xOxOx we get class 1 (001)
# At the configuration XXXXX or OOOOO we get class 2 (010)
# At each other configuration we get class 3 (100)
def condition(inp):
    a = inp[0]
    b = inp[1]
    c = inp[2]
    d = inp[3]
    e = inp[4]
    if (a == c) & (c == e) & (b == d) & (a != b):
        return 0
    elif (a == b) & (c == d) & (a == e) & (b == c):
        return 1
    else:
        return 2


# Now define the dimensions:
# Each X or O position is a single input neuron
# Therefore, we need 5 input neurons
nr_input_neurons = 5

# Let us assume we have  2000 data points
nr_input_data = 2000

# We have three different classes, thus 3 output neurons
nr_classes = 3

# Input features should be random configurations
raw_input = np.random.randint(2, size=(nr_input_data, nr_input_neurons))
output = []

# Input target, here called output, is the rule applied to the input
for inp in raw_input:
    output.append( condition( inp ) )

# Define the architecture, here we have 3 hidden layers
architecture = [nr_input_neurons, 6, 6, 4, nr_classes]

# Create the network
network = NH.create_network(architecture)

# Give the data to the network
data = NH.construct_data(raw_input, output)


# Train Test split
train_data, cv_data, test_data = NH.part_data(data, 
                                    parts=[0.6, 0.2, 0.2],
                                    shuffle=True)

# Some info for us
print('shape of the training data: ', train_data.shape)

# Let us try a lambda parameter of 0.2
lambda_parameter = 0.2

# With this method we train the network
result = NH.training_with_test(network, train_data, cv_data,
                    nr_batches=5, display='graphic and text',
                    lam = lambda_parameter, nr_iterations=10,
                    shuffle=True)

# The returning parameter is a dictionary. Lets get the
# interesting values
thetalist = result['Thetalist']
Jtestlist = result['Jtestlist']
Jtrainlist = result['Jtrainlist']


# What exactly you do with these informations is up to
# you, but for this example let us use the iteration
# where the cost of the cv data set is minimal
opt_cv_pos = np.argmin(Jtestlist)

# And set the network theta to this
network.Theta = thetalist[opt_cv_pos]

# Make predictions with the test inputs.
# For this we have to extract the input
# and output from the test data
test_input, test_output = NH.separate_data(test_data)
test_pred = NH.make_prediction(network, test_input)

# For educational purpose we want to take a look at the first
# seven input entries and compare their outcome
print(' ') # new line for better visibility
for i in range( 6 ):
    print(f'Input: {test_input[i]} with Output: {test_output[i]} and Prediction: {test_pred[i]}')
    

# Now compare the prediction with the data to see how sucessfull the network is
hits = 0
for i in range( len(test_output) ):
    if test_output[i] == test_pred[i]:
        hits += 1

print(' ')
print(f'From {len(test_output)} predictions were {hits} right.')

# Saving the network
NH.save_network(network, 'Testnetwork')

#####################################################################
# So far so good.
# Now you know everything about my Neural Network class
# and the corresponding handler. Finally, it is up to you
# to play around with it and try some things out. For instance,
# the accuracy of the predictions highly depend on the number of 
# batches, the weight parameter lambda and the number of iterations
#####################################################################