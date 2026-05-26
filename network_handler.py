from platform import architecture
import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork as NN
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

"""
    ***********************************************************************************

    Because the purpose of the network class is simply for having a virtuel network and
    not for validation etc, we construct a handler module.
    
    ***********************************************************************************
"""

"""
    Create a whole neural network at once.

    Takes:      * a list where each element stands for a layer (also in- and output).
                The element is an integer and tells how many neurons the layer should have.
                Note that the last element defines the number of classes.

    Returns:    * an initialized network of the given architecture
"""
def create_network(architecture):
    try:
        nr_start = architecture[0]
        nr_end = architecture[-1]
        network = NN(nr_start, nr_end)

        for i in range( 1, len(architecture) - 1 ):
            network.add_layer( architecture[i] )

        network.build_network()
        return network
    
    except:
        print("Error in creating the network.")
        return -1

"""
    Since we want to classify, the required output is an index,
    or a list of indeces (with the lenght of the batch size), where
    each index stands for the corresponding class. However, for the
    algorithm the output has to be a vector (or list of vectors = matrix)
    whose entries are one for the class they are and zero else.
    Therefore, the index-list has to be translated.

    Takes:      * a list of integers, which indicates the class.
                Note that this list goes from 0 to number outputs minus 1
                E.g. 4 -> [0, 0, 0, 1, 0, 0]
        
    Returns:    * an I x J matrix, where I is the batch size and J is the number
                of output layers. The entries are one or zero
"""
def translate(network, outputs):
    m = len(outputs)    # this is the batch size
    # y should be batch_size x nr_ends
    y = np.zeros((m, network.NR_END))

    for i, iy in enumerate(outputs):
        # check if outputs is a correct index list
        if int(iy) == iy and iy > -1 and iy < network.NR_END:
            # the iy-th entry of the i-th list item has to be
            # one since this is the corresponding index
            # everything else is already zero
            y[i, iy] = 1
        else:
            # This is a user error
            print('No index list were given. Check output!')
            return -1
            
    return y

"""
    This does the reverse of the translate-method above. The predictions we
    get from the network is a two dimenstional array, where each row is a
    vector of probalities for the class-outcom of the corresponding input row.

    While those probabilities might be useful we often want a clear decision from
    the neural network. For the sake of this learning project, we consider the
    class with the highest probability as the predicted class.

    Takes:      * An I x J matrix, where I is the batch size and J is the number
                  of classes

    Returns:    * A list of integers which indicate what class it is most likely
"""
def get_index_list(y):
    try:
        ind_list = []
        for iy in y:
            ind_list.append( np.argmax(iy) )
        return ind_list
    except:
        print("Something goes wrong in the translation to index list.")
        return -1


"""
    Since I have no external restrictions in terms of how the data looks like, 
    I define my own structure. This method brings the data in exactly this structure.
    The data is an I x J array, where I is the number of data sets and 
    J is the number of input layers + 1. Each row is the input vector with the output
    value (class index) at the end.
    E.g. The input [1,2,3,4] corresponds to the class number 0 and the input [1,1,1,1]
    to the class number 2, the data = [ [1,2,3,4,0] , [1,1,1,1,2]]
"""
def construct_data(inputs, outputs):
    try:
        outputs = np.array([outputs])
        data = np.concatenate((inputs.T,outputs)).T
        return np.array( data )
    except:
        print("Fail to construct data structure. Check dimensions")
        return -1
    
"""
    Of course we need the data in the separated form for calculations.
    That will be done in this extra method because if we change the data
    structure, we can simply adapt this method.

    Takes:      * The data structrue

    Returns:    * (inputs, outputs)
                * Inputs as an I x J Array as usual
                * Outputs as a list of integers 
"""
def separate_data(data):
    inp = data[:,:-1]
    out = np.array( data[:,-1], dtype=int)
    return inp, out

"""
    It is reasonable to randomize the given data. This method does exactly that.

    Takes:      * a 2D array

    Return:     * the same array, where the rows were shuffled randomly
"""
def shuffle_data(data):
    try:
        np.random.shuffle( data )
        return 0
    except:
        print('Cannot shuffle data')
        return -1


"""
    Partitionate the data into Training, Cross validation and
    Test data.

    Takes:      * The data in the given form
                * A list of three real numbers, where each
                  value stands for the part (< 1) of the data
                  [train, cv, test]
                * Bool if we should shuffle the data first

    Return:     * A tuple with data (Train, CV, Test)
"""
def part_data(data, parts=[0.6, 0.2, 0.2], shuffle=False):
    # If wished shufle data
    if shuffle:
        terminating_code = shuffle_data(data)

    # define the separation of the data
    nr_datasets = data.shape[0]

    limit1 = int( np.round( nr_datasets * parts[0] ) )
    limit2 = limit1 + int( np.round( nr_datasets * parts[1] ) )
    
    # If the sum is one or even larger, just set
    # Limit three to the end
    if np.sum(parts) >= 1:
        limit3 = -1
    
    # If not, it is probably intented not to use the whole data
    else:
        limit3 = limit2 + int( np.round( nr_datasets * parts[2] ) )

    # separate data into training, cross validation and test data
    train_data = data[: limit1]
    cv_data = data[limit1 : limit2]
    test_data = data[limit2 : limit3]
    
    return train_data, cv_data, test_data


"""
    Separates the input data into batches and trains the
    given network for each batch consecutively. Note that
    the network will be updated after each batch training.

    Takes:      * The network as an Neural Network object
                * The training data in given data format
                * The number of batches which should be
                  constructed as an integer. The data will
                  be distributed evenly on the batches
                  (except the last maybe)
                * Bool if the data should be shuffled before

    Returns:    * Termination code if error (-1) or not (0)
"""
def train(network, train_data, nr_batches=10, shuffle=True):
    # If wished shuffle the data
    if shuffle:
        termination_code = shuffle_data(train_data)

    # Separate into batches. We request a fix number
    # of batches and have a large number of data. It could
    # be that the last batch may contain another number of data
    datalist = []
    nr_of_data = train_data.shape[0]
    data_per_set = int( nr_of_data / nr_batches )

    # Fill the first N-1 batches with data per set
    # If it is only one batch, this loop will be skipped
    last_point = 0
    for i in range(nr_batches - 1):
        datalist.append( train_data[i*data_per_set : (i+1) * data_per_set] )
        last_point = (i+1) * data_per_set

    # Fill the last batch with the remaining data
    datalist.append( train_data[last_point : -1 , :])

    # Now train for each batch
    for batch in datalist:
        # Take in- and output from the data
        input, output = separate_data(batch)

        # Translate the output into output vectors
        y = translate(network, output)

        # Calculate the optimal theta, without showing the progress
        # Or for debug reasons, show it
        res_theta = network.minimizeJ(input, y, display=False)

        # Updating the network theta
        network.Theta = res_theta

        # debug
        network.debug += 1

    return 0



"""
    This function do most steps which are usually done when training a network.
    Training the network with the train data and test/validate it with the given test data.
    The network will be updated with each training. 

    Takes:      * The network as an Neural Network object
                * The training data in given data format
                * The testing data in given data format
                * Integer of how many batches the training will use
                * String what should be displayed. If it contains 'text'
                  the progress will be written on the screen. If it
                  contains 'graphic' the costs will be plotted
                * Real lam which defines the used lambda weight
                  for the training
                * Integer number of iterations

    Returns:    * Dictionary with:"Thetalist" : thetalist, "Jtestlist" : test_Jlist, "Jtrainlist" : train_Jlist
                - Thetalist : List where each entry is the resulting theta of the network after
                              the corresponding iteration
                - Jtestlist : List where each entry is the cost of the test data for the
                              corresponding iteration
                - Jtrainlist: List where each entry is the cost of the ttrain data for the
                              corresponding iteration
"""
def training_with_test(network, train_data, test_data, nr_batches=10, shuffle=True,
                    display='graphic and text', lam=0, nr_iterations=10):
    train_Jlist = []
    test_Jlist = []
    train_input, train_output = separate_data(train_data)
    test_input, test_output = separate_data(test_data)

    thetalist = [network.Theta]

    train_y = translate(network, train_output)
    test_y = translate(network, test_output)

    # set lambda
    network.LAMBDA = lam
    
    # Think of it as the zero-th iteration 
    train_J = network.cost_without_lambda(train_input, train_y)
    train_Jlist.append(train_J)
    test_J = network.cost_without_lambda(test_input, test_y)
    test_Jlist.append(test_J)

    if 'text' in display:
        print ("{:<9} {:<15} {:<15}".format('# It', 'Cost Train', 'Cost Test'))
        print ("{:<9} {:<15} {:<15}".format('---------', '----------', '----------'))
        print ("{:<9} {:<15} {:<15}".format(0, np.round(train_J,5), np.round(test_J,5)))

    
    # We do this until we have the number of iterations the user wishes.
    # Here is the point where additional convergency conditions may be added
    for nr_it in range(1, nr_iterations+1):
        # Train the network. The termination code tells us if the training
        # have had any errors - only for debugging
        term_code = train(network, train_data, nr_batches=nr_batches, shuffle=shuffle)

        # Separate the in- and output from the data and bring them in the
        # required form
        train_in, train_out = separate_data(train_data)
        test_in, test_out = separate_data(test_data)
        train_y = translate(network, train_out)
        test_y = translate(network, test_out)

        # Save the cost of the training and testing data
        train_J = network.cost_without_lambda(train_in, train_y)
        train_Jlist.append(train_J)
        test_J = network.cost_without_lambda(test_in, test_y)
        test_Jlist.append(test_J)

        # Save the theta in a list since we may want to set our network to
        # it later
        thetalist.append(network.Theta)

        if 'text' in display:
            print ("{:<9} {:<15} {:<15}".format(nr_it, np.round(train_J,5), np.round(test_J,5)))
        

    # If wished, the costs over the iterations will be plotted
    if 'graphic' in display:
        plt.plot(train_Jlist, '-o', label='Train')
        plt.plot(test_Jlist, '-o', label='Test')
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Cost', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid()
        plt.show()
    
    # Return a dictionary with all the informations
    return {"Thetalist" : thetalist, "Jtestlist" : test_Jlist, "Jtrainlist" : train_Jlist}

"""
    When a network is trained, we want to use it to make predictions for inputs.

    Takes:      * The network as a Neural Network object
                * Input as a matrix
                
    Returns:    * An index list where each entry indicates the class
                  of the corresponding input vector
"""
def make_prediction(network, inputs):
    pred = network.calc_h(inputs)
    class_list = get_index_list(pred)
    return class_list








"""
    Saves the properties of the given network in a file, so that it can
    be loaded with this handler.

    Takes:      * The network as an object of the Neural Network class

    Returns:    * Termination code
"""
def save_network(network, filename):
    try:
        # Overwrites any existing file
        # TODO: Check if filename is taken
        with open(filename + '.pkl', 'wb') as outp:  
            pickle.dump(network, outp, pickle.HIGHEST_PROTOCOL)
        print('Network saved in ' + filename)
        return 0
    except:
        return -1


"""
    Loads the properties of network from a file, which is written in such
    a way that it has the required format. That basically means it is 
    written with exactly the save method from this handler

    Takes:      * The name of the save file

    Returns:    * The network as an object of the Neural Network class

    TODO:
        - im prinzip ganz einfach: 
            lade die daten von der gesichterten Datei ein und erstelle
            daraus mit internen methoden ein network. Orientiere dich
            an save_network
"""
def load_network(filename):
    try:
        with open(filename + '.pkl', 'rb') as inp:
            network = pickle.load(inp)
            return network
    except:
        print('Failed to load Network')
        return -1