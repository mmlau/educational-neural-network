import numpy as np
from scipy import optimize

"""
Multiclass classification neural network.
Initialize it with with number input neurons (how many input parameter), output neurons (how many classes), 
and optional with the weight lambda (default 0). After that use the add_layer method to add a hidden layer
to this network. Note that it will be added right before the output layer. Then use the method build_network 
to create the fix network and initialize the theta matrices with random values. After the network is build 
up, its architecture can not be changed. 

Parameters:
    * NR_LAYERS = the number of layers (incl input and output)
    * NR_START = number of input neurons, i.e. number of parameters
    * NR_END = number of output neurons, i.e. number of classes
    * LAMBDA = the weigth parameter lambda
    * architecture = a list of integers, where each integer is the
                    number of neuron in the corresponding layer
    * Theta = a list of matrices, where each matrix is the theta-matrix
                    of the corresponding layer

    * debug = some paramater which you can use if you need a spy in the class
                for debugging

"""


class NeuralNetwork(object):
    NR_LAYERS = 2
    NR_START = 0
    NR_END = 0
    LAMBDA = 0

    debug = 0

    # handles which layer has how many neurons (without bias)
    architecture = []

    Theta = []

    """
        This is the constructor.

        Takes:      * number of input neurons as integer
                    * number of output neurons as integer
                    * optional parameter lambda as real
    """
    def __init__(self, nr_start, nr_end, param_lambda=0):
        self.NR_START = nr_start
        self.NR_END = nr_end
        self.architecture = [nr_start, nr_end]
        self.LAMBDA = param_lambda
        self.Theta = []

    """
    Adds a hidden Layer right before the output layer.
    Takes:      Integer, the number of neurons this layer should have.
    Returns:    Zero if sucessfull, else minus one
    """
    def add_layer(self, nr_neurons):
        try:
            # Increase nr of layers by one
            self.NR_LAYERS += 1

            # put the information in the architecture penultimate position
            self.architecture.insert(-1, nr_neurons)
            return 0
        except:
            return -1

    
    """
    Creates the whole network as a fix system. Initialize thetas with random values.
    Note: after that the architecture can not be changed.
    Takes:      Nothing
    Returns:    Zero if sucessfull, else -one
    """
    def build_network(self):
        try:
            # Theta is a list of matrices
            # Between each layer is a theta matrix
            for i in range(self.NR_LAYERS - 1):
                # theta is a M x N matrix, where M is the
                # number of neurons from the start layer plus one bias,
                # N is the number of neurons from the next layer
                ind_one = self.architecture[i]
                ind_two = self.architecture[i+1]
                mat_theta = (np.random.random(( ind_one + 1, ind_two)) - 0.5) * 10

                # Put this theta matrix in the list
                self.Theta.append(mat_theta)
            return 0
        except:
            return -1
            
    """
    This is basically the calculation behind the forward propagation.
    It takes the input-array and calculate through each layer of the
    network, until it reaches the output layer. Then it returns the
    output values.

    Takes:      * Inputs as an IxJ - matrix, where I is the batch size and
                  J is the number of input neurons.
                * Boolean self_theta: if true: use the internal 
                  class parameter theta, if false, use an external theta
                * Matrix list theta: only if self_theta is true. This should
                  have the same form as the internal

    Returns:    Outputs as an IxK - matrix, where I is the batch size and
                K is the number of output neurons. I.e. for each input vector
                it produces a vector h which includes the propabilities for
                the classes.
    """
    def calc_h(self, inputs, theta=0, self_theta=True):
        # Decide if you want to use the class theta
        if self_theta:
            theta = self.Theta

        # add bias to the inputs
        # the shape should be batch_size x (nr_inputs + 1)
        bias = np.ones((inputs.shape[0], 1))
        x = np.concatenate(( bias, inputs), axis=1)

        # Start with the first layer and calculate z
        # z1_mj = sum_k x_mk * Theta(0)_kj
        # keep the batch_size as first index
        z = np.dot( x, theta[0])

        # Calculate a, which is g(z)
        a = NeuralNetwork.sigmoid(z)

        # Now we add the bias
        a = np.concatenate(( bias, a), axis=1)

        # Let us go through each layer and repeat it unitl
        # we reach the last layer before the output layer. 
        # We can recycle the variables z and a 
        for l in range(1, self.NR_LAYERS - 2):
            # z(l)_mj = sum_k z(l-1)_mk Theta(l)_kj
            # Note that the batch size is allways the first index
            z = np.dot( a, theta[l] )
            a = NeuralNetwork.sigmoid( z )
            a = np.concatenate(( bias, a), axis=1)

        # h is the last layer, thus we only need one more propagation
        z = np.dot( a, theta[self.NR_LAYERS-2] )
        h = NeuralNetwork.sigmoid( z )
        
        # While testing I run in numerical issues for being to close
        # to zero when using the logarithm later, therefore:
        # avoid h = 1 or h = 0
        h = np.clip(h, 1e-10, 1 - 1e-10)

        return h

   
    
    """
    Calculates the cost J, where 
    J = 1/m (y_ik * log( h )_ik + (1 - y_ik) * log( 1- h )_ik 
        + 0.5 * lambda * sum_all (Theta_ijk)^2)


    Takes:      * Inputs as an IxK - matrix, where I is the batch size and
                  K is the number of input neurons.
                * The translated outputs as an IxM - matrix, where I is the batch size and
                  M is the number of output neurons.
                * Boolean self_theta: if true: use the internal 
                  class parameter theta, if false, use an external theta
                * Matrix list theta: only if self_theta is true. This should
                  have the same form as the internal
    
    Returns:    * The cost J
    """
    def cost(self, inputs, y, theta=0, self_theta=True):
        # if lambda is zero, we can save some numerical time
        if self.LAMBDA == 0:
            return self.cost_without_lambda(inputs, y, theta=theta, self_theta=self_theta)
        
        # Decide if you want to use the class theta
        if self_theta:
            theta = self.Theta
        
        # Calculating the hypothesis h with the internal method 
        h = self.calc_h(inputs, theta=theta, self_theta=self_theta)

        # It is better readable to calculate the three summation parts
        # separately
        sum1 = np.trace( np.dot(y, np.log(h).T) )
        sum2 = np.trace( np.dot(1-y, np.log(1-h).T) )
        
        # The third part is a bit more complicated
        sum3 = 0
        for l in range(self.NR_LAYERS-1):
            tmp_theta = np.copy(theta[l])
            tmp_theta[:,0] = 0

            sum3 -= np.trace( np.dot( tmp_theta, tmp_theta.T ) )

        # do not forget the minus and the normalization
        J = (-1.) * (sum1 + sum2 + self.LAMBDA * 0.5 * sum3) / y.shape[0]

        return J

    
    """
    Because this is inspired by the Coursera curse, we use the sigmoid function.
    If we want another function, e.g. relu, we can use it.

    Takes:      A numpy array or a scalar

    Returns:    A numpy array, where on each element the sigmoid function
                were applied. 
    """
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    
    """
    For the minimization we use an external solver which requires theta
    to be a 1D array. But since our theta is a list of arrays (important,
    it is NOT a ndarray), we cannot simply flatten it. Thus this workaround.

    Takes:      a list of arrays 

    Returns:    this list of arrays in a flattened form
                e.g. [ [[1,2],[3,4]], [[5,6,7],[8,9,0]] -> [1,2,3,4,5,6,7,8,9,0]
    """
    def flatten_theta(th):
        l = np.array([])
        for ith in th:
            l = np.concatenate((l, ith.flatten()))
        
        return l

    """
    Sometimes we want to bring a flattened theta back into its original form.
    Therefore this method, which resembles the theta from an 1D array.
    Theta should be Theta_klm where
    k = number of layers - 1
    l = number of neurons in the k-th layer + 1 (bias)
    m = number of neurons in the (k+1)-th layer
    Note that in this program theta is list of length k, where each list element 
    is an l,m matrix, where the range of l,m depends on k

    Takes:      a list of values, which has to be flattened EXACTLY as in the
                flatten_theta method

    Returns:    a list of matrices, which is exactly in the form of the class
                parameter theta
    """
    def deflat_theta(self, x):
        try:
            # define a counter which keeps track of the position in x
            counter = 0
            theta = []
            for i in range(self.NR_LAYERS-1):
                # define the dimension of the corresponding theta-matrix
                l, m = self.architecture[i]+1, self.architecture[i+1]
                
                # get the values which belongs to the corresponding theta matrix
                # i.e. the matrix which is the i-th element of the theta-list
                # This goes from the counter (which points at the last element of the
                # previous theta list element) to the number of elements of this theta-matrix
                theta_flat = x[counter : counter+l*m]
                counter += l*m  # update counter
                
                # append the theta matrix to the matrix list in the appropriate form
                theta.append( theta_flat.reshape((l,m)) )
            
            return theta
        except:
            print('Did not work! - Check dimensions!')
            return -1

 
    """
    That function calculates the cost for a given in- and output depending on Theta and is given to the optimizer. 
    It takes the flattened Theta list as an input and the in- and output as constant arguments.

    Takes:      * x: 1D array which consists of the entries of Theta. That is basically a flattened theta-list
                * args: tuple of input and y as matrices. Note that y should have the form of a vector for each
                        sample, which consists of zeros and a single one which indicates the class.

    Returns:    * the cost value J - basically the same as the cost method
    """
    def cost_for_optimizer(self, x, *args):
        # the given constant arguments are the input and y
        inputs, y = args

        # theta is given as an 1D array x, so we have to bring it in the appropriate form:
        theta = self.deflat_theta(x)

        # Now we can simply use the cost function
        J = self.cost(inputs, y, theta=theta, self_theta=False)

        return J

    """
    To ensure an optimal minimization progress, we give the gradient $\frac{\partial J}{\partial \Theta_{ij}}$
    to the optimizer. This gradient has the same form as theta (or in this case the flattened theta),
    where each entry dtheta_ij is the partial derivative of the cost J w.r.t. the corresponding theta_ij.

    Takes:      * x: 1D array which consists of the entries of Theta. That is basically a flattened theta-list
                * args: tuple of input and y as matrices. Note that y should have the form of a vector for each
                        sample, which consists of zeros and a single one which indicates the class.

    Returns:    * 1D array which consists of the partial derivative of the cost function w.r.t. the 
                  corresponding theta
    """
    def gradJ(self, x, *args):
        # if lambda is zero, let us save some computational time
        if self.LAMBDA == 0:
            return self.gradJ_without_lambda(x, *args)

        # the given constant arguments are the input and y
        inputs, y = args
        m = inputs.shape[0]

        # x is given as an 1D array, so we have to bring it in the appropriate form
        theta = self.deflat_theta(x)

        # grad theta have to be the same dimension as theta
        gradtheta = []
        for i in range(len(theta)):
            gradtheta.append( np.copy(theta[i]) )

        # Perform a forward propagation to calculate the a - arrays
        # for all layers. The list ab contains the same, only the we
        # add the bias for each layer. That is $\tilde{a}$ 

        # declare the lists a[l] and with bias ab[l]
        a_list = [] 
        ab_list = []

        # Since the first layer a = inputs, we start with that separately
        a_list.append(inputs)
        
        # add bias
        bias = np.ones((inputs.shape[0], 1))
        ab_list.append(np.concatenate(( bias, inputs), axis=1))
        # shape: batch_size x (nr_inputs + 1)

        # For the rest this should be done dynamically
        # i.e. this should work for any number of layers
        # Go through each layer until right before the last
        for l in range(self.NR_LAYERS - 2):
            # z(l)_mj = sum_k z(l-1)_mk * Theta(l)_kj
            # keep the batch_size as first index
            # z can be recycled and do not have to be a list
            z = np.dot( ab_list[l], theta[l] )

            # call the sigmoid function for a
            ia = NeuralNetwork.sigmoid( z )

            # put the value a, and a_tilde into it
            a_list.append( ia )
            ab_list.append( np.concatenate(( bias, ia), axis=1) )

        # h is the last layer
        z = np.dot( ab_list[-1], theta[-1] )
        h = NeuralNetwork.sigmoid( z )
        
        # avoid h = 1 or h = 0
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if h[i,j] <= 0:
                    h[i,j] += 1e-10
                if h[i,j] >= 1:
                    h[i,j] -= 1e-10

        # the difference between prediction and target
        delta = h - y

        # Now we go through each layer backwards, starting from the
        # last hidden layer. Note that we already used the output layer
        for il in range(self.NR_LAYERS-2, 0, -1):
            # gradJ(l)_ij = sum_k ab(l)_ki * delta(l+1)_kj 
            gradtheta[il] = np.dot( ab_list[il].T, delta )

            # Coincidentally the weight part of grad J is exactly a 
            # weighted theta. However, the first coloumn (which comes
            # from the bias) should be zero
            tmp_theta = np.copy(theta[il])
            tmp_theta[:,0] = 0
            gradtheta[il] += self.LAMBDA * tmp_theta

            
            # define a temporary variable var
            # var(l)_ij = sum_k delta(l+1)_ik * theta(l)_jk
            tmp_var = np.dot( delta, theta[il].T )

            # and cut off the bias terms -> var(l)_i,j+1
            tmp_var = tmp_var[:,1:]

            # calculate the new delta
            # delta(l)_ij = var(l)_ij * a_ij * (1 - a_ij) 
            delta = tmp_var * a_list[il] * (1 - a_list[il])

        # calculate the last gradJ
        # This one needs to be done separately since in the 0th layer we
        # need regularization on all neurons
        gradtheta[0] = np.dot( ab_list[0].T, delta ) + self.LAMBDA * theta[0]
        tmp_theta = np.copy(theta[0])
        tmp_theta[:,0] = 0
        gradtheta[0] += self.LAMBDA * tmp_theta

        # flat the gradient in order to use it in the optimization
        gradtheta = (1/m) * NeuralNetwork.flatten_theta(gradtheta)

        return gradtheta

  
    """
    This method minimizes the cost function J(theta) w.r.t. theta. It takes a sample of inputs as an array
    and a sample (with the same size!) of outputs as an array and calculates the theta which minimizes the 
    cost function for these in- and outputs. It also prints the optimization progress on the terminal.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html#scipy.optimize.fmin_cg

    Takes:      * inputs as an IxJ array, where I is the batch size and J is the number of input layers
                * outputs as an IxK array, where K is the number of output layers. The outputs are the translated
                  y-vectors, which only has a nonzero value at the corresponding class index, e.g. [0,1,0,0]
                * bool display (default true): tells if the progress should be printed on the screen 

    Returns:    * theta list which minimizes the cost for the given in- and output. This has the exact form as
                  the class theta.
    """
    def minimizeJ(self, inputs, y, display=True):
        
        # Initial guess x0 is theta0 flattened
        x0 = NeuralNetwork.flatten_theta(self.Theta)

        # The arguments are the in- and outputs - tuple
        # just make sure they are given in the same way
        # as requested from the cost function       
        args = (inputs, y)

        # Call the optimizer fmin_cg from the scipy package and
        # let them do the work
        res = optimize.fmin_cg(self.cost_for_optimizer, x0, fprime=self.gradJ, args=args, disp=display)

        # resemble the theta from the resulting 1D array
        res_theta = self.deflat_theta(res)

        return res_theta


##############################################################################################################
# The following methods do not contribute to your understanding, therefore they are down here. They would
# not make sense analytically but save us a lot of computational time
##############################################################################################################

    """
    Sometimes we need to calculate the cost without the weighted sum.
    Even though for us humans it would be more easy to simply set the lambda
    to zero, I want to save some  computational time and create a new method

    This is exactly the same as the method cost, however with lambda = 0 
    """
    def cost_without_lambda(self, inputs, y, theta=0, self_theta=True):
            # Decide if you want to use the class theta
        if self_theta:
            theta = self.Theta
        
        # Calculating the hypothesis h with the internal method 
        h = self.calc_h(inputs, theta=theta, self_theta=self_theta)

        # It is better readable to calculate the three summation parts
        # separately
        sum1 = np.trace( np.dot(y, np.log(h).T) )
        sum2 = np.trace( np.dot(1-y, np.log(1-h).T) )

        # do not forget the minus and the normalization
        J = (-1.) * ( sum1 + sum2 ) / y.shape[0]

        return J
    

    """
    Just as in the cost function, it saves some computational time to not
    consider the lambda part if it is zero anyway.
    """
    def gradJ_without_lambda(self, x, *args):
        # the given constant arguments are the input and y
        inputs, y = args
        m = inputs.shape[0]

        # x is given as an 1D array, so we have to bring it in the appropriate form
        theta = self.deflat_theta(x)

        # grad theta have to be the same dimension as theta
        gradtheta = np.copy( theta )

        # Perform a forward propagation to calculate the a - arrays
        # for all layers. The list ab contains the same, only the we
        # add the bias for each layer. That is $\tilde{a}$ 

        # declare the lists a[l] and with bias ab[l]
        a_list = [] 
        ab_list = []

        # Since the first layer a = inputs, we start with that separately
        a_list.append(inputs)
        
        # add bias
        bias = np.ones((inputs.shape[0], 1))
        ab_list.append(np.concatenate(( bias, inputs), axis=1))
        # shape: batch_size x (nr_inputs + 1)

        # For the rest this should be done dynamically
        # i.e. this should work for any number of layers
        # Go through each layer until right before the last
        for l in range(self.NR_LAYERS - 2):
            # z(l)_mj = sum_k z(l-1)_mk * Theta(l)_kj
            # keep the batch_size as first index
            # z can be recycled and do not have to be a list
            z = np.dot( ab_list[l], theta[l] )

            # call the sigmoid function for a
            ia = NeuralNetwork.sigmoid( z )

            # put the value a, and a_tilde into it
            a_list.append( ia )
            ab_list.append( np.concatenate(( bias, ia), axis=1) )

        # h is the last layer
        z = np.dot( ab_list[-1], theta[-1] )
        h = NeuralNetwork.sigmoid( z )
        
        # avoid h = 1 or h = 0
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if h[i,j] <= 0:
                    h[i,j] += 1e-10
                if h[i,j] >= 1:
                    h[i,j] -= 1e-10

        
        # difference between prediction and target
        delta = h - y

        # Now we go through each layer backwards, starting from the
        # last hidden layer. Note that we already used the output layer
        for il in range(self.NR_LAYERS-2, 0, -1):
            # gradJ(l)_ij = sum_k ab(l)_ki * delta(l+1)_kj 
            gradtheta[il] = np.dot( ab_list[il].T, delta )
            
            # define a temporary variable var
            # var(l)_ij = sum_k delta(l+1)_ik * theta(l)_jk
            tmp_var = np.dot( delta, theta[il].T )

            # and cut off the bias terms -> var(l)_i,j+1
            tmp_var = tmp_var[:,1:]

            # calculate the new delta
            # delta(l)_ij = var(l)_ij * a_ij * (1 - a_ij) 
            delta = tmp_var * a_list[il] * (1 - a_list[il])

        # calculate the last gradJ
        gradtheta[0] = np.dot( ab_list[0].T, delta )

        # flat the gradient in order to use it in the optimization
        gradtheta = (1/m) * NeuralNetwork.flatten_theta(gradtheta)

        return gradtheta
