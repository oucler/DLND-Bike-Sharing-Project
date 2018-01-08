import numpy as np
np.random.seed(21)

class NeuralNetworkMultiLayers(object):
     def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes,hidden3_nodes, output_nodes, learning_rate):
         # Set number of nodes in input, hidden and output layers.
         self.input_nodes = input_nodes
         self.hidden1_nodes = hidden1_nodes
         self.hidden2_nodes = hidden2_nodes
         self.hidden3_nodes = hidden3_nodes
         self.output_nodes = output_nodes
 
         # Initialize weights
         self.weights_input_to_hidden1 = np.random.normal(0.0, self.input_nodes**-0.5, 
                                        (self.input_nodes, self.hidden1_nodes))
 
         self.weights_hidden1_to_hidden2 = np.random.normal(0.0, self.hidden1_nodes**-0.5, 
                                        (self.hidden1_nodes, self.hidden2_nodes))
         
         self.weights_hidden2_to_hidden3 = np.random.normal(0.0, self.hidden2_nodes**-0.5, 
                                        (self.hidden2_nodes, self.hidden3_nodes))
                  
         self.weights_hidden_to_output = np.random.normal(0.0, self.hidden3_nodes**-0.5, 
                                        (self.hidden3_nodes, self.output_nodes))
         self.lr = learning_rate
         
         #### TODO: Set self.activation_function to your implemented sigmoid function ####
         #
         # Note: in Python, you can define a function with a lambda expression,
         # as shown below.
         self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.
         
         ### If the lambda code above is not something you're familiar with,
         # You can uncomment out the following three lines and put your 
         # implementation there instead.
         #
         #def sigmoid(x):
         #    return 0  # Replace 0 with your sigmoid calculation here
         #self.activation_function = sigmoid
                     
 
     def train(self, features, targets):
         ''' Train the network on batch of features and targets. 
         
             Arguments
             ---------
             
             features: 2D array, each row is one data record, each column is a feature
             targets: 1D array of target values
         
         '''
         n_records = features.shape[0]
         delta_weights_i_h1 = np.zeros(self.weights_input_to_hidden1.shape)
         delta_weights_h1_h2 = np.zeros(self.weights_hidden1_to_hidden2.shape)
         delta_weights_h2_h3 = np.zeros(self.weights_hidden2_to_hidden3.shape)
         delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
         for X, y in zip(features, targets):
             
             final_outputs, hidden1_outputs, hidden2_outputs,hidden3_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
             # Implement the backproagation function below
             delta_weights_i_h1, delta_weights_h1_h2,delta_weights_h2_h3, delta_weights_h_o = self.backpropagation(final_outputs, hidden1_outputs,hidden2_outputs, 
                                                                        hidden3_outputs, X, y, delta_weights_i_h1,delta_weights_h1_h2, delta_weights_h2_h3,delta_weights_h_o)
         self.update_weights(delta_weights_i_h1, delta_weights_h1_h2,delta_weights_h2_h3, delta_weights_h_o, n_records)
 
 
     def forward_pass_train(self, X):
         ''' Implement forward pass here 
          
             Arguments
             ---------
             X: features batch
 
         '''
         #### Implement the forward pass here ####
         ### Forward pass ###
         # TODO: Hidden layer - Replace these values with your calculations.
         hidden1_inputs = np.dot(X,self.weights_input_to_hidden1) # signals into hidden layer
         hidden1_outputs = self.activation_function(hidden1_inputs) # signals from hidden layer
 
         hidden2_inputs = np.dot(hidden1_outputs,self.weights_hidden1_to_hidden2)
         hidden2_outputs = self.activation_function(hidden2_inputs)
         
         hidden3_inputs = np.dot(hidden2_outputs,self.weights_hidden2_to_hidden3)
         hidden3_outputs = self.activation_function(hidden3_inputs)
         
         # TODO: Output layer - Replace these values with your calculations.
         final_inputs = np.dot(hidden3_outputs,self.weights_hidden_to_output) # signals into final output layer
         final_outputs = final_inputs#self.activation_function(final_inputs) # signals from final output layer
         
         return final_outputs, hidden1_outputs, hidden2_outputs, hidden3_outputs
 
     def backpropagation(self, final_outputs, hidden1_outputs, hidden2_outputs,hidden3_outputs,X, y, delta_weights_i_h1, delta_weights_h1_h2,delta_weights_h2_h3,delta_weights_h_o):
         ''' Implement backpropagation
          
             Arguments
             ---------
             final_outputs: output from forward pass
             y: target (i.e. label) batch
             delta_weights_i_h: change in weights from input to hidden layers
             delta_weights_h_o: change in weights from hidden to output layers
 
         '''
         #### Implement the backward pass here ####
         ### Backward pass ###
         # TODO: Output error - Replace this value with your calculations.
         error = y-final_outputs # Output layer error is the difference between desired target and actual output.
         
         # TODO: Backpropagated error terms - Replace these values with your calculations.
         output_error_term = error#*final_outputs*(1-final_outputs)
         
         hidden3_error = np.dot(self.weights_hidden_to_output,output_error_term)
         hidden3_error_term = hidden3_error * hidden3_outputs * (1 - hidden3_outputs)
         
         # TODO: Calculate the hidden layer's contribution to the error
         hidden2_error = np.dot(self.weights_hidden2_to_hidden3,hidden3_error_term)
         hidden2_error_term = hidden2_error * hidden2_outputs * (1 - hidden2_outputs)
         
        
         hidden1_error = np.dot(self.weights_hidden1_to_hidden2,hidden2_error_term)
         hidden1_error_term = hidden1_error * hidden1_outputs * (1 - hidden1_outputs)
         
        # Weight step (input to hidden)
         delta_weights_i_h1 += hidden1_error_term*X[:,None] 
         delta_weights_h1_h2 += hidden1_outputs[:,None]*hidden2_error_term
         delta_weights_h2_h3 += hidden2_outputs[:,None]*hidden3_error_term
         # Weight step (hidden to output)
         delta_weights_h_o += hidden3_outputs[:,None]*output_error_term
     
   
         return delta_weights_i_h1,delta_weights_h1_h2,delta_weights_h2_h3, delta_weights_h_o
 
     def update_weights(self, delta_weights_i_h1, delta_weights_h1_h2, delta_weights_h2_h3,delta_weights_h_o, n_records):
         ''' Update weights on gradient descent step
          
             Arguments
             ---------
             delta_weights_i_h: change in weights from input to hidden layers
             delta_weights_h_o: change in weights from hidden to output layers
             n_records: number of records
 
         '''
         self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
         self.weights_input_to_hidden1 += self.lr*delta_weights_i_h1/n_records # update input-to-hidden weights with gradient descent step
         self.weights_hidden1_to_hidden2 += self.lr*delta_weights_h1_h2/n_records # update input-to-hidden weights with gradient descent step
         self.weights_hidden2_to_hidden3 += self.lr*delta_weights_h2_h3/n_records # update input-to-hidden weights with gradient descent step
 
     def run(self, features):
         ''' Run a forward pass through the network with input features 
         
             Arguments
             ---------
             features: 1D array of feature values
         '''
         
         #### Implement the forward pass here ####
         # TODO: Hidden layer - replace these values with the appropriate calculations.
         hidden1_inputs = np.dot(features,self.weights_input_to_hidden1) # signals into hidden layer
         hidden1_outputs = self.activation_function(hidden1_inputs) # signals from hidden layer
         
         hidden2_hidden1 = np.dot(hidden1_outputs,self.weights_hidden1_to_hidden2)
         hidden2_outputs = self.activation_function(hidden2_hidden1)
         
         hidden3_hidden2 = np.dot(hidden2_outputs,self.weights_hidden2_to_hidden3)
         hidden3_outputs = self.activation_function(hidden3_hidden2)         
         
         # TODO: Output layer - Replace these values with the appropriate calculations.
         final_inputs = np.dot(hidden3_outputs, self.weights_hidden_to_output)# signals into final output layer
         final_outputs = final_inputs#self.activation_function(final_inputs) # signals from final output layer 
         
         return final_outputs
 
 
 #########################################################
 # Set your hyperparameters here
 ##########################################################
iterations = 10000
learning_rate = 0.4
hidden1_nodes = 12
hidden2_nodes = 12
hidden3_nodes = 10
output_nodes = 1

