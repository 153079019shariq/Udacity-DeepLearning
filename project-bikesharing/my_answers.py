import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        self.activation_function = lambda x : 1/(1+np.exp(-x)) 
        
  

    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        

        ###################################
        #Without vector computation
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            
            X=np.reshape(X,(1,-1))
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)       
            self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        


    def forward_pass_train(self, X):
       
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs =self.activation_function(hidden_inputs) # signals from hidden layer
        final_inputs =  np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer    
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        error = (-final_outputs +y) # Output layer error is the difference between desired target and actual output.
        hidden_error = np.multiply(self.weights_hidden_to_output.T,(error*hidden_outputs*(1.0-hidden_outputs)))
        output_error_term = error 
        X=np.reshape(X,(1,-1))
        hidden_error_term = np.dot(X.T,hidden_error)  
        
        hidden_outputs =np.reshape(hidden_outputs,(1,-1))
        output_error_term=np.reshape(output_error_term,(1,-1))
        
        ###############################################################
        #Without vector computation
        #print("output_error_term",output_error_term.shape,hidden_outputs.shape,delta_weights_h_o.shape)
        delta_weights_i_h += hidden_error_term
        delta_weights_h_o += np.dot(output_error_term.T,hidden_outputs).T
        ##############################################################
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        
        self.weights_hidden_to_output +=delta_weights_h_o*self.lr/n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h*self.lr/n_records# update input-to-hidden weights with gradient descent step

    def run(self, features):
        
        hidden_inputs = np.dot(features,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs =self.activation_function(hidden_inputs) # signals from hidden layer
        final_inputs =  np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer  
        #print(features.shape,self.input_nodes,hidden_outputs.shape,final_outputs.shape)

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1200
learning_rate = 0.1
hidden_nodes = 6
output_nodes = 1

