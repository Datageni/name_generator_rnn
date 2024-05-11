# [Libraries]{Deep Learning}
import torch 
import torch.nn as nn
# [Library]{Local Docs}
from data import *

# [Class]{Recurrrent Neural Network}
class RNN(nn.Module):
    # [Constructor]{Initializing Components}
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # [Layer]{Linear Layer}(Outputs hidden_size vector)
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        # [Layer]{Linear Layer}(Outputs output_size vector)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        # [Layer]{Linear Layer}(Outputs final output_size vector)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        # [Layer]{Dropout Layer}(Helps prevent overfitting)
        self.dropout = nn.Dropout(0.1)
        # [Layer]{Softmax Layer}(Normalizes the output probability)
        self.softmax = nn.LogSoftmax(dim=1)

    # [Method]{Forward Propagation}
    def forward(self, category, input, hidden):
        """
        Defines the forward pass of the RNN
        """
        # Passing the combined input
        input_combined = torch.cat((category, input, hidden), 1)
        # Passing the input data into the first layer
        hidden = self.i2h(input_combined)
        # Passing the input data into de output layer
        output = self.i2o(input_combined)
        # Getting the combined output
        output_combined = torch.cat((hidden, output),1)
        # Passing the output_combined to the the o2o layer
        output = self.o2o(output_combined)
        # Passing output data into dropout layer
        output = self.dropout(output)
        # Passing the output through the softmax layer
        output = self.softmax(output)
        # Returining the output and the hidden 
        return output, hidden
    
    # [Method]{Hidden State Initialization}
    def initHidden(self):
        """
        Initializes the hidden state to a tensor of zeros at the start of processing a sequence.
        """
        return torch.zeros(1, self.hidden_size)



