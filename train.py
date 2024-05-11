# [Libraries]{Operations}
import random
import time 
import math
# [Libraries]{Visualization}
import matplotlib.pyplot as plt
# [Libraries]{Local Docs}
from data import *
from model import *


# [Function]{Random Item}
def randomChoice(l):
    """
    This function gets a random item from a list.
    """
    return l[random.randint(0, len(l) - 1)]

# [Function]{Random Category Random Line}
def randomTrainingPair():
    """
    This funciton gets a random training pair obtaining a random category and a random line from that category
    """
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# [Function]{One-hot Vector Category}
def categoryTensor(category):
    """
    This function gets One-hot vector for category
    """
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# [Function]{One-hot matrix of to last letters}
def inputTensor(line):
    """
    This function gets a One-hot matrix of first to last letters (not including EOS) for input
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# [Function]{LongTensor of second letters}
def targetTensor(line):
    """
    This function gets the LongTensor of second letters to end (EOS) for target
    """
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

# [Function]{Tensor Conversion}
def randomTrainingExample():
    """
    This function fetches a radom category and a radom line pair and turns them into the required tensor
    """
    # Getting a random pair training 
    category, line = randomTrainingPair()
    # Converting the category into tensor
    category_tensor = categoryTensor(category)
    # Converting the line into tensor
    input_line_tensor = inputTensor(line)
    # Converting target line into tensor
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


# [Hyperparamter]{Loss Criterion}
criterion = nn.NLLLoss()

# [Hyperparameter]{Learning Rate}
learning_rate = 0.0005

# [Function]{Training Function}
def train(category_tensor, input_line_tensor, target_line_tensor):
    """
    This function is for training the neural network
    """
    target_line_tensor.unsqueeze_(-1)
    # Initializing hidden state
    hidden = rnn.initHidden()
    # Setting gradients to zero
    rnn.zero_grad()
    # Initializing loss
    loss = torch.Tensor([0])
    # Calculating loss at every step
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        # Summing losses at each step
        loss += l

    # Doing backward propagation
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)

# [Function]{Time function}
def timeSince(since):
    """
    This function is to keep track of how long training takes and returns a human readable string   
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

rnn = RNN(n_letters, 128, n_letters)

# [Hyperparameters]
n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0

# Starting time 
start = time.time()

# Printing loss every print_every in n_iters
for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

# Plotting losses
plt.figure()
plt.plot(all_losses)