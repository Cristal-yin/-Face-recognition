import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# fake data
x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np  = x.data.numpy()


y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
