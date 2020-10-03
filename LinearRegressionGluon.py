from mxnet import autograd, nd
from mxnet import init
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet import gluon
# import random

# create the sample data
num_input = 2
num_example = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_example, num_input))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# set the data iterator
batch_size = 40
# 将训练数据的特征和标签组合
data_set = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(data_set, batch_size, shuffle=True)

# create model
net = nn.Sequential()
net.add(nn.Dense(1))

# init the model parameters
net.initialize(init.Normal(sigma=0.01))

# define the loss function
loss = gloss.L2Loss()

# define the optimize algorithm
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# optimize the process
epochs = 3
for epoch in range(1, epochs+1):
    for X, y in data_iter:
        with autograd.record():
            lss = loss(net(X), y)
        lss.backward()
        trainer.step(batch_size)
    train_loss = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, train_loss.mean().asnumpy()))
