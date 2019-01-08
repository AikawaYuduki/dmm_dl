import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import pandas as pd
import random

df = pd.read_csv("dmm_dl_data.csv")
#print(df.head(5))

X = df.iloc[:,1:]
Y = df.iloc[:,0]
#print(X.head(3))
#print(Y.head(3))
X = X.values.astype(np.float32)
Y = Y.values.astype(np.float32)
Y = np.reshape(Y,(X.shape[0],1))

train,test = datasets.split_dataset_random(chainer.datasets.TupleDataset(X,Y),600)
train_iter = chainer.iterators.SerialIterator(train, 32)
test_iter = chainer.iterators.SerialIterator(test, 32, repeat=False, shuffle=True)

class DMMChain(Chain):
    def __init__(self):
        super(DMMChain,self).__init__(
            l1=L.Linear(df.shape[1]-1,128),
            l2=L.Linear(128,128),
            l3=L.Linear(128,16),
            l4=L.Linear(16,1)
            )

    def __call__(self,x):
        h = F.relu(self.l1(x))
        h = F.dropout(F.relu(self.l2(h)))
        h = F.dropout(F.relu(self.l3(h)))
        return F.relu(self.l4(h))

model = L.Classifier(DMMChain(),lossfun=F.mean_squared_error)
model.compute_accuracy = False
optimizer = chainer.optimizers.Adam(alpha = 0.0001)
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

updater = training.StandardUpdater(train_iter,optimizer,device=-1)
trainer = training.Trainer(updater,(1000,"epoch"),out="result")

trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename="snapshot_epoch-{.updater.epoch}"))
trainer.extend(extensions.Evaluator(test_iter,model,device=-1))
trainer.extend(extensions.PrintReport(["epoch","main/loss","validation/main/loss","elapsed_time"]))
trainer.extend(extensions.PlotReport(["main/loss","validation/main/loss"],x_key="epoch",file_name="loss.png"))
trainer.extend(extensions.dump_graph("main/loss"))

trainer.run()

infer_net = DMMChain()
serializers.load_npz(
    'result/snapshot_epoch-500',
    infer_net, path='updater/model:main/predictor/')

for i in range(100):
    x, t = test[i]

    x = infer_net.xp.asarray(x[None, ...])
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)

    print('予測', y[0],'正解', t)