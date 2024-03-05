import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--n-test', type=int, default=300,
                    help='Size of test set')
parser.add_argument('--ts-code', type=str, default='601988',
                    help='Stock code')                    

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start',series[0])
    print('Training end',series[lt-n_test-1])
    print('Testing start',series[lt-n_test])
    print('Testing end',series[lt-1])

set_seed(args.seed,args.cuda)

class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim,args.hidden),
            Mamba(self.config),
            nn.Linear(args.hidden,out_dim),
            nn.Tanh()
        )
    
    def forward(self,x):
        x = self.mamba(x)
        return x.flatten()

def PredictWithData(trainX, trainy, testX):
    clf = Net(len(trainX[0]),1)
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z,yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(xv)
    if args.cuda: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return yhat

data = pd.read_csv(args.ts_code+'.SH.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
close = data.pop('close').values
ratechg = data['pct_chg'].apply(lambda x:0.01*x).values
data.drop(columns=['pre_close','change','pct_chg'],inplace=True)
dat = data.iloc[:,2:].values
trainX, testX = dat[:-args.n_test, :], dat[-args.n_test:, :]
trainy = ratechg[:-args.n_test]
predictions = PredictWithData(trainX, trainy, testX)
time = data['trade_date'][-args.n_test:]
data1 = close[-args.n_test:]
finalpredicted_stock_price = []
pred = close[-args.n_test-1]
for i in range(args.n_test):
    pred = close[-args.n_test-1+i]*(1+predictions[i])
    finalpredicted_stock_price.append(pred)

dateinf(data['trade_date'],args.n_test)
print('MSE RMSE MAE R2')
evaluation_metric(data1, finalpredicted_stock_price)
plt.figure(figsize=(10, 6))
plt.plot(time, data1, label='Stock Price')
plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()