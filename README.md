# MambaStock: Selective state space model for stock prediction

Mamba (Structured state space sequence models with selection mechanism and scan module, S6) has achieved remarkable success in sequence modeling tasks. This paper proposes a Mamba-based model to predict the stock price.

## Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy==1.16.5
matplotlib==3.1.0
sklearn==0.21.3
pandas==0.25.1
pytorch==1.7.1
```

The stock data used in this repository was downloaded from [TuShare](https://tushare.pro/). The stock data on [TuShare](https://tushare.pro/) are with public availability. Some code of the Mamba model is from https://github.com/alxndrTL/mamba.py

## Usage

```
python main.py
```

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
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
```

## Citation

```
@article{shi2024mamba,
  title={MambaStock: Selective state space model for stock prediction},
  author={Zhuangwei Shi},
  journal={arXiv preprint arXiv:2402.18959},
  year={2024},
}
```