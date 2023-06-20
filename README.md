# pytorch-siamese

This is a port of [chainer-siamese](https://github.com/mitmul/chainer-siamese) and
a fork of [pytorch-siamese](https://github.com/delijati/pytorch-siamese). It implements
additional training datasets, as well as few-shot prediction code for the siamese 
network. 

## Install

This installation requires `cuda` to be installed. 

```
$ poetry install
```

or

```
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Run

```
$ env/bin/python train_mnist.py --epoch 10
$ env/bin/python train_omniglot.py --epoch 10
$ env/bin/python train_omniglot_by_alphabet.py --epoch 10
```

This dumps for every epoch the current `state` and creates a `result.png`. Model
state is saved in a new directory unique to each run.