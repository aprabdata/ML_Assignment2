import numpy as np
import pandas as pd
import os
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
import random

from sklearn.datasets import make_classification
from torch import nn, optim
from skorch import NeuralNetClassifier
from pyperch.neural.backprop_nn import BackpropModule
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

from skorch.callbacks import EpochScoring
from pyperch.neural.ga_nn import GAModule
from pyperch.neural.rhc_nn import RHCModule
from pyperch.neural.sa_nn import SAModule

#Wine Dataset
# fetch wine dataset
def fetch_wine_data():
    wine = fetch_ucirepo(id=109)

    # data (as pandas dataframes)
    X = wine.data.features
    y = wine.data.targets

    # metadata
    print(wine.metadata)

    # variable information
    print(wine.variables)
    return X, y



if __name__ == "__main__":
    #current_dir = (os.getcwd())
    #df = pd.read_csv(os.path.join(current_dir, 'wine.data'), header=None)
    #X_wine = df.iloc[:, 1:]
    #y_wine = df.iloc[:, 0]

    X_wine, y_wine = fetch_wine_data()
    random.seed(251)

    X_normalized = MinMaxScaler().fit_transform(X_wine)

    X = X_normalized
    y = y_wine.values

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    fig, ax = plt.subplots(2,2, constrained_layout=True)
    fig_val, ax_val = plt.subplots(1, constrained_layout=True)
    fig_time, ax_time = plt.subplots(1, constrained_layout=True)

    ##Backprop
    #fig_backprop, ax_backprop = plt.subplots(1, constrained_layout=True)
    net_backprop = NeuralNetClassifier(
        module=BackpropModule,
        module__input_dim=13,
        module__output_dim=4,
        module__hidden_units=20,
        module__hidden_layers=2,
        max_epochs=3000,
        verbose=0,
        callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), ],
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        lr=.05,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    net_backprop.fit(X, y)

    # plot the iterative learning curve (loss)
    ax[0,0].plot(net_backprop.history[:, 'train_loss'], label='Train Loss', c='red', linestyle="solid")
    ax[0,0].plot(net_backprop.history[:, 'valid_loss'], label='Validation Loss', c='red', linestyle="dotted")
    # plot the iterative learning curve (accuracy)
    ax[0,0].plot(net_backprop.history[:, 'train_acc'], label='Train Acc', c='green', linestyle="solid")
    ax[0,0].plot(net_backprop.history[:, 'valid_acc'], label='Validation Acc', c='green', linestyle="dotted")
    print("Backprop: Best Validation Accuracy" + str(max(net_backprop.history[:, 'valid_acc'])))

    # plot time
    ax_time.plot(net_backprop.history[:, 'dur'], label='Backprop', c='c', linestyle="solid")
    #plot validation
    ax_val.plot(net_backprop.history[:, 'valid_acc'], label='Backprop', c='c', linestyle="solid")

    ax[0,0].set_xlabel("Iteration")
    ax[0,0].set_ylabel("Loss or Accuracy")
    ax[0,0].set_title("Backprop")
    ax[0,0].grid(visible=True, alpha=0.1)


    ##GA
    #fig_GA, ax_GA = plt.subplots(1, constrained_layout=True)
    net_GA = NeuralNetClassifier(
        module=GAModule,
        module__input_dim=13,
        module__output_dim=4,
        module__hidden_units=20,
        module__hidden_layers=2,
        module__population_size=40,
        module__to_mate=15,
        module__to_mutate=5,
        max_epochs=3000,
        verbose=0,
        callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), ],
        # use nn.CrossEntropyLoss instead of default nn.NLLLoss
        # for use with raw prediction values instead of log probabilities
        criterion=nn.CrossEntropyLoss(),
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )

    GAModule.register_ga_training_step()
    net_GA.fit(X, y)

    # plot the iterative learning curve (loss)
    ax[0, 1].plot(net_GA.history[:, 'train_loss'], label='Train Loss', c='red', linestyle="solid")
    ax[0, 1].plot(net_GA.history[:, 'valid_loss'], label='Validation Loss', c='red', linestyle="dotted")
    # plot the iterative learning curve (accuracy)
    ax[0, 1].plot(net_GA.history[:, 'train_acc'], label='Train Acc', c='green', linestyle="solid")
    ax[0, 1].plot(net_GA.history[:, 'valid_acc'], label='Validation Acc', c='green', linestyle="dotted")
    print("GA: Best Validation Accuracy" + str(max(net_GA.history[:, 'valid_acc'])))

    # plot time
    ax_time.plot(net_GA.history[:, 'dur'], label='GA', c='m', linestyle="solid")
    # plot validation
    ax_val.plot(net_GA.history[:, 'valid_acc'], label='GA', c='m', linestyle="solid")

    ax[0, 1].set_xlabel("Iteration")
    ax[0, 1].set_ylabel("Loss or Accuracy")
    ax[0, 1].set_title("GA")
    ax[0, 1].grid(visible=True, alpha=0.1)

    ## RHC
    net_RHC = NeuralNetClassifier(
        module=RHCModule,
        module__input_dim=13,
        module__output_dim=4,
        module__hidden_units=20,
        module__hidden_layers=2,
        module__step_size=.05,
        max_epochs=3000,
        verbose=0,
        callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), ],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    RHCModule.register_rhc_training_step()
    net_RHC.fit(X, y)

    # plot the iterative learning curve (loss)
    ax[1, 0].plot(net_RHC.history[:, 'train_loss'], label='Train Loss', c='red', linestyle="solid")
    ax[1, 0].plot(net_RHC.history[:, 'valid_loss'], label='Validation Loss', c='red', linestyle="dotted")
    # plot the iterative learning curve (accuracy)
    ax[1, 0].plot(net_RHC.history[:, 'train_acc'], label='Train Acc', c='green', linestyle="solid")
    ax[1, 0].plot(net_RHC.history[:, 'valid_acc'], label='Validation Acc', c='green', linestyle="dotted")
    print("RHC: Best Validation Accuracy" + str(max(net_RHC.history[:, 'valid_acc'])))

    # plot time
    ax_time.plot(net_RHC.history[:, 'dur'], label='RHC', c='y', linestyle="solid")
    # plot validation
    ax_val.plot(net_RHC.history[:, 'valid_acc'], label='RHC', c='y', linestyle="solid")

    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].set_ylabel("Loss or Accuracy")
    ax[1, 0].set_title("RHC")
    ax[1, 0].grid(visible=True, alpha=0.1)


    ## SA
    net_SA = NeuralNetClassifier(
        module=SAModule,
        module__input_dim=13,
        module__output_dim=4,
        module__hidden_units=20,
        module__hidden_layers=2,
        module__step_size=.1,
        module__t=50000,
        module__cooling=.99,
        max_epochs=3000,
        verbose=0,
        callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), ],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    SAModule.register_sa_training_step()
    net_SA.fit(X, y)

    # plot the iterative learning curve (loss)
    ax[1, 1].plot(net_SA.history[:, 'train_loss'], label='Train Loss', c='red', linestyle="solid")
    ax[1, 1].plot(net_SA.history[:, 'valid_loss'], label='Validation Loss', c='red', linestyle="dotted")
    # plot the iterative learning curve (accuracy)
    ax[1, 1].plot(net_SA.history[:, 'train_acc'], label='Train Acc', c='green', linestyle="solid")
    ax[1, 1].plot(net_SA.history[:, 'valid_acc'], label='Validation Acc', c='green', linestyle="dotted")
    print("SA: Best Validation Accuracy" + str(max(net_SA.history[:, 'valid_acc'])))

    # plot time
    ax_time.plot(net_SA.history[:, 'dur'], label='SA', c='chartreuse', linestyle="solid")
    # plot validation
    ax_val.plot(net_SA.history[:, 'valid_acc'], label='SA', c='chartreuse', linestyle="solid")

    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_ylabel("Loss or Accuracy")
    ax[1, 1].set_title("SA")
    ax[1, 1].grid(visible=True, alpha=0.1)

    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=2)
    # fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.suptitle("Loss Curve and Accuracy Curve")
    # fig.savefig('SA ' + problem_name + '.png')
    fig.savefig('NN-Individual Compare.png', bbox_inches='tight')

    ax_time.set_xlabel("Iteration")
    ax_time.set_ylabel("Time")
    ax_time.set_title("Time Vs Iteration")
    ax_time.grid(visible=True, alpha=0.1)
    ax_time.legend(frameon=False)
    fig_time.savefig('NN-Time Compare.png')

    ax_val.set_xlabel("Iteration")
    ax_val.set_ylabel("Accuracy")
    ax_val.set_title("Validation Curve")
    ax_val.grid(visible=True, alpha=0.1)
    ax_val.legend(frameon=False)
    fig_val.savefig('NN-Validation Compare.png')

    plt.clf()

    #print("Hello World - NN Optimization")

"""
Reference:
mlrose__hiive: https://github.com/hiive/mlrose/

pyperch: https://github.com/jlm429/pyperch/
"""



