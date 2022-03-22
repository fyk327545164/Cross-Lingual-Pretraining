import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import pickle

with open("data_entropy", 'rb') as fr:
    data = pickle.load(fr)

alpha = 0.99

def plot_cross():
    f1 = plt.figure(figsize=(5,3))
    entropy = data['666666']

    data_preproc = pd.DataFrame({
        'Training Steps': [_ for _ in range(len(entropy[0][1]))],
        'layer-1': entropy[0][1],
        'layer-2': entropy[1][1],
        'layer-3': entropy[2][1],
        'layer-4': entropy[3][1],
        'layer-5': entropy[4][1],
        'layer-6': entropy[5][1]})
    data_preproc = data_preproc.ewm(alpha=(1 - alpha)).mean()

    sns.lineplot(x='Training Steps', y='value', hue='variable',
                 data=pd.melt(data_preproc, ['Training Steps']))
    plt.ylabel("Cross-Attention Distribution", fontsize='8')
    plt.xlabel("Training Steps", fontsize='8')
    plt.legend(["layer-"+str(i) for i in range(1,7)], fontsize='8')
    plt.ylim(1.0, 3.6)
    plt.savefig("images/analysis-cross.pdf")
    plt.clf()

    # plt.clf()
    f2 = plt.figure(2)
    entropy = data['000066']

    data_preproc = pd.DataFrame({
        'x': [_ for _ in range(len(entropy[5][1]))],
        'layer-5': entropy[4][1],
        'layer-6': entropy[5][1]})
    data_preproc = data_preproc.ewm(alpha=(1 - alpha)).mean()

    sns.lineplot(x='x', y='value', hue='variable',
                 data=pd.melt(data_preproc, ['x']))
    plt.ylim(1.0, 3.6)
    plt.savefig("images/analysis-cross-56.pdf")
    plt.clf()

    f3 = plt.figure(3)
    entropy = data['006600']

    data_preproc = pd.DataFrame({
        'x': [_ for _ in range(len(entropy[3][1]))],
        'layer-3': entropy[2][1],
        'layer-4': entropy[3][1]})
    data_preproc = data_preproc.ewm(alpha=(1 - alpha)).mean()

    sns.lineplot(x='x', y='value', hue='variable',
                 data=pd.melt(data_preproc, ['x']))
    plt.ylim(1.0, 3.6)
    plt.savefig("images/analysis-cross-34.pdf")
    plt.clf()    # plt.clf()

    f4 = plt.figure(4)
    entropy = data['660000']
    data_preproc = pd.DataFrame({
        'x': [_ for _ in range(len(entropy[0][1]))],
        'layer-1': entropy[0][1],
        'layer-2': entropy[1][1]})
    data_preproc = data_preproc.ewm(alpha=(1 - alpha)).mean()

    sns.lineplot(x='x', y='value', hue='variable',
                 data=pd.melt(data_preproc, ['x']))
    plt.ylim(1.0, 3.6)
    plt.savefig("images/analysis-cross-12.pdf")
    plt.clf()


def plot_self():
    f1 = plt.figure(figsize=(5,3))
    entropy = data['666666']
    data_preproc = pd.DataFrame({
        'Training Steps': [_ for _ in range(len(entropy[0][0]))],
        'layer-1': entropy[0][0],
        'layer-2': entropy[1][0],
        'layer-3': entropy[2][0],
        'layer-4': entropy[3][0],
        'layer-5': entropy[4][0],
        'layer-6': entropy[5][0]})
    data_preproc = data_preproc.ewm(alpha=(1 - alpha)).mean()

    sns.lineplot(x='Training Steps', y='value', hue='variable',
                 data=pd.melt(data_preproc, ['Training Steps']))
    plt.ylabel("Self-Attention Distribution", fontsize='8')
    plt.xlabel("Training Steps", fontsize='8')

    plt.legend(["layer-"+str(i) for i in range(1,7)], fontsize='8')
    plt.ylim(0.2, 2.8)
    plt.savefig("images/analysis-self.pdf")
    plt.clf()
    # #
    f2 = plt.figure(6)
    entropy = data['000066']
    data_preproc = pd.DataFrame({
        'x': [_ for _ in range(len(entropy[0][0]))],
        'layer-1': entropy[0][0],
        'layer-2': entropy[1][0],
        'layer-3': entropy[2][0],
        'layer-4': entropy[3][0],
        'layer-5': entropy[4][0],
        'layer-6': entropy[5][0]})
    data_preproc = data_preproc.ewm(alpha=(1 - alpha)).mean()

    sns.lineplot(x='x', y='value', hue='variable',
                 data=pd.melt(data_preproc, ['x']))
    plt.ylim(0.2, 2.8)

    plt.savefig("images/analysis-self-56.pdf")
    plt.clf()

    f3 = plt.figure(7)
    entropy = data['006600']

    data_preproc = pd.DataFrame({
        'x': [_ for _ in range(len(entropy[0][0]))],
        'layer-1': entropy[0][0],
        'layer-2': entropy[1][0],
        'layer-3': entropy[2][0],
        'layer-4': entropy[3][0],
        'layer-5': entropy[4][0],
        'layer-6': entropy[5][0]})
    data_preproc = data_preproc.ewm(alpha=(1 - alpha)).mean()

    sns.lineplot(x='x', y='value', hue='variable',
                 data=pd.melt(data_preproc, ['x']))
    plt.ylim(0.2, 2.8)
    plt.savefig("images/analysis-self-34.pdf")
    plt.clf()

    f4 = plt.figure(8)
    entropy = data['660000']
    data_preproc = pd.DataFrame({
        'x': [_ for _ in range(len(entropy[0][0]))],
        'layer-1': entropy[0][0],
        'layer-2': entropy[1][0],
        'layer-3': entropy[2][0],
        'layer-4': entropy[3][0],
        'layer-5': entropy[4][0],
        'layer-6': entropy[5][0]})
    data_preproc = data_preproc.ewm(alpha=(1 - alpha)).mean()
    sns.lineplot(x='x', y='value', hue='variable',
                 data=pd.melt(data_preproc, ['x']))
    plt.ylim(0.2, 2.8)
    plt.savefig("images/analysis-self-12.pdf")
    plt.clf()

plot_cross()
plot_self()
