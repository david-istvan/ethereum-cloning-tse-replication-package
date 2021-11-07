import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import math
import seaborn as sns

mode = 'all'

data = pd.read_pickle("../04_staged_data/fulldata.p")

def showplt(plt):
    if isinstance(mode, int):
        plt.show()

def observation1():
    pass

def observation2():
    df2 = data.drop_duplicates(['type', 'classid'])[['type', 'classid', 'nclones']].sort_values(by='nclones', ascending=False)
    
    print(df2)
    
    df2 = df2.reset_index(drop=True)
    
    print(df2)
    
    sumClusters = len(df2)
    sumClones = df2['nclones'].sum()
    
    print("Number of clusters: {}.".format(sumClusters))
    print("Sum of clones: {}.".format(sumClones))
    
    df2['cumulativeClusterPercentage'] = df2.apply(lambda row: round(((row.name+1)/sumClusters)*100, 4), axis=1)
    df2['cumulativeClonePercentage'] = round((df2.nclones.cumsum()/sumClones)*100, 4)
    
    print(df2)
    
    plt.figure()
    x = df2['cumulativeClusterPercentage']
    y = df2['cumulativeClonePercentage']
    plt.plot(x,y)
    plt.xlabel('Cumulative percentage of clusters')
    plt.ylabel('Cumulative percentage of clones')
    
    cumulativeClonePercentageAt2 = df2.iloc[(df2['cumulativeClusterPercentage']-2.07).abs().argsort()[:1]]['cumulativeClonePercentage'].values[0]
    cumulativeClonePercentageAt20 = df2.iloc[(df2['cumulativeClusterPercentage']-20).abs().argsort()[:1]]['cumulativeClonePercentage'].values[0]
 
    print("Cumulative clone percentage at 2.07% cumulative cluster percentage: {}".format(cumulativeClonePercentageAt2))
    print("Cumulative clone percentage at 20% cumulative cluster percentage: {}".format(cumulativeClonePercentageAt20))
    
    plt.yticks(list([0, 10, 20, 30, 40, 50, 60, 80, 90, 100]) + [cumulativeClonePercentageAt20])
    plt.ylim([-5, 105])
    plt.xticks(list([0, 2.07, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
    plt.xlim([-5, 105])
    
    plt.axvline(x=2.07, color='r')
    plt.axhline(y=50, color='r')
    plt.axvline(x=20, color='r')
    plt.axhline(y=cumulativeClonePercentageAt20, color='r')
    
    
    plt.grid(axis='both')
    
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation2.pdf')
    
    showplt(plt)

def getDate(tstring):
    if(tstring!=0):
        return datetime.strptime(tstring, '%Y-%m-%d %H:%M:%S')
    return tstring
        
    
def observation3():
    allcontracts = pd.read_pickle("../04_staged_data/observation3data.p")
    
    quarterlyClones = allcontracts.groupby(['quarter'])[['quarter', 't1', 't2', 't2c', 't3', 't32', 't32c']].sum().reset_index()

    quarterlyClones['t1plus'] = quarterlyClones.apply(lambda row: row.t2+row.t2c+row.t3+row.t32+row.t32c, axis=1)
    quarterlyClones['all'] = quarterlyClones.apply(lambda row: row.t1+row.t1plus, axis=1)
    
    labels = quarterlyClones['quarter']
    x = np.arange(len(labels))
    width = 0.4
    
    fig, axs = plt.subplots(2)
    
    axs[0] = quarterlyClones[['all']].plot(kind='bar', ax=axs[0], width=width, rot=0)
    axs[0].set_ylabel('Number of new clones')
    axs[0].set_ylim([0, 30000])
    axs[0].bar_label(axs[0].containers[0])
    
    colors = ['#911eb4', '#ffe119', '#e6194B', '#469990', '#42d4f4']
    quarterlyT1plusClones100 = quarterlyClones[['t2', 't2c', 't3', 't32', 't32c', 'all']].apply(lambda x: round(x*100/x['all'], 0), axis=1)
    quarterlyT1plusClones100 = quarterlyT1plusClones100[['t2', 't2c', 't3', 't32', 't32c']]
    axs[1] = quarterlyT1plusClones100.plot(kind='bar', stacked = True, ax=axs[1], width=width, rot=0, color=colors)
    axs[1].set_ylabel('Number of new non-type-1 clones')
    axs[1].set_xlabel('Quarter')
    axs[1].legend(('type-2b', 'type-2c', 'type-3', 'type-3b', 'type-3c'), loc='upper left')
    
    container = axs[1].containers[2]
    labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
    axs[1].bar_label(container, labels=labels, label_type='center', color='white')

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis='y')
        ax.legend(loc='upper left')

    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation3.pdf')
    
    showplt(plt)
    
def observation4():
    pass
    
def observation5():
    pass
    
def observation6():
    pass

def observation7():
    pass
    
def observation8():
    pass
    
def observation9():
    pass
    
def observation10():
    authorDf = pd.read_pickle("../04_staged_data/observation10data.p")
    
    #cluster = data[(data['type']=='type-3-2c') & (data['classid']=='7350')]
    #print(cluster.nclones.values[0] < 10)
    authorDf = authorDf.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
    print(authorDf)
    print(authorDf['entropy'].sum())
    
    authorDf['entropy'].plot()
    plt.xlabel('Cluster')
    plt.ylabel('Entropy')
    
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation10.pdf')
    
    showplt(plt)
    
def observation11():
    authorDf = pd.read_pickle("../04_staged_data/observation10data.p")
    
    #cluster = data[(data['type']=='type-3-2c') & (data['classid']=='7350')]
    #print(cluster.nclones.values[0] < 10)
    authorDf = authorDf.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
    print(authorDf)
    
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(authorDf['size'], authorDf['entropy'], alpha=0.2)
    #hb = ax.hexbin(x = authorDf['size'], y = authorDf['entropy'], cmap ='Greys', ec="#555555", mincnt=1, gridsize=50) 
    #cb = fig.colorbar(hb, ax=ax)
    #cb.set_label('counts')
    ax.set_xscale('log')
    
    plt.axvline(x=authorDf['size'].median(), color='r')
    plt.axhline(y=authorDf['entropy'].median(), color='r')
    
    plt.xlabel('Cluster size')
    plt.ylabel('Entropy')
    
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation11.pdf')
    
    showplt(plt)


if mode == 'all':
    for o in range(1, 12, 1):
        locals()["observation{}".format(o)]()
        plt.clf()
else:
    locals()["observation{}".format(mode)]()