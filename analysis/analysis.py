import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import math

observationNumber = 3

data = pd.read_pickle("../fulldata/fulldata.p")

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
    
    #df2[['cumulativeClusterPercentage', 'cumulativeClonePercentage']].plot()
    
    plt.figure()
    x = df2['cumulativeClusterPercentage']
    y = df2['cumulativeClonePercentage']
    plt.plot(x,y)
    
    plt.axvline(x=20, color='r')
    cumulativeClonePercentageAt20 = df2.iloc[(df2['cumulativeClusterPercentage']-20).abs().argsort()[:1]]['cumulativeClonePercentage'].values[0]
    
    print("Cumulative clone percentage at 20% cumulative cluster percentage: {}".format(cumulativeClonePercentageAt20))
    
    plt.axhline(y=cumulativeClonePercentageAt20, color='r')
    
    plt.show()

def getDate(tstring):
    if(tstring!=0):
        return datetime.strptime(tstring, '%Y-%m-%d %H:%M:%S')
    return tstring
        
    
def observation3():
    allcontracts = pd.read_pickle("../data/observation3data-detailed.p")
    
    print(allcontracts)
    
    quarterlyClones = allcontracts.groupby(['quarter'])[['quarter', 't1', 't2', 't2c', 't3', 't32', 't32c']].sum().reset_index()
    
    print(quarterlyClones)
    
    #quarterlyClones.plot.bar(x='quarter', y='nclones', rot=0)
    
    labels = quarterlyClones['quarter']
    x = np.arange(len(labels))
    width = 0.1
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2.5*width, quarterlyClones['t1'], width, label='t1')
    rects2 = ax.bar(x - 1.5*width, quarterlyClones['t2'], width, label='t2')
    rects3 = ax.bar(x - 0.5*width, quarterlyClones['t2c'], width, label='t2c')
    rects4 = ax.bar(x + 0.5*width, quarterlyClones['t3'], width, label='t3')
    rects5 = ax.bar(x + 1.5*width, quarterlyClones['t32'], width, label='t32')
    rects6 = ax.bar(x + 2.5*width, quarterlyClones['t32c'], width, label='t32c')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of new clones')
    ax.set_title('Quarter')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=1)
    ax.bar_label(rects2, padding=1)
    ax.bar_label(rects3, padding=1)
    ax.bar_label(rects4, padding=1)
    ax.bar_label(rects5, padding=1)
    ax.bar_label(rects6, padding=1)

    fig.tight_layout()
    
    plt.show()
    
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
    authorDf = pd.read_pickle("../data/observation10data.p")
    
    
    
    #cluster = data[(data['type']=='type-3-2c') & (data['classid']=='7350')]
    #print(cluster.nclones.values[0] < 10)
    print(authorDf.sort_values(by=['entropy'], ascending=False))
    print(authorDf['entropy'].sum())
    

def getEntropy(cluster):
    groupedCluster = cluster.groupby(['author']).size().reset_index(name="numContracts")    
    n = groupedCluster['numContracts'].sum()
    b = 2
    
    entropy = 0
    
    for index, row in groupedCluster.iterrows():
        pxi = row.numContracts/n
        subResult = row.numContracts * ((pxi*math.log(pxi, b)) /( math.log(n, b)))
        entropy += subResult
    
    entropy = -1 * entropy
    
    return entropy
    
def observation11():
    pass    

locals()["observation{}".format(observationNumber)]()