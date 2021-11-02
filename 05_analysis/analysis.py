import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import math

observationNumber = 11

data = pd.read_pickle("../04_staged_data/fulldata.p")

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
    allcontracts = pd.read_pickle("../04_staged_data/observation3data.p")
    
    quarterlyClones = allcontracts.groupby(['quarter'])[['quarter', 't1', 't2', 't2c', 't3', 't32', 't32c']].sum().reset_index()
    
    print(quarterlyClones)
    
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
    authorDf = pd.read_pickle("../04_staged_data/observation10data.p")
    
    #cluster = data[(data['type']=='type-3-2c') & (data['classid']=='7350')]
    #print(cluster.nclones.values[0] < 10)
    authorDf = authorDf.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
    print(authorDf)
    print(authorDf['entropy'].sum())
    
    authorDf['entropy'].plot()
    plt.show()
    
def observation11():
    authorDf = pd.read_pickle("../04_staged_data/observation10data.p")
    
    #cluster = data[(data['type']=='type-3-2c') & (data['classid']=='7350')]
    #print(cluster.nclones.values[0] < 10)
    authorDf = authorDf.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
    print(authorDf)
    
    fig = plt.figure()
    ax = plt.gca()
    #ax.scatter(authorDf['size'], authorDf['entropy'], alpha=0.5)
    hb = ax.hexbin(x = authorDf['size'], y = authorDf['entropy'], cmap ='Greys', ec="#555555", mincnt=1, gridsize=50) 
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts')
    #ax.set_xscale('log')
    
    plt.axvline(x=authorDf['size'].median(), color='r')
    plt.axhline(y=authorDf['entropy'].median(), color='r')
    
    
    
    plt.show()

locals()["observation{}".format(observationNumber)]()