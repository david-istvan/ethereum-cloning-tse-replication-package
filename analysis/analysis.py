import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime

observationNumber = 2

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
    
def observation3():
    pass
    
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
    authorGroups = data.groupby(['author']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    print(len(authorGroups))
    print(sum(authorGroups.counts))
    
def observation11():
    pass    

locals()["observation{}".format(observationNumber)]()