import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import math
import seaborn as sns

mode = 14 #'all'

data = pd.read_pickle("../04_staged_data/fulldata.p")
corpusLOC = 4004543

def showplt(plt):
    if isinstance(mode, int):
        plt.show()

def observation1():
    df1 = pd.read_pickle("../04_staged_data/observation1data.p")
    
    dfg = df1.groupby(['type'])['sumlines'].sum().reset_index(name ='sumlines')
    totalCloneNumber = dfg['sumlines'].sum()
    totalClonePercentage = round((totalCloneNumber*100)/corpusLOC, 2)
    print('Ratio of clones in the corpus: {}%.'.format(totalClonePercentage))
    
    groups = ['clone-free', 'cloned']
    values = [corpusLOC-totalCloneNumber, totalCloneNumber]
    
    colors = ['#1f77b4', '#ff0000'] #'#911eb4', '#ffe119',  '#469990', '#42d4f4'

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(values, wedgeprops=dict(width=0.48), startangle=-40, colors=colors)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        #ax.annotate("{} ({}%)".format(groups[i], round(((values[i]/corpusLOC)*100), 2)), xy=(x, y), xytext=(1*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, **kw)

    ax.legend(["{} ({}%)".format(groups[i], round(((values[i]/corpusLOC)*100), 2)) for i in range(0,2)], loc='center', frameon=False)
    
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation1.pdf')

    showplt(plt)

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
    plt.gca().get_xticklabels()[1].set_color('r')
    plt.gca().get_yticklabels()[5].set_color('r')
    
    plt.axvline(x=20, color='r')
    plt.axhline(y=cumulativeClonePercentageAt20, color='r')
    plt.gca().get_xticklabels()[3].set_color('red')
    plt.gca().get_yticklabels()[10].set_color('red')
    
    
    plt.grid(axis='both')
    
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation2.pdf')
    
    showplt(plt)        
    
def observation3():
    allcontracts = pd.read_pickle("../04_staged_data/observation3data.p")
    
    quarterlyClones = allcontracts.groupby(['quarter'])[['quarter', 't1', 't2', 't2c', 't3', 't32', 't32c']].sum().reset_index()

    quarterlyClones['t1plus'] = quarterlyClones.apply(lambda row: row.t2+row.t2c+row.t3+row.t32+row.t32c, axis=1)
    quarterlyClones['all'] = quarterlyClones.apply(lambda row: row.t1+row.t1plus, axis=1)
    
    qlabels = quarterlyClones['quarter']
    x = np.arange(len(qlabels))
    width = 0.4
    
    fig, axs = plt.subplots(2)
    
    axs[0] = quarterlyClones[['all']].plot(kind='bar', ax=axs[0], width=width, rot=0)
    axs[0].set_ylabel('Number of new clones')
    axs[0].set_ylim([0, 30000])
    axs[0].bar_label(axs[0].containers[0])
    axs[0].legend(loc='upper left')
    
    
    colors = ['#911eb4', '#ffe119', '#e6194B', '#469990', '#42d4f4']
    quarterlyT1plusClones100 = quarterlyClones[['t2', 't2c', 't3', 't32', 't32c', 'all']].apply(lambda x: round(x*100/x['all'], 0), axis=1)
    quarterlyT1plusClones100 = quarterlyT1plusClones100[['t2', 't2c', 't3', 't32', 't32c']]
    axs[1] = quarterlyT1plusClones100.plot(kind='bar', stacked = True, ax=axs[1], width=width, rot=0, color=colors)
    axs[1].set_ylabel('% of new non-type-1 clones')
    axs[1].set_xlabel('Quarter')
    axs[1].legend(('type-2b', 'type-2c', 'type-3', 'type-3b', 'type-3c'), loc='upper left')
    
    container = axs[1].containers[2]
    labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
    axs[1].bar_label(container, labels=labels, label_type='center', color='white')

    for ax in axs:
        ax.set_xticklabels(qlabels)
        ax.grid(axis='y')

    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation3.pdf')
    
    print(quarterlyClones)
    
    showplt(plt)
    
def observation4():
    df1 = pd.read_pickle("../04_staged_data/observation1data.p")
    
    dfg = df1.groupby(['type'])['sumlines'].sum().reset_index(name ='sumlines')
    totalCloneNumber = dfg['sumlines'].sum()
    totalClonePercentage = round((totalCloneNumber*100)/corpusLOC, 2)
    
    groups = ['clone-free', 'type-1', 'type-3', 'other']
    values = [corpusLOC-totalCloneNumber, dfg[dfg['type']=='type-1']['sumlines'].values[0], dfg.loc[dfg['type']=='type-3']['sumlines'].values[0], dfg.loc[~dfg.type.isin(groups)].sum().values[1]]
    percValues = [round((x/corpusLOC)*100, 2) for x in values]
    
    colors = ['#1f77b4', '#fcba03', '#e6194B', '#03fcad'] #'#911eb4', '#ffe119',  '#469990', '#42d4f4'
    
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(values, wedgeprops=dict(width=0.48), startangle=-40, colors=colors)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        #ax.annotate("{} ({}%)".format(groups[i], round(((values[i]/corpusLOC)*100), 2)), xy=(x, y), xytext=(1*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, **kw)

    ax.legend(["{} ({}%)".format(groups[i], round(((values[i]/corpusLOC)*100), 2)) for i in range(0,4)], loc='center', frameon=False)
    
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation4.pdf')

    showplt(plt)
    
    #TODO: code file length comparison
    
def observation5():
    observation4()
    
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
    
    authorDf = authorDf.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
    
    nbins = int(round(len(authorDf['entropy'])/10,0))
    avg = authorDf['entropy'].mean()
    
    plt.hist(authorDf['entropy'], bins=nbins)
    plt.xlabel('Entropy')
    plt.ylabel('Number of clusters')
    
    plt.axvline(x=authorDf['entropy'].mean(), color='r')
    plt.axhline(y=authorDf['size'].median(), color='r')
    plt.xticks(list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, round(authorDf['entropy'].mean(), 2), 0.7, 0.8, 0.9, 1.0]))
    plt.gca().get_xticklabels()[6].set_color('red')
    plt.yticks(list([0, 10, authorDf['size'].median(), 20, 30, 40, 50, 60, 70, 80]))
    plt.gca().get_yticklabels()[2].set_color('red')

    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation10.pdf')
    
    showplt(plt)
    
def observation11():
    authorDf = pd.read_pickle("../04_staged_data/observation10data.p")
    
    authorDf = authorDf.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
    print(authorDf)
    
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(authorDf['size'], authorDf['entropy'], alpha=0.2)
    ax.set_xscale('log')
    
    plt.axvline(x=authorDf['size'].median(), color='r')
    plt.axhline(y=authorDf['entropy'].median(), color='r')
    plt.yticks(list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, round(authorDf['entropy'].median(),2), 0.8, 0.9, 1.0]))
    plt.gca().get_yticklabels()[7].set_color('red')
    ax.text(15.75, -0.1, str(int(authorDf['size'].median())), color='red')
    
    
    plt.xlabel('Cluster size')
    plt.ylabel('Entropy')
    
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    plt.savefig('./figures/observation11.pdf')
    
    showplt(plt)
    
def observation12():
    t1 = pd.read_csv('../00_rq3/result_csv/type-1.csv')
    # t2b = pd.read_csv('../00_rq3/result_csv/type-2b.csv')
    # t2c = pd.read_csv('../00_rq3/result_csv/type-2c.csv')
    
    # dx = pd.concat([t1, t2b, t2c])
    dx = t1
    print(dx)
    # import pdb
    # pdb.set_trace()
    filteredDx = (dx[dx['startline_y'].notnull()].drop_duplicates(subset=['filename_x']))
    unique_contracts = dx.drop_duplicates(subset=['filename_x'])

    print('Total len: {}.'.format(len(unique_contracts)))
    print('Filtered len: {}.'.format(len(filteredDx)))
    
    print('{}%'.format(round((len(filteredDx)/len(unique_contracts))*100, 2)))
    
def observation13():
    t1 = pd.read_csv('../00_rq3/result_csv/type-1.csv')
    # t2b = pd.read_csv('../00_rq3/result_csv/type-2b.csv')
    # t2c = pd.read_csv('../00_rq3/result_csv/type-2c.csv')
    
    # dx = pd.concat([t1, t2b, t2c])
    dx = t1
    filteredDx = (dx[dx['filename_y'].notnull()]).drop_duplicates(subset=['filename_x', 'startline_x', 'endline_x'])
    

    t1 = pd.read_csv('../00_rq3/corpus_contracts.csv')
    print('Total len: {}.'.format(len(t1)))
    print('Filtered len: {}.'.format(len(filteredDx)))
    
    print('{}%'.format(round((len(filteredDx)/len(t1))*100, 2)))

def observation14():
    # dx = pd.read_pickle("../04_staged_data/observation14data.p")
    t1 = pd.read_csv('../00_rq3/result_csv/type-1.csv').filename_y.apply(lambda x:'/'.join(x.split('/')[4:]) if not pd.isna(x) else x)
    dx = t1

    sum = dx
    # print(dx)
    sum = dx.value_counts().sum()

    print(sum)
    distinct = len(dx.unique())
    
    counts = dx.value_counts().rename_axis('contract').reset_index(name='count')
    
    print(counts)
    counts['cumsum'] = counts['count'].cumsum()
    
    # #counts.apply(lambda row: round((row['count'/sum)*100,2)).head(10)
    
    counts['perc'] = round((counts['count']/sum)*100, 2)
    counts['cumulativePerc'] = round((counts['cumsum']/sum)*100, 2)
    
    
    print(counts)
    
    print('Cumulative 80%:')
    print(counts.head(counts[counts.cumulativePerc > 80].index[0]))
    
    print(len(counts))
    
    print('First 20:')
    print(counts.head(int(len(counts)*0.2)))

if mode == 'all':
    for o in range(1, 12, 1):
        locals()["observation{}".format(o)]()
        plt.clf()
else:
    locals()["observation{}".format(mode)]()