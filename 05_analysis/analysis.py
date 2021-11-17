import numpy as np
import os
import pandas as pd
import shutil
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from statistics import mean, stdev, median


mode = 8 #'all'
resultsPath = '../06_results'
corpusLOC = 4004543

def observation0():
    data = pd.read_pickle("../04_staged_data/clonesWithAuthors.p")
    print(data)

def observation1():
    df1 = pd.read_pickle("../04_staged_data/data_rq1.p")
    
    dfg = df1.groupby(['type'])['sumlines'].sum().reset_index(name ='sumlines')
    totalCloneNumber = dfg['sumlines'].sum()
    totalClonePercentage = round((totalCloneNumber*100)/corpusLOC, 2)
    report = [
        'Ratio of clones in the corpus: {}%.'.format(totalClonePercentage)
    ]
    printTextReport('01', report)
    
    ### Chart ###
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
    
    savefig('01')
    showplt(plt)

def observation2():
    data = pd.read_pickle("../04_staged_data/clonesWithAuthors.p")
    df2 = data.drop_duplicates(['type', 'classid'])[['type', 'classid', 'nclones']].sort_values(by='nclones', ascending=False)
    
    df2 = df2.reset_index(drop=True)
    
    sumClusters = len(df2)
    sumClones = df2['nclones'].sum()    
    
    df2['cumulativeClusterPercentage'] = df2.apply(lambda row: round(((row.name+1)/sumClusters)*100, 4), axis=1)
    df2['cumulativeClonePercentage'] = round((df2.nclones.cumsum()/sumClones)*100, 4)
    cumulativeClonePercentageAt2 = df2.iloc[(df2['cumulativeClusterPercentage']-2.07).abs().argsort()[:1]]['cumulativeClonePercentage'].values[0]
    cumulativeClonePercentageAt20 = df2.iloc[(df2['cumulativeClusterPercentage']-20).abs().argsort()[:1]]['cumulativeClonePercentage'].values[0]

    report = [
        "Number of clusters: {}.".format(sumClusters),
        "Sum of clones: {}.".format(sumClones),
        "Cumulative clone percentage at 2.07% cumulative cluster percentage: {}".format(cumulativeClonePercentageAt2),
        "Cumulative clone percentage at 20% cumulative cluster percentage: {}".format(cumulativeClonePercentageAt20)
    ]
    printTextReport('02', report)
    
    ### Chart ###
    
    plt.figure()
    x = df2['cumulativeClusterPercentage']
    y = df2['cumulativeClonePercentage']
    plt.plot(x,y)
    plt.xlabel('Cumulative percentage of clusters')
    plt.ylabel('Cumulative percentage of clones')
    
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
    
    savefig('02')
    showplt(plt)        
    
def observation3():
    allcontracts = pd.read_pickle("../04_staged_data/data_observation3.p")
    
    quarterlyClones = allcontracts.groupby(['quarter'])[['quarter', 't1', 't2', 't2c', 't3', 't32', 't32c']].sum().reset_index()
    quarterlyClones = quarterlyClones.rename(columns={'t1':'type-1', 't2':'type-2b', 't2c':'type-2c', 't3':'type-3', 't32':'type-3b', 't32c':'type-3c'})
    
    quarterlyClones['t1plus'] = quarterlyClones.apply(lambda row: row['type-2b']+row['type-2c']+row['type-3']+row['type-3b']+row['type-3c'], axis=1)
    quarterlyClones['all'] = quarterlyClones.apply(lambda row: row['type-1']+row.t1plus, axis=1)
    
    report = [
        ('Quarterly clones', quarterlyClones, '')
    ]
    printHtmlReport('03', report)
    
    ### Chart ###

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
    quarterlyT1plusClones100 = quarterlyClones[['type-2b', 'type-2c', 'type-3', 'type-3b', 'type-3c', 'all']].apply(lambda x: round(x*100/x['all'], 0), axis=1)
    quarterlyT1plusClones100 = quarterlyT1plusClones100[['type-2b', 'type-2c', 'type-3', 'type-3b', 'type-3c']]
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
    
    savefig('03')
    showplt(plt)
    
def observation4():
    df1 = pd.read_pickle("../04_staged_data/data_rq1.p")
    
    dfg = df1.groupby(['type'])['sumlines'].sum().reset_index(name ='sumlines')
    totalCloneNumber = dfg['sumlines'].sum()
    totalClonePercentage = round((totalCloneNumber*100)/corpusLOC, 2)
    
    ### Chart ###
    
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
    
    savefig('04')
    showplt(plt)
    
    #TODO: code file length comparison
    
def observation5():
    pass
    #Same as observation4()
    
def observation6():
    pass

def observation7():
    pass
    
def observation8():
    df = pd.read_pickle("../04_staged_data/gini.p")
    
    df['type'] = df['type'].replace(to_replace='type-2', value='type-2b')
    df['type'] = df['type'].replace(to_replace='type-3-2', value='type-3b')
    df['type'] = df['type'].replace(to_replace='type-3-2c', value='type-3c')
    
    df['nclonesperc'] = df.apply(lambda row: round((row['nclones']/(df['nclones'].max())),2), axis=1)
    df = df.sort_values(by=['nclonesperc'])
    
    
    #HEXBIN CHART
    fig = plt.figure()
    ax = plt.gca()
    
    x = df['nclonesperc']
    y = df['gini']
    hb = ax.hexbin(x, y, gridsize = 35, cmap ='binary', edgecolor='gray', mincnt=1)
    
    plt.axvline(x=round(df['nclones'].median()/df['nclones'].max(), 2), color='r')
    plt.axhline(y=df['gini'].median(), color='r')
    plt.yticks(list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, round(df['gini'].median(),2), 0.9, 1.0]))
    plt.gca().get_yticklabels()[9].set_color('red')
    ax.text(15.75, -0.1, str(int(df['nclones'].median())), color='red')

    plt.xlabel('Normaized cluster size')
    plt.ylabel('Gini-coefficient')
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')
    
    fig.set_size_inches(8, 6)
    
    savefig('08-1')
    showplt(plt)
    
    #BOX PLOT
    
    bp = df.boxplot(column=['gini'], by='type', patch_artist = True, return_type='both', medianprops=dict(linewidth=1, color='black'), whiskerprops=dict(linewidth=1, color='black'))
    #, , '#42d4f4'
    colors = ['#1f77b4', '#ffe119', '#e6194B', '#469990']
    
    for row_key, (ax,row) in bp.iteritems():
        ax.set_xlabel('')
        for i,box in enumerate(row['boxes']):
            box.set(color=colors[i], linewidth=2)
            
    plt.xlabel('Clone type')
    plt.ylabel('Gini-coefficient')
            
    savefig('08-2')
    showplt(plt)
    
    
    reportDf = df[['type']]
    reportDf = reportDf.drop_duplicates().reset_index(drop=True)
    reportDf['median'] = reportDf.apply(lambda row: median(df.loc[(df['type']==row['type'])]['gini']), axis=1)
    #reportDf['mean'] = reportDf.apply(lambda row: mean(df.loc[(df['type']==row['type'])]['gini']), axis=1)
    #reportDf['stdev'] = reportDf.apply(lambda row: stdev(df.loc[(df['type']==row['type']['gini'])]), axis=1)
    
    reportDf = reportDf.sort_values(by=['type']).reset_index(drop=True)
    
    report = [
        ('Median Gini-coefficients', reportDf[['type', 'median']], 'G=0.85 roughly equals to a cluster of 10 contracts with nine contracts having 1 transaction, and one transaction having 250.<br/>G=0.75 roughly equals to a cluster of 10 contracts with nine contracts having 1 transaction, and one transaction having 50.<br/>')
    ]
    
    printHtmlReport('08-3', report)
    
    
def observation9():
    pass
    
def observation10():
    authorDf = pd.read_pickle("../04_staged_data/data_rq2.p")
    
    authorDf = authorDf.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
    
    ### Chart ###
    
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
    
    savefig('10')
    showplt(plt)
    
def observation11():
    authorDf = pd.read_pickle("../04_staged_data/data_rq2.p")
    
    authorDf = authorDf.sort_values(by=['entropy'], ascending=False).reset_index(drop = True)
    
    authorDf['sizeperc'] = authorDf.apply(lambda row: round((row['size']/(authorDf['size'].max())),2), axis=1)
    authorDf = authorDf.sort_values(by=['sizeperc'])
    
    ### Chart ###
    
    fig = plt.figure()
    ax = plt.gca()
    
    x = authorDf['sizeperc'] #np.random.randn(8873)
    y = authorDf['entropy'] #np.random.randn(8873)
    hb = ax.hexbin(x, y, gridsize = 35, cmap ='binary', edgecolor='gray', mincnt=1)
    
    #ax.scatter(authorDf['sizeperc'], authorDf['entropy'], alpha=0.2)
    #ax.set_xscale('log')
    
    plt.axvline(x=authorDf['size'].median()/authorDf['size'].max(), color='r')
    plt.axhline(y=authorDf['entropy'].median(), color='r')
    plt.yticks(list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, round(authorDf['entropy'].median(),2), 0.8, 0.9, 1.0]))
    plt.gca().get_yticklabels()[7].set_color('red')
    #ax.text(15.75, -0.1, str(int(authorDf['sizeperc'].median())), color='red')
    
    
    
    plt.xlabel('Normalized cluster size')
    plt.ylabel('Entropy')
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')
    
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    
    savefig('11')
    showplt(plt)
    
def observation12():
    df = pd.read_csv('../03_clones/rq3/type-1.csv')

    uniqueContractsWithIdenticalOZBlock = (df[df['startline_y'].notnull()].drop_duplicates(subset=['filename_x']))
    unique_contracts = df.drop_duplicates(subset=['filename_x'])

    report = [
        'Total number of contracts: {}.'.format(len(unique_contracts)),
        'Number of distinct contracts with an OpenZeppelin record associated: {}.'.format(len(uniqueContractsWithIdenticalOZBlock)),
        'Percentage ratio: {}%.'.format(round((len(uniqueContractsWithIdenticalOZBlock)/len(unique_contracts))*100, 2))
    ]
    
    printTextReport(12, report)
    
def observation13():
    df = pd.read_csv('../03_clones/rq3/type-1.csv') 
    all_corpus_contracts = pd.read_csv('../03_clones/rq3/corpus_contracts.csv')

    openZeppelinRecords = (df[df['filename_y'].notnull()]).drop_duplicates(subset=['filename_x', 'startline_x', 'endline_x'])
    
    report = [
        'Total number of contracts: {}.'.format(len(all_corpus_contracts)),
        'Number of distinct OpenZeppelin records: {}.'.format(len(openZeppelinRecords)),
        'Percentage ratio: {}%.'.format(round((len(openZeppelinRecords)/len(all_corpus_contracts))*100, 2))
    ]
    
    printTextReport(13, report)


def observation14():
    df = pd.read_csv('../03_clones/rq3/type-1.csv') 
    ozFiles = df.filename_y.apply(lambda x: x.replace('"','').split('/')[-1] if not pd.isna(x) else x)
    
    totalOZFiles = ozFiles.value_counts().sum()
    uniqueOZFileNames = len(ozFiles.unique())
    
    ozCounts = ozFiles.value_counts().rename_axis('contract').reset_index(name='count')
    
    ozCounts['cumulativeSum'] = ozCounts['count'].cumsum()
    ozCounts['perc'] = round((ozCounts['count']/totalOZFiles)*100, 2)
    ozCounts['cumulativePerc'] = round((ozCounts['cumulativeSum']/totalOZFiles)*100, 2)
    
    report = [
        ('Head 10', ozCounts.head(10), ''),
        ('Cumulative 80%', ozCounts.head(ozCounts[ozCounts.cumulativePerc > 80].index[0]), ''),
        ('First 20', ozCounts.head(int(len(ozCounts)*0.2)), '')
    ]
    
    printHtmlReport(14, report)

########################## HELPER METHODS ##########################
def showplt(plt):
    if isinstance(mode, int):
        plt.show()
        
def savefig(observationNumber):
    plt.savefig('{}/observation{}.pdf'.format(resultsPath, observationNumber))
        
def printTextReport(observationNumber, reports):
    f = open('{}/observation{}.txt'.format(resultsPath, observationNumber), 'w')
    for r in reports:
        f.write(r+'\n')
        if mode != 'all':
            print(r)
    f.close()
    
def printHtmlReport(observationNumber, reports):
    f = open('{}/observation{}.html'.format(resultsPath, observationNumber), 'w')
    for title, df, comment in reports:
        f.write('<h2>{}</h2>'.format(title))
        f.write(df.to_html())
        f.write('<p>{}</p>'.format(comment))
        if mode != 'all':
            print(title)
            print(df)
            print(comment)
            print('\n')
    f.close()

############################### MAIN ###############################

if not os.path.exists(resultsPath):
    os.mkdir(resultsPath)
    
if mode == 'all':
    if os.path.exists(resultsPath) and os.path.isdir(resultsPath):
        shutil.rmtree(resultsPath)
    os.mkdir(resultsPath)
    
    for o in range(1, 15, 1):
        print('Analyzing observation {}.'.format(o))
        locals()["observation{}".format(o)]()
        plt.clf()
else:
    locals()["observation{}".format(mode)]()