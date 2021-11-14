import math
import pandas as pd
import time

from datetime import datetime
from matplotlib import pyplot as plt


dataToPrepare = ['RQ1', 'RQ2', 'RQ3']

data = pd.read_pickle("./clonesWithAuthors.p")

def prepareRQ1():
    df = data.drop_duplicates(['type', 'classid'])[['type', 'classid', 'nclones', 'nlines']].sort_values(by='nclones', ascending=False)
    df = df.reset_index(drop=True)
    df['sumlines'] = df.apply(lambda row: row.nclones*row.nlines, axis=1)
    
    df.to_pickle('./data_rq1.p')
    
def prepareObservation3():
    start_time = time.time()
    allcontracts = pd.read_json('../02_authordata/authorinfo.json').transpose()
    allcontracts = allcontracts.loc[allcontracts['time'] != 0]
    
    allcontracts['time'] = allcontracts.apply(lambda row: getDate(row.time), axis=1)
    allcontracts['quarter'] = allcontracts.apply(lambda row: str(row.time.year)+"."+str(row.time.quarter), axis=1)
    allcontracts['t1'] =  allcontracts.apply(lambda row: getNClones(row.name, 'type-1'), axis=1)
    allcontracts['t2'] =  allcontracts.apply(lambda row: getNClones(row.name, 'type-2'), axis=1)
    allcontracts['t2c'] =  allcontracts.apply(lambda row: getNClones(row.name, 'type-2c'), axis=1)
    allcontracts['t3'] =  allcontracts.apply(lambda row: getNClones(row.name, 'type-3'), axis=1)
    allcontracts['t32'] =  allcontracts.apply(lambda row: getNClones(row.name, 'type-3-2'), axis=1)
    allcontracts['t32c'] =  allcontracts.apply(lambda row: getNClones(row.name, 'type-3-2c'), axis=1)
    
    allcontracts['nclones'] =  allcontracts.apply(lambda row: row.t1+row.t2+row.t2c+row.t3+row.t32+row.t32c, axis=1)
    
    allcontracts = allcontracts.sort_values(by=['time'], ascending=True)
    
    allcontracts.to_pickle('./data_observation3.p')
    print("---Execution took %s seconds ---" % (time.time() - start_time))
    
def getDate(tstring):
    if(tstring!=0):
        return datetime.strptime(tstring, '%Y-%m-%d %H:%M:%S')
    return tstring
    
def getNClones(contract, type):
    try:
        return len(data[(data['file'].str.contains(contract)) & (data['type'] == type)])
    except KeyError:
        return 0
        

def prepareRQ2():
    authorDf = pd.DataFrame(columns = ['cluster', 'entropy', 'size'])

    types = data['type'].unique()
    for type in types:
        print("Calculating {}".format(type))
        typeData = data[data['type']==type]
        numClasses = typeData['classid'].unique()
        for classid in numClasses:
            cluster = typeData[typeData['classid']==classid]
            if cluster.nclones.values[0] >= 10:
                clusterid = str(type)+"--"+str(classid)
                print("\t Calculating {}".format(clusterid))
                groupedCluster = cluster.groupby(['author']).size().reset_index(name="numContracts")    
                entropy = getEntropy(groupedCluster)
                authorDf = authorDf.append({'cluster':clusterid, 'entropy':entropy, 'size':groupedCluster['numContracts'].sum()}, ignore_index=True)
            
    authorDf.to_pickle('./data_rq2.p')

def getEntropy(groupedCluster):
    n = groupedCluster['numContracts'].sum()
    b = 2
    
    entropy = 0
    
    for index, row in groupedCluster.iterrows():
        pxi = row.numContracts/n
        subResult = ((pxi*math.log(pxi, b)) /( math.log(n, b)))
        entropy += subResult
    
    if entropy != 0:
        entropy = -1 * entropy
    
    return entropy

def prepareRQ3():
    t1 = pd.read_csv('../03_clones/rq3/type-1.csv')
    t2b = pd.read_csv('../03_clones/rq3/type-2b.csv')
    t2c = pd.read_csv('../03_clones/rq3/type-2c.csv')

    df = pd.concat([t1, t2b, t2c])

    df.to_pickle('./data_rq3.p')
    
for d in dataToPrepare:
    locals()["prepare{}".format(d)]()