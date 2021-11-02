import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import time
import math


observationsToPrepareDataFor = [10]

data = pd.read_pickle("./fulldata.p")


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
    
    allcontracts.to_pickle('./observation3data.p')
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
        

def prepareObservation10():
    authorDf = pd.DataFrame(columns = ['cluster', 'entropy'])

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
                entropy = getEntropy(cluster)
                authorDf = authorDf.append({'cluster':clusterid, 'entropy':entropy}, ignore_index=True)
            
    authorDf.to_pickle('./observation10data.p')

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
        
for o in observationsToPrepareDataFor:
    locals()["prepareObservation{}".format(o)]()