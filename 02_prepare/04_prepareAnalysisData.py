import math
import argparse
import numpy as np
import os
import pandas as pd
import pickle
import re
import time

from datetime import datetime
from matplotlib import pyplot as plt

from hashlib import sha256

dataFolder = '../01_data'
clonedataFolder = f'{dataFolder}/clonedata'
metadataFolder = f'{dataFolder}/metadata'
preparedFolder = f'{dataFolder}/prepared'

class Preparation():

    def toPickle(self, data, name):
        data.to_pickle(f'{preparedFolder}/{name}')

    """
    Prepares data for RQ1
    Benchmarked execution time: 0.3s.
    """
    def prepareRQ1(self):
        data = pd.read_pickle(f'{preparedFolder}/clonesWithAuthors.p')
        data = data.drop_duplicates(['type', 'classid'])[['type', 'classid', 'nclones', 'nlines']].sort_values(by='nclones', ascending=False)
        data = data.reset_index(drop=True)
        data['sumlines'] = data.apply(lambda row: row.nclones*row.nlines, axis=1)
        
        self.toPickle(data, 'data_rq1.p')

    """
    Prepares data for Observation 3
    Benchmarked execution time: 8720s. (About 2.5h.)
    """    
    def prepareObservation3(self):
        start_time_obs3 = time.time()
        data = pd.read_pickle(f'{preparedFolder}/clonesWithAuthors.p')
        allcontracts = pd.read_json(f'{metadataFolder}/authorinfo.json').transpose()
        filelength = pd.read_json(f'{metadataFolder}/filelength.json', typ='series')
        
        #In about 3.5% of contracts, the creation date cannot be retrieved. Those are dropped from the analysis of Observation 3.
        allcontracts = allcontracts.loc[allcontracts['time'] != 0]
        
        print('---Data read---')
        
        allcontracts['time'] = allcontracts.apply(lambda row: self.getDate(row.time), axis=1)
        allcontracts['quarter'] = allcontracts.apply(lambda row: str(row.time.year)+'.'+str(row.time.quarter), axis=1)
        print('---Calculating type-1---')
        allcontracts['t1'] =  allcontracts.apply(lambda row: self.getNClones(data, row.name, 'type-1'), axis=1)
        print('---Execution finished at %s seconds ---' % (time.time() - start_time_obs3))
        print('---Calculating type-2b---')
        allcontracts['t2'] =  allcontracts.apply(lambda row: self.getNClones(data, row.name, 'type-2'), axis=1)
        print('---Execution finished at %s seconds ---' % (time.time() - start_time_obs3))
        print('---Calculating type-2c---')
        allcontracts['t2c'] =  allcontracts.apply(lambda row: self.getNClones(data, row.name, 'type-2c'), axis=1)
        print('---Execution finished at %s seconds ---' % (time.time() - start_time_obs3))
        print('---Calculating type-3---')
        allcontracts['t3'] =  allcontracts.apply(lambda row: self.getNClones(data, row.name, 'type-3'), axis=1)
        print('---Execution finished at %s seconds ---' % (time.time() - start_time_obs3))
        print('---Calculating type-3b---')
        allcontracts['t32'] =  allcontracts.apply(lambda row: self.getNClones(data, row.name, 'type-3-2'), axis=1)
        print('---Execution finished at %s seconds ---' % (time.time() - start_time_obs3))
        print('---Calculating type-3c---')
        allcontracts['t32c'] =  allcontracts.apply(lambda row: self.getNClones(data, row.name, 'type-3-2c'), axis=1)
        print('---Execution finished at %s seconds ---' % (time.time() - start_time_obs3))
        
        print('---Calculating nclones---')
        allcontracts['nclones'] =  allcontracts.apply(lambda row: row.t1+row.t2+row.t2c+row.t3+row.t32+row.t32c, axis=1)
        print('---Execution finished at %s seconds ---' % (time.time() - start_time_obs3))
        
        print('---Calculating filelength---')
        allcontracts['filelength'] = allcontracts.apply(lambda row: filelength[row.name], axis=1)
        print('---Execution finished at %s seconds ---' % (time.time() - start_time_obs3))
        
        allcontracts = allcontracts.sort_values(by=['time'], ascending=True)
        
        self.toPickle(allcontracts, 'data_observation3.p')
        
    def getDate(self, tstring):
        if(tstring!=0):
            return datetime.strptime(tstring, '%Y-%m-%d %H:%M:%S')
        return tstring
        
    def getNClones(self, data, contract, type):
        try:
            df = data[(data['file'].str.contains(contract)) & (data['type'] == type)]
            nclones = len(df)
            return nclones
        except KeyError:
            return 0
            
    """
    Prepares data for RQ2
    Benchmarked execution time: 45.02s.
    """
    def prepareRQ2(self):
        data = pd.read_pickle(f'{preparedFolder}/clonesWithAuthors.p')
        authorData = pd.DataFrame(columns = ['cluster', 'entropy', 'size', 'authors'])

        types = data['type'].unique()
        for type in types:
            print("Calculating {}".format(type))
            typeData = data[data['type']==type]
            numClasses = typeData['classid'].unique()
            for classid in numClasses:
                cluster = typeData[typeData['classid']==classid]
                if cluster.nclones.values[0] >= 10:
                    clusterid = str(type)+"--"+str(classid)
                    #print("\t Calculating {}".format(clusterid))
                    groupedCluster = cluster.groupby(['author']).size().reset_index(name="numContracts")    
                    entropy = self.getEntropy(groupedCluster)
                    authorData = authorData.append({'cluster':clusterid, 'entropy':entropy, 'size':groupedCluster['numContracts'].sum(), 'authors':len(groupedCluster)}, ignore_index=True)
                
        self.toPickle(authorData, 'data_rq2.p')

    def getEntropy(self, groupedCluster):
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

    """
    Prepares data for Observation 6
    Benchmarked execution time: 12839.46s. (About 3.5h.)
    """
    def prepareObservation6(self):
        def get_classids(file_path)->int:
            a = open(file_path).read()
            b = re.findall('.*?classid="(\d+)".*', a)

            return b

        def extract_code(clone_type, file_path, classids, save_path):
            f  = open(file_path, encoding='utf-8').read()
            save_list:list[tuple] = []
            for count, i in enumerate(map(str, classids)):
                p = r'^(<class classid="{}"[\s\S]*?<\/class>)$'.format(i)
                cs = re.findall(p, f, re.MULTILINE)
                p = r'^<source[\s\S]*?>([\s\S]*?)<\/source>$'
                cs = re.findall(p, cs[0], re.MULTILINE)
                p = r'[\s\S]*?function[ ]?([\s\S]*?)\{[\s\S]*\}'
                all_fs = re.findall(p, cs[0], re.MULTILINE)
                if len(all_fs)==0:
                    print('still not caught', cs[0])
                else:
                    func = (sha256(all_fs[0].strip().encode()).hexdigest(), all_fs[0], clone_type) 
                save_list.append(func)
                print(f'class id {i} done, total count {count}')
            
            df = pd.DataFrame(save_list)
            df.columns = ['hash', 'ids', 'type']
            pickle.dump(df, open(save_path, 'wb'))

            
        def extract_functions_ids():
            print('extr')
            original_paths = ['type-1', 'type-2', 'type-2c', 'type-3-1', 'type-3-2', 'type-3-2c']
            os.makedirs('clonedata/duplicates/function-ids', exist_ok=True)
            for filepath in original_paths:
                print(f'starting {filepath}')
                classids = get_classids('clonedata/duplicates/final/'+filepath+".xml")
                print(f'classsids {len(classids)}')
                complete_file_path = 'clonedata/raw/withsource/{}'.format(filepath+'.xml')
                save_path_pickle = 'clonedata/duplicates/function-ids/{}'.format(filepath+'.p')
                extract_code(filepath, complete_file_path, classids, save_path_pickle)
            
                save_path_txt = open('clonedata/duplicates/function-ids/{}'.format(filepath+'.csv'), 'w', encoding='utf-8')
                df = pickle.load(open(save_path_pickle, 'rb'))
                print('saving file', save_path_txt)
                df.to_csv(save_path_txt)

        
        try:
            os.chdir('../01_data')
        except FileNotFoundError:
            raise FileNotFoundError('No data folder found.')
        
        extract_functions_ids()
        
    """
    Prepares data for Observation 8
    Benchmarked execution time: 4.19s.
    """    
    def prepareObservation8(self):
        data = pd.read_pickle(f'{preparedFolder}/clonesWithAuthors.p')
        data = data[data['nclones'] >= 10].reset_index(drop=True)
        
        giniData = data[['classid', 'type', 'similarity', 'nclones']].drop_duplicates(['type', 'classid']).reset_index(drop=True)
        
        giniData['gini'] = giniData.apply(lambda x: self.gini(x.name, data.loc[(data['classid']==x['classid']) & (data['type']==x['type'])]['txnumber'].tolist()), axis=1)
        
        self.toPickle(giniData, 'gini.p')

    def gini(self, rowid, x):
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return round(g, 3)
        
    """
    Prepares data for Observation 9
    Benchmarked execution time: 66.14s.
    """    
    def prepareObservation9(self):
        data = pd.read_pickle(f'{preparedFolder}/clonesWithAuthors.p')
        
        data['clusterid'] = data.apply(lambda row: row['type']+'-'+row['classid'], axis=1)
        
        data = data.drop(data[data['creationdate'] == 0].index)
        
        clusterids = data['clusterid'].unique()
        
        rankData = data[['clusterid', 'type', 'nclones']].drop_duplicates().reset_index(drop=True)
        
        rankData['topTxCreationRank'] = rankData.apply(lambda x: (data[data['clusterid'] == x['clusterid']].sort_values(by='creationdate').reset_index(drop=True))['txnumber'].idxmax()+1, axis=1)
        
        self.toPickle(rankData, 'creationrank.p')

    """
    Prepares data for RQ3
    Benchmarked execution time: 3.21s.
    """
    def prepareRQ3(self):
        t1 = pd.read_csv(f'{clonedataFolder}/openzeppelin/type-1.csv')
        t2b = pd.read_csv(f'{clonedataFolder}/openzeppelin/type-2b.csv')
        t2c = pd.read_csv(f'{clonedataFolder}/openzeppelin/type-2c.csv')    
        
        data = pd.concat([t1, t2b, t2c])

        self.toPickle(data, 'data_rq3.p')

    def runAll(self):
        print('Preparing data for every RQ and observation.\n')
        #dataToPrepare = ['RQ1', 'RQ2', 'RQ3', 'Observation3', 'Observation6', 'Observation8', 'Observation9']
        #Checked an OK: RQ1, RQ2, RQ3, Observation8, Observation9
        #To check: Observation3, Observation6
        dataToPrepare = ['RQ1', 'RQ2', 'RQ3', 'Observation8', 'Observation9'] # TODO: include the two other after testing
        for prepId in dataToPrepare:
            self.runOne(prepId)
    
    def runOne(self, prepId):
        start_time=time.time()
        print('------------------')
        print('Preparing data for {}.\n'.format(prepId))
        getattr(self, 'prepare{}'.format(prepId))()
        print('Data preparation for {} finished. Elapsed time: {}.'.format(prepId, round(time.time()-start_time, 2)))
        print('------------------')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--prepare', help='RQ or observation to prepare data for.', type=str, nargs=1)
    args = parser.parse_args()

    preparation = Preparation()
    if not args.prepare:
        preparation.runAll()
    else:
        prepId = args.prepare[0]
        if prepId.lower().startswith('rq'):
            preparation.runOne(args.prepare[0].upper())
        elif prepId.lower().startswith('observation'):
            preparation.runOne(args.prepare[0].lower().capitalize())
        else:
            raise Exception('Invalid input')