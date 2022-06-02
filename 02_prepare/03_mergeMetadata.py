import pandas as pd

from datetime import datetime
import time


"""
Merges metadata with clone data.
Benchmarked execution time: 31.6s.
"""

dataFolder = '../01_data'
clonedataFolder = f'{dataFolder}/clonedata'
metadataFolder = f'{dataFolder}/metadata'
preparedFolder = f'{dataFolder}/prepared'

print('---Starting merge---')
start_time = time.time()

data = pd.read_pickle(f'{clonedataFolder}/duplicates/clones.p')
authors = pd.read_json(f'{metadataFolder}/authorinfo.json').transpose()
transactions = pd.read_json(f'{metadataFolder}/transactioninfo.json', typ='series')
filelength = pd.read_json(f'{metadataFolder}/filelength.json', typ='series')

def getAuthor(contract):
    try:
        authorinfo = authors.loc[contract, :]
        return authorinfo.author
    except KeyError:
        return 'N/A'
        
def getDate(contract):
    try:
        authorinfo = authors.loc[contract, :]
        t = authorinfo.time
        if(t==0):
            return t
        else:
            return datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    except KeyError:
        return 'N/A'

def getContract(fileseries):
    return fileseries.split('/')[3].split('.')[0]
    
data['author'] = data.apply(lambda row: getAuthor(getContract(row.file)), axis=1)
data['creationdate'] = data.apply(lambda row: getDate(getContract(row.file)), axis=1)
data['txnumber'] = data.apply(lambda row: transactions.loc[getContract(row.file)], axis=1)
data['filelength'] = data.apply(lambda row: filelength[getContract(row.file)], axis=1)

data = data[['type', 'classid', 'nclones', 'nlines', 'similarity', 'startline', 'endline', 'file', 'author', 'creationdate', 'filelength', 'txnumber']].reset_index(drop=True)

data.to_pickle(f'{preparedFolder}/clonesWithAuthors.p')

print('---Merge done. Execution took %s seconds ---' % (time.time() - start_time))