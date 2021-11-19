import pandas as pd

from datetime import datetime


data = pd.read_pickle("../03_clones/rq1-rq2/clones.p")
authors = pd.read_json('../02_metadata/authorinfo.json').transpose()
transactions = pd.read_json('../02_metadata/transactioninfo.json', typ='series')
filelength = pd.read_json('../02_metadata/filelength.json', typ='series')

"""
data['type'] = data['type'].replace(to_replace='type-2', value='type-2b')
data['type'] = data['type'].replace(to_replace='type-3-2', value='type-3b')
data['type'] = data['type'].replace(to_replace='type-3-2c', value='type-3c')
"""
def getAuthor(contract):
    try:
        authorinfo = authors.loc[contract, :]
        return authorinfo.author
    except KeyError:
        return "N/A"
        
def getDate(contract):
    try:
        authorinfo = authors.loc[contract, :]
        t = authorinfo.time
        if(t==0):
            return t
        else:
            return datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    except KeyError:
        return "N/A"

def getContract(fileseries):
    return fileseries.split('/')[3].split('.')[0]
    
data['author'] = data.apply(lambda row: getAuthor(getContract(row.file)), axis=1)
data['creationdate'] = data.apply(lambda row: getDate(getContract(row.file)), axis=1)
data['txnumber'] = data.apply(lambda row: transactions.loc[getContract(row.file)], axis=1)
data['filelength'] = data.apply(lambda row: filelength[getContract(row.file)], axis=1)

data = data[['type', 'classid', 'nclones', 'nlines', 'similarity', 'startline', 'endline', 'file', 'author', 'creationdate', 'filelength', 'txnumber']].reset_index(drop=True)

data.to_pickle('./clonesWithAuthors.p')