import argparse
import json
import os
import requests
import re
from bs4 import BeautifulSoup
from scapy.all import *
import csv
import html5lib
import urllib.request
import re
import argparse
import time



__author__ = "Istvan David"
__license__ = "GPLv3"
__version__ = "1.0.0"


class AppURLopener(urllib.request.FancyURLopener):
    version = "App/1.7"

class Scraper():
    
    hexPattern = "0[xX][0-9a-fA-F]+"
    hexPattern2 = "[0-9a-fA-F]+\.sol"

    def __init__(self, rootFolder):
        self._rootFolder = rootFolder
        self._opener = AppURLopener()
    
    def getURL(self, contractAddress):
        return "https://etherscan.io/txs?a=0x{}&f=5".format(contractAddress)
        
    def getURL2(self, contractAddress):
        return "https://etherscan.io/address/{}".format(contractAddress)
    
    def getAddress(self, fileName):
        print(fileName)
        return re.search(self.hexPattern2, fileName).group(0).split('.')[0]
    
    def run(self):
        '''
        contracts = {}
        
        print("===RUN===")
        print(self._rootFolder)
        os.chdir(self._rootFolder)        
        
        dir = ""
        
        for r, d, f in os.walk(rootFolder):
            print("Scanning {}".format(r))
            for contract in f:
                if contract not in contracts:
                    contracts[self.getAddress(contract)] = {'author':0, 'time':0}
                    #contracts.append(os.path.join(r, contract))
        
        with open('authorinfo.json', 'w') as outfile:
            print("dumping json")
            json.dump(contracts, outfile, indent=4, sort_keys=True)
        
        return
        '''
        
        contractData = None
        
        contractfilespath = rootFolder+"\\authorinfo.json"
        
        with open(contractfilespath) as data_file:
            contractData = json.load(data_file)
        
        contractIndex = 0
        for address, data in contractData.items():
            try:
                creator = data['author']
                creationTime = data['time']
                if not creator:
                    print("{} [{}/33k - {}%]".format(address, contractIndex, round(contractIndex/330, 2)))
                    c, t = self.getCreationData(address)
                    #print(c)
                    #print(t)
                    contractData[address]= {'author':c, 'time':t}
            except:
                self.dumpCreatorData(contractfilespath, contractData)
                return
                
            contractIndex+=1
            
        self.dumpCreatorData(contractfilespath, contractData)
    
    def dumpCreatorData(self, contractfilespath, contractData):
        with open(contractfilespath, 'w') as outfile:
            print("dumping json")
            json.dump(contractData, outfile, indent=4, sort_keys=True)

    def getCreationData(self, address):
        time.sleep(0.8)
        
        url = self.getURL(address)
        response = self._opener.open(url)
        
        pattern = "Contract\S*Creator:[\S\s]*0[xX][0-9a-fA-F]+"
        
        #body=str(response.read())
        soup = BeautifulSoup(response.read(), 'html.parser')
        
        if 'There are no matching entries' in soup.text:
            url = self.getURL2(address)
            response = self._opener.open(url)
            soup = BeautifulSoup(response.read(), 'html.parser')
            
            x = soup.findAll('a', {'title' : 'Creator Address'})[0]
            creator = x.attrs['href'].split('/')[2]
            
            #print(creator)
            
            return (creator, 0)
        
        tbody = soup.find_all('tbody')[0]
        tds = tbody.find_all('td')
        
        ageTd = 0
        i = 0
        for td in tds:
            if 'class' in td.attrs:
                if td.attrs['class'][0] == 'showAge':
                    ageTd = i
            i+=1
        
        creationTime = tds[ageTd].find('span').attrs['title']
        #print(creationTime)
        creator = tds[ageTd+1].find('a').attrs['href'].split('/')[2]
        #print(creator)
        return (creator, creationTime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-root",
        default="c:",
        help=("Provide root directory without trailing separator."
              "Example '-root d:\my_folder'"
              )
        )
        
    options = parser.parse_args()
    #rootFolder = options.root
    rootFolder = "d:\\GitHub\\sol-contracts\\contracts"
    Scraper(rootFolder).run()