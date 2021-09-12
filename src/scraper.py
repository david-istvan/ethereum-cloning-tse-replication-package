import argparse
import json
import os
import requests
import re
from bs4 import BeautifulSoup
from scapy.all import *
import csv
import html5lib
import time
import urllib.request
import re
import argparse
import time
import os



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
                    contracts[self.getAddress(contract)] = 0
                    #contracts.append(os.path.join(r, contract))
        
        with open('contractfiles.json', 'w') as outfile:
            print("dumping json")
            json.dump(contracts, outfile, indent=4, sort_keys=True)
        
        return
        '''
        
        contractData = None
        
        contractfilespath = rootFolder+"\\contractfiles.json"
        contractfilespath2 = rootFolder+"\\contractfiles-2.json"
        
        with open(contractfilespath) as data_file:
            contractData = json.load(data_file)
            
        for address, creator in contractData.items():
            try:
                if not creator:
                    print(address)
                    c = self.getCreator(address)
                    contractData[address]=c
            except:
                self.dumpCreatorData(contractfilespath, contractData)
                return
                
        self.dumpCreatorData(contractfilespath, contractData)
    
    def dumpCreatorData(self, contractfilespath, contractData):
        with open(contractfilespath, 'w') as outfile:
            print("dumping json")
            json.dump(contractData, outfile, indent=4, sort_keys=True)

    def getCreator(self, address):
        time.sleep(0.8)
        authorinfo = {}
        outfilepath = rootFolder+"\\authorinfo.json"
        
        url = self.getURL(address)
        response = self._opener.open(url)
        
        pattern = "Contract\S*Creator:[\S\s]*0[xX][0-9a-fA-F]+"
        
        body=str(response.read())
        s = re.search(pattern, body)
        #print(s)
        rawCreator = re.search(self.hexPattern, s.group(0))
        creator = rawCreator.group(0)
        
        return creator


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