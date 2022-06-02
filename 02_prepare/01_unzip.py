import zipfile

dataFolder = '../01_data'

zippedFileNames = ['clonedata/openzeppelin']


def getFullPath(fileName):
    return '{}/{}'.format(dataFolder, fileName)
    
for fileName in zippedFileNames:
    fullPath = getFullPath(fileName)
    print('Unzipping {}.'.format(fullPath))
    with zipfile.ZipFile(fullPath+'.zip', 'r') as zip_ref:
        zip_ref.extractall(fullPath)

print('Unzipping done.')