import argparse
import os

__author__ = "Istvan David"
__license__ = "GPLv3"
__version__ = "1.0.0"




class DuplicateScanner():
    files = {}
    
    def run(self, rootFolder):
        print(rootFolder)
        os.chdir(rootFolder)
        
        files = []
        
        for r, d, f in os.walk(rootFolder):
            print("Scanning {}".format(r))
            for file in f:
                if file not in files:
                    files.append(os.path.join(r, file))
                else:
                    print(file)
        
        #for f in files:
        #    print(f)
            

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
    rootFolder = options.root
    
    DuplicateScanner().run(rootFolder)