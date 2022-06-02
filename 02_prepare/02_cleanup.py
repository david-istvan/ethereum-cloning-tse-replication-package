import xml.etree.ElementTree as ET
import os, shutil
import time
from distutils.dir_util import copy_tree
import pandas as pd 
import pickle

"""
Cleans up raw data. Removes duplicates as described in the paper, and creates a merged dataframe for analysis purposes.
Benchmarked execution time: 530s. (About 9 minutes.)
"""

root_folder = ''
def remove_typeN(type1_path, dirty_path, new_path):
    global root_folder
    
    type1_path= root_folder+type1_path
    dirty_path = root_folder+dirty_path
    new_path = root_folder+new_path

    type1f = open(type1_path)
    f = open(dirty_path)

    tree_1 = ET.parse(type1f)
    root = tree_1.getroot()
    
    dirty_tree = ET.parse(f)
    dirty_root = dirty_tree.getroot()
 
    # create a list of sources
    sources = []
    for elem in root[4:]:
        sources.extend(elem[:])
    
    sources = [(x.attrib['file'],x.attrib['startline'], x.attrib['endline']) for x in sources[:]]
     
    #iterate through every dirty element and check if it exists in the type1 thing, remove it
    for i, elem in enumerate(dirty_root[4:],4):
        head=0
        for j, source in enumerate(elem[:]):
                if (source.attrib['file'], source.attrib['startline'], source.attrib['endline']) in sources:
                        dirty_root[i].remove(dirty_root[i][head])
                else:
                    head+=1
    
    head=4 
    for i, elem in enumerate(dirty_root[4:], 4):
        if len(elem[:])==0:
            dirty_root.remove(dirty_root[head])
        else:
            head+=1
    
    dirty_tree.write(new_path)


def remove_all_duplicates():
    global root_folder
    type1_path = 'raw/withoutsource/type-1.xml'
    type2b_path = 'raw/withoutsource/type-2.xml'
    type2c_path = 'raw/withoutsource/type-2c.xml'
    type3_path = 'raw/withoutsource/type-3-1.xml'
    type3b_path = 'raw/withoutsource/type-3-2.xml'
    type3c_path = 'raw/withoutsource/type-3-2c.xml'

    os.makedirs('duplicates/final', exist_ok=True)
    parent_folder = 'duplicates/'
    #remote type1 from the rest
    print('---Removing type-1 from the rest.---')
    type2b_filtered_type1 = 'type2b_filtered_type1.xml'
    type2c_filtered_type1 = 'type2c_filtered_type1.xml'
    type3_filtered_type1 = 'type3_filtered_type1.xml'
    type3b_filtered_type1 = 'type3b_filtered_type1.xml'
    type3c_filtered_type1 = 'type3c_filtered_type1.xml'
    
    remove_typeN(type1_path, type2b_path, parent_folder+type2b_filtered_type1)
    remove_typeN(type1_path, type2c_path,  parent_folder+type2c_filtered_type1)
    remove_typeN(type1_path, type3_path,  parent_folder+type3_filtered_type1)
    remove_typeN(type1_path, type3b_path,  parent_folder+type3b_filtered_type1)
    remove_typeN(type1_path, type3c_path,  parent_folder+type3c_filtered_type1)

    root_folder = 'duplicates/' 
    print('---done---')
    
    #remove type2c from all type3
    print('---Removing type-2c from type-3.---')
    type3_filtered_type2c = 'type3_filtered_type2c.xml'
    type3b_filtered_type2c = 'type3b_filtered_type2c.xml'
    type3c_filtered_type2c = 'type3c_filtered_type2c.xml'

    remove_typeN(type2c_filtered_type1, type3_filtered_type1, type3_filtered_type2c)
    remove_typeN(type2c_filtered_type1, type3b_filtered_type1, type3b_filtered_type2c)
    remove_typeN(type2c_filtered_type1, type3c_filtered_type1, type3c_filtered_type2c)
    print('---done---')
    
    #remove type2b from all type3
    print('---Removing type-2b from type-3.---')
    type3_filtered_type2b = 'type3_filtered_type2b.xml'
    type3b_filtered_type2b = 'type3b_filtered_type2b.xml'
    type3c_filtered_type2b = 'type3c_filtered_type2b.xml'
    
    remove_typeN(type2b_filtered_type1, type3_filtered_type2c, type3_filtered_type2b)
    remove_typeN(type2b_filtered_type1, type3b_filtered_type2c, type3b_filtered_type2b)
    remove_typeN(type2b_filtered_type1, type3c_filtered_type2c, type3c_filtered_type2b)
    print('---done---')

    # remove type2c from type2b 
    print('---Removing type-2c from type-2b.---')
    type2b_filtered_type1_type2c = 'type2b_filtered_type1_type2c.xml'
    remove_typeN(type2c_filtered_type1, type2b_filtered_type1, type2b_filtered_type1_type2c)
    print('---done---')

    #remove type3 from type3b and type3c
    print('---Removing type-3 from type-3b and type-3c.---')
    type3b_filtered_type2b_type3 = 'type3b_filtered_type2b_type3.xml'
    remove_typeN(type3_filtered_type2b, type3b_filtered_type2b, type3b_filtered_type2b_type3)

    type3c_filtered_type2b_type3 = 'type3c_filtered_type2b_type3.xml'
    remove_typeN(type3_filtered_type2b, type3c_filtered_type2b, type3c_filtered_type2b_type3)
    print('---done---')
   
    #remove type3c from type3b
    print('---Removing type-3c from type-3b.---')
    type3b_filtered_type2b_type3_type3c = 'type3b_filtered_type2b_type3_type3c.xml'
    remove_typeN(type3c_filtered_type2b_type3, type3b_filtered_type2b_type3, type3b_filtered_type2b_type3_type3c)
    print('---done---')

    os.makedirs('duplicates/final', exist_ok=True)
    shutil.copy(type1_path, 'duplicates/final/type-1.xml')
    shutil.move('duplicates/'+type2c_filtered_type1, 'duplicates/final/type-2c.xml')
    shutil.move('duplicates/'+type2b_filtered_type1_type2c, 'duplicates/final/type-2.xml')
    shutil.move('duplicates/'+type3_filtered_type2b, 'duplicates/final/type-3-1.xml')
    shutil.move('duplicates/'+type3c_filtered_type2b_type3, 'duplicates/final/type-3-2c.xml')
    shutil.move('duplicates/'+type3b_filtered_type2b_type3_type3c, 'duplicates/final/type-3-2.xml')
    
    # TODO: enable this when we want to do the correlation calculation from scratch
    #created_merged_df()


def convert_file_with_subelem(clone_type, xml_file, df_file):
    global root_folder
    
    xml_file = root_folder+xml_file
    df_file = root_folder+'subelem_'+df_file
    tree = ET.parse(xml_file)

    root = tree.getroot()

    elements = []
    for i, elem in enumerate(root[4:]):
        for source in elem[:]:
            d={}
            d['classid'] = elem.attrib['classid']
            d['nclones'] = int(len(elem[:]))
            d['nlines'] = int(elem.attrib['nlines'].replace('"',''))
            d['similarity'] = elem.attrib['similarity']
            d['startline'] = source.attrib['startline']
            d['endline'] = source.attrib['endline']
            d['file'] = source.attrib['file']
            d['type'] = clone_type
            elements.append(d)
        
    df = pd.DataFrame(elements)
    pickle.dump(df, open(df_file, 'wb'))

def create_merged_df():
    global root_folder
    root_folder='duplicates/'
    
    type1 = 'final/type-1.xml'
    type1df = 'final/type-1.p'

    type2b = 'final/type-2.xml'
    type2bdf = 'final/type-2.p'

    type2c = 'final/type-2c.xml'
    type2cdf = 'final/type-2c.p'

    type3 = 'final/type-3-1.xml'
    type3df = 'final/type-3-1.p'
    
    type3b = 'final/type-3-2.xml'
    type3bdf = 'final/type-3-2.p'
    
    type3c= 'final/type-3-2c.xml'
    type3cdf = 'final/type-3-2c.p'

    os.makedirs('duplicates/subelem_final', exist_ok=True)

    file_pairs = [('type-1',type1, type1df), ('type-2', type2b, type2bdf), ('type-2c',type2c, type2cdf), ('type-3',type3, type3df), ('type-3-2', type3b, type3bdf), ('type-3-2c',type3c, type3cdf)]
    for ctype, x,y in file_pairs[:]:
        print('starting', ctype, x,y)
        convert_file_with_subelem(ctype, x,y)
        print("done")
    
    fs = os.listdir('duplicates/subelem_final')
    merged_df = pd.concat([pickle.load(open('duplicates/subelem_final/'+x, 'rb')) for x in fs])
    
    print("Persisting DataFrame into p file")
    pickle.dump(merged_df, open('duplicates/clones.p', 'wb'))
    
    shutil.rmtree('duplicates/subelem_final', ignore_errors=True)
    
if __name__=="__main__":
    start_time=time.time()
    os.chdir('../01_data/clonedata')
    print('------------------')
    print('Removing duplicates.')
    remove_all_duplicates()
    print('Creating clones.p.')
    create_merged_df()
    print('Cleanup finished. Elapsed time: {}.'.format(round(time.time()-start_time, 2)))
    print('------------------')