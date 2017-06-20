import numpy as np
import pandas as pd
import sys
from sklearn.utils import shuffle
#sys.path.insert(0, '/home/zeng/caffe_wyd/python')
#import caffe



def shuffle_filelist(filelist):

    data = pd.read_table(filelist, delim_whitespace=True)
    sf_data = shuffle(data)
    sf_data.to_csv(filelist+'_out', index=False)
    
    

if __name__ == '__main__':
    filelist = sys.argv[1]
    shuffle_filelist(filelist)
    
