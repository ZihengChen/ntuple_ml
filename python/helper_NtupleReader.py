#!/usr/bin/env python

from root_pandas import read_root
from pylab import *
import pandas as pd
from tqdm import trange
import time

class NtupleReader():
    def __init__(self, inputFileName, datasetFormat="LAYERS"):
        
        self.inputFileName = inputFileName

        self.variableName = [
            'cluster2d_layer',
            'cluster2d_energy',
            'cluster2d_eta',
            'cluster2d_phi',
            'gen_energy',
            'gen_pdgid'
            ]
        
        self.datasetFormat = datasetFormat

    def makeDataset(self):
        events = pd.DataFrame(self.getEvents())
        return events
        
    def getEvents(self):
        # load root file into dataframe
        df = read_root(self.inputFileName, 'ana/hgc', columns=self.variableName )
        # loop over events
        numberOfEvents = len(df)
        for ievt in trange(numberOfEvents, unit=' events', ncols=100 ):
            event = df.loc[ievt]
            
            # get features
            feature = []
            
            for l in range(1,53,1):
                ## select good layer clusters
                slt = event.cluster2d_layer==l 
                slt &= event.cluster2d_eta>0
                slt &= event.cluster2d_energy>0
                if self.datasetFormat == "LAYERS":
                    # a list of 52 arrays, shape=(nCls,3). each array is cluster2d on a given layer.
                    layer_feature = [event['cluster2d_{}'.format(var)][slt] \
                                     for var in ['eta','phi','energy']]
                    layer_feature = np.array(layer_feature).T
                    layer_feature = layer_feature[layer_feature[:,2].argsort()]
                    feature.append(layer_feature)
                        
                elif self.datasetFormat == "LAYERS_SUMCLUSTERS":
                    # a list of 52 floats. each float is sum of energy of 2D clusters on a given layer.
                    layer_feature = [event['cluster2d_energy'][slt].sum()]
                    feature.append(layer_feature)
                
                elif self.datasetFormat == "CLUSTERS":
                    # a list of ncls array, shape=(4). each array is a 2d cluster.
                    layer_feature = [event['cluster2d_{}'.format(var)][slt] \
                                     for var in ['layer','eta','phi','energy']]
                    layer_feature = np.array(layer_feature).T
                    layer_feature = layer_feature[layer_feature[:,3].argsort()]
                    for cl in layer_feature:
                        feature.append(cl)
                    
                else:
                    print("datasetFormat is not defined. Choose from [CLUSTERS,LAYERS,LAYERS_LAYERPOOL]")

                
            # get label
            ## select good genpart
            gen_pid = np.abs(event.gen_pdgid)[0]
            gen_energy = event.gen_energy[0]

            if gen_pid ==11:
                label=0
            if gen_pid == 22:
                label=1
            if gen_pid == 211:
                label=2
            if gen_pid == 13:
                label=3
                
            # save feature and label
            eventFeatureLabel = {'feature':feature, 'label':label, 'gen_energy':gen_energy}

            yield eventFeatureLabel


if __name__ == '__main__':
    # configuration
    datasetFormat, partition = "CLUSTERS", 30000
    dirctory = "/Users/zihengchen/Documents/HGCal/TICL/data"

    # read data from root files
    df = []
    for pid in ['11','13','22','211']:
        inputFileName = dirctory + '/root/step4_NTUPLE_pid{}.root'.format(pid)
        print("processing " + inputFileName)
        time.sleep(1) # wait one second before start processing
        df.append( NtupleReader(inputFileName, datasetFormat ).makeDataset() )
    df = pd.concat(df,ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # partition dataset to train and test
    trn = df[0:partition].copy().reset_index(drop=True)
    tst = df[partition: ].copy().reset_index(drop=True)
    # save to hdf file
    trn.to_pickle( dirctory + "/pickle/dataset_{}_train.pkl".format(datasetFormat))
    tst.to_pickle( dirctory + "/pickle/dataset_{}_test.pkl".format(datasetFormat))