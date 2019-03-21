from root_pandas import read_root
from pylab import *
import pandas as pd
from tqdm import trange

class NtupleReader():
    def __init__(self, inputFileName, n=-1, tqdmLabel=None, layerPool=False):
        
        self.inputFileName = inputFileName

        self.n = n
        self.layerPool = layerPool
        if tqdmLabel is None:
            self.tqdmLabel = 'Making Dataset'
        else:
            self.tqdmLabel = tqdmLabel

        self.variableName = [
            'cluster2d_layer',
            'cluster2d_energy',
            'cluster2d_eta',
            'cluster2d_phi',
            'gen_energy',
            'gen_pdgid'
            ]

    def makeDataset(self,outputFileName=None):
        self.events  = pd.DataFrame(self.getEvents())
        if not outputFileName is None:
            self.events.to_pickle(outputFileName)
        

    def getEvents(self):

        df = read_root(self.inputFileName, 'ana/hgc', columns=self.variableName )

        if self.n <= 0:
            numberOfEvents = len(df)
        else:
            numberOfEvents = self.n

        for ievt in trange(numberOfEvents, desc=self.tqdmLabel, unit=' events', ncols=100 ):
            event = df.loc[ievt]

            # get feature
            feature = []
            for l in range(1,53,1):
                ## select good layer clusters
                slt = event.cluster2d_layer==l 
                slt &= event.cluster2d_eta>0
                slt &= event.cluster2d_energy>0
                if self.layerPool:
                    layer_feature = event['cluster2d_energy'][slt].sum()
                    feature.append(layer_feature)
                else:
                    layer_feature = [event['cluster2d_{}'.format(var)][slt] \
                                     for var in ['eta','phi','energy']]
                    layer_feature = np.array(layer_feature).T
                    layer_feature = layer_feature[layer_feature[:,2].argsort()]
                    feature.append(layer_feature)
                    
            if self.layerPool:
                feature = np.array(feature)
                normalizeFactor = feature.sum()
                if normalizeFactor > 0:
                    feature /= normalizeFactor
                
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



                  


        

