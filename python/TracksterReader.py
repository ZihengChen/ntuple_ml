#!/usr/bin/env python

from root_pandas import read_root
from numba import jit
from pylab import *
import pandas as pd

@jit(nopython=True)
def tracksterLabel(trcl, cpPdg):
    labelcount = {}
    label,maxcount = -1, 0
    for idx in trcl:
        l = cpPdg[idx]
        if l != -1:
            if l in labelcount:
                labelcount[l] += 1
            else:
                labelcount[l] = 1

    for key in labelcount:
        if labelcount[key] > maxcount:
            label = key
            maxcount = labelcount[key]
    return label

def tracksterImage(trcl, layer, energy, eta, phi):
    nlayer, nclu = 50, 10

    imagedict = {}        
    for l in range(nlayer):
        imagedict[l] = []

    for idx in trcl:
        clusterFeature = (eta[idx],phi[idx],energy[idx])
        imagedict[layer[idx]-1].append(clusterFeature)

    image = []
    for l in range(nlayer):
        nclustersOnLayer = len(imagedict[l])
        if nclustersOnLayer>0:
            imagedictl = imagedict[l]
            imagedictl.sort(key=lambda x: x[2],reverse=True)
            if nclustersOnLayer >= nclu:
                imagedictl = imagedictl[0:nclu]
            else:
                imagedictl = imagedict[l] + [(0,0,0)]*(nclu - nclustersOnLayer)
        else:
            imagedictl = [(0,0,0)]*nclu
        image.append(imagedictl)   
        
    image = np.array(image)
    return image


class TracksterReader():
    def __init__(self, inputFileName="step4.root"):
        
        self.inputFileName = inputFileName
        self.variableName  = ['cluster2d_layer', 
                              'cluster2d_energy',
                              'cluster2d_eta',
                              'cluster2d_phi',
                              'cluster2d_cpId',
                              'cluster2d_best_cpPdg',
                              'cluster2d_best_cpEnergy',
                              'trackster_clusters']
                

    def run(self):
        df = read_root(self.inputFileName, 'ana/hgc', columns=self.variableName )
        self.df = df
        dataset = self.processTracksters(df)
        dataset = pd.DataFrame(dataset)
        
        dataset['label'] = dataset['pid'].map({-11:0, 11:0, 22:1, 13:2, -13:2, -211:3, 211:3, 311:4, -1:5})
        return dataset
            
    def processTracksters(self, df):
        
        for ievent in range(len(df)):
            cpPdg = df.cluster2d_best_cpPdg[ievent]
            layer = df.cluster2d_layer[ievent]
            energy= df.cluster2d_energy[ievent]
            eta   = df.cluster2d_eta[ievent]
            phi   = df.cluster2d_phi[ievent]

            for tr in df.trackster_clusters[ievent]:
                trackster = {}
                            
                trackster["feature"] = tracksterImage(tr,layer,energy,eta,phi)
                trackster["pid"] = tracksterLabel(tr,cpPdg) 
                yield trackster


if __name__ == '__main__':
  rd = TracksterReader("../data/root/step4.root")
  dataset = rd.run()
  dataset.to_pickle("../data/pickle/step4.pkl")
