from root_pandas import read_root
from pylab import *
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot

from TracksterReader import tracksterLabel

def truncatedCone(eta0,eta1,z0,z1):
    R,r = 200,400
    theta0 = 2*np.arctan(np.exp(-eta0))
    theta1 = 2*np.arctan(np.exp(-eta1))
        
    phi = np.linspace(0, 2 * np.pi, 120)
    rad = np.linspace(0, 2 * np.pi, 120)
    phi, rad = np.meshgrid(phi,rad)
    
    z   = ((z0+z1)/2 + r*np.sin(rad)).clip(z0,z1)
    rho = (R+r*np.cos(rad))
    
    for i in range(120):
        for j in range(120):
            rhomax = z[i,j]*np.tan(theta0)
            rhomin = z[i,j]*np.tan(theta1)
            if rho[i,j] > rhomax:
                rho[i,j] = rhomax
            if rho[i,j] < rhomin:
                rho[i,j] = rhomin
    rho = rho.clip(0,250)
                
    x = rho*np.cos(phi) 
    y = rho*np.sin(phi) 
    
    return x,y,z


class TracksterDisplay():
  def __init__(self, inputFileName="step4.root"):
    self.inputFileName = inputFileName

    self.variableName  = [ 
      'cluster2d_layer', 
      'cluster2d_energy',
      'cluster2d_x',
      'cluster2d_y',
      'cluster2d_z',
      'cluster2d_best_cpPdg',
      'gunparticle_eta', 'gunparticle_phi', 'gunparticle_id','gunparticle_energy',
      'trackster_clusters'
      ]
      
      
      
    self.df = read_root(self.inputFileName, 'ana/hgc',columns=self.variableName)

    self.pidToLabelMap = {11:0, -11:0, 22:1, 13:2, -13:2, 211:3, 311:4, -311:4, -1:-1}
    self.cmapdict = {-1:"black", 0:"blue",1:"red",2:"green",3:"purple",4:"orange",5:"black"}



  def figureHGCalGeometry(self):
    x,y,z = truncatedCone(1.48,3.0,320,352)
    geomEE = go.Surface(x=x, y=z, z=y, surfacecolor=0.5*np.ones_like(y),cmin=0,cmax=1,opacity=0.3,showscale=False)
    x,y,z = truncatedCone(1.48,3.0,357,410)
    geomFH = go.Surface(x=x, y=z, z=y, surfacecolor=0.5*np.ones_like(y),cmin=0,cmax=1,opacity=0.3,showscale=False)
    x,y,z = truncatedCone(1.4,3.0,415,500)
    geomBH = go.Surface(x=x, y=z, z=y, surfacecolor=0.5*np.ones_like(y),cmin=0,cmax=1,opacity=0.3,showscale=False)
    return [geomEE,geomFH,geomBH]

  def figureGentLines(self, eventIds):
    genLinesList = []

    for idx in eventIds:
      event = self.df.loc[idx]
      n = len(event.gunparticle_id)
      x = np.zeros(n*2)
      y = np.zeros(n*2)
      z = np.zeros(n*2)
    
      # labels = pidToLabelMap[event.gunparticle_id]
      # genLinesColor = cmapdict[labels]
    
      for i in range(n):
        eta, phi = event.gunparticle_eta[i],event.gunparticle_phi[i]
        theta = 2*np.arctan(np.exp(-eta))
        x[2*i]= 320*np.tan(theta)*np.cos(phi)
        y[2*i]= 320*np.tan(theta)*np.sin(phi)
        z[2*i]= 320

      genLines = go.Scatter3d(x=x, y=z, z=y, name="event{} GenParticle".format(idx), mode='lines', marker=dict(size=0,opacity=1,line=dict(width=1,color='black'))) 
      genLinesList.append(genLines)
    return genLinesList
    
  def figureClusters(self,eventIds):
      
      clustersList = []
      for idx in eventIds:
        event = self.df.loc[idx]
        
        e = event.cluster2d_energy
        x = event.cluster2d_x
        y = event.cluster2d_y
        z = event.cluster2d_z
          
        for tr in event.trackster_clusters:
          e = np.delete(e,tr)
          x = np.delete(x,tr)
          y = np.delete(y,tr)
          z = np.delete(z,tr)
          
        clusters = go.Scatter3d(x=x, y=z, z=y, name="event{} NoiseClusters".format(idx), mode='markers', marker=dict(size=5*e**0.5,color='black',line=dict(width=0))) 
        clustersList.append(clusters)
      return clustersList


  def figureTrackster(self,eventIds):
      
    trackstersList = []
    for idx in eventIds:
      event = self.df.loc[idx]
      for j,tr in enumerate(event.trackster_clusters):
        # make trackster
        e = event.cluster2d_energy[tr]
        x = event.cluster2d_x[tr]
        y = event.cluster2d_y[tr]
        z = event.cluster2d_z[tr]

        cpPdg = event.cluster2d_best_cpPdg
        pid = tracksterLabel(tr,cpPdg)
        label = self.pidToLabelMap[pid]

        if label == 2:
          markerSize = 20
        else:
          markerSize = 5
        trackster = go.Scatter3d(x=x, y=z, z=y, name="event{} Trackster{} pid{}".format(idx,j,pid), mode='markers', marker=dict(size=markerSize*e**0.5,color=self.cmapdict[label],line=dict(width=0))) 
        trackstersList.append(trackster)
    return trackstersList

  def plot(self,eventIds):

    geom     = self.figureHGCalGeometry()
    genparts = self.figureGentLines(eventIds)
    clusters  = self.figureClusters(eventIds)
    tracksters  = self.figureTrackster(eventIds)

    fig = go.Figure(data   = geom+genparts+clusters+tracksters,
                    layout = go.Layout(scene = dict(xaxis=dict(title='x (cm)'),
                                                    yaxis=dict(title='z (cm)'),
                                                    zaxis=dict(title='y (cm)')),
                                        margin = dict(l=0,r=0,b=0,t=0)
                                      )
                    )
    plot(fig, filename='test')