from pylab import *

def plotEvent_EtaPhi(events,ievt):
    evt = events.loc[ievt]
    feature = evt.feature
    label = evt.label
    evtcolor = 'C{:1.0f}'.format(label)
    
    # make plot
    fig, axes = plt.subplots(2, 1, sharex=True, 
                             gridspec_kw={'height_ratios':[1,1]},
                             figsize=(8,4),facecolor='w')

    fig.subplots_adjust(hspace=0.1)


    for i in range(len(feature)):
        clusters = feature[i]
        ncls = len(clusters)
        
        ilayer = i+1
        
        if ilayer<=28:
            linew = 0.5
        elif ilayer <= 42:
            linew = 2
        else:
            linew = 5
            
        axes[0].axvline(ilayer,color='k',linestyle='-',lw=linew,alpha=0.2)
        axes[1].axvline(ilayer,color='k',linestyle='-',lw=linew,alpha=0.2)
        
        if ncls>0:
            eta    = clusters[:,0]
            phi    = clusters[:,1]
            energy = clusters[:,2]
            layer  = np.ones(ncls)*ilayer

            axes[0].scatter(layer,eta,color=evtcolor,s=energy*40,alpha=0.8)
            axes[1].scatter(layer,phi,color=evtcolor,s=energy*40,alpha=0.8)

    axes[0].set_ylim(1.3,3.0)
    axes[1].set_ylim(-3.3,3.3)
    axes[0].set_ylabel(r'$\eta$',fontsize=12)
    axes[1].set_ylabel(r'$\phi$',fontsize=12)
    axes[1].set_xlabel('layer',fontsize=12)
    axes[1].set_xlim(-4,57)
    axes[0].set_title('Label={}, Energy={:3.0f} GeV'.format(label,evt.gen_energy))