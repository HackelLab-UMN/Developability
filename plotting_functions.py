import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def plot_yield_predictions(true_yield,predicted_yield,model_name,test_var_plot,test_err_plot,fig_name):
    from joblib import load


    boxcox=load('./make_datasets/Yield_boxcox_fit.joblib')
    true_yield=boxcox.inverse_transform(np.array(true_yield).reshape(-1,1))
    predicted_yield=boxcox.inverse_transform(np.array(predicted_yield).reshape(-1,1))
    true_yield[true_yield<=0.1]=0.1
    predicted_yield[predicted_yield<=0.1]=0.1

    fig, ax = plt.subplots(1,figsize=[3,3],dpi=300)
    fontsize=8
    ax.scatter(true_yield, predicted_yield,s=16,marker='.',color='black',alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.08,100])
    ax.set_ylim([0.08,100])
    ax.set_aspect('equal')
    ax.set_xlabel('True Yields (mg/L)',fontsize=fontsize)
    ax.set_ylabel('Predicted Yields (mg/L)',fontsize=fontsize)
    ax.set_title(model_name+' '+str(round(test_err_plot,3))+'\n Exp. Var '+str(round(test_var_plot,3)),fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(fig_name)
    plt.close()

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    plt.sca(last_axes)
    return cbar

def plot_assay_predictions(true,predicted,model_name,test_var_plot,test_err_plot,fig_name):

    fig, ax = plt.subplots(1,figsize=[3,3],dpi=300)
    fontsize=8
    _,_,_,img1=ax.hist2d(true, predicted, bins=(np.linspace(0,1,21), np.linspace(0,1,21)), cmap=plt.cm.jet,cmin=1)
    colorbar(img1)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect('equal')
    ax.set_xlabel('True Scores',fontsize=fontsize)
    ax.set_ylabel('Predicted Scores',fontsize=fontsize)
    ax.set_title(model_name+' '+str(round(test_err_plot,3))+'\n Exp. Var '+str(round(test_var_plot,3)),fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(fig_name)
    plt.close()