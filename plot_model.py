import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class model_plot():
    def colorbar(self,mappable):
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


    def __init__(self):
        self.fig, self.ax = plt.subplots(1,2,figsize=[6,3],dpi=300)

class x_to_yield_plot(model_plot):
    def add_axis(self):
        for i in range(2):
            self.ax[i].set_xlabel('Predicted Yield')
            self.ax[i].set_ylabel('True Yield')
            self.ax[i].set_xlim([-3,3])
            self.ax[i].set_ylim([-3,3])


    def __init__(self,model):
        super().__init__()
        self.ax[0].scatter(model.plotpairs_cv[1], model.plotpairs_cv[0],s=16,marker='.',color='black',alpha=0.5)
        self.ax[0].set_title('CV_MSE='+str(round(model.model_stats['cv_avg_loss'],3))+r'$\pm$'+str(round(model.model_stats['cv_std_loss'],3)))

        self.ax[1].scatter(model.plotpairs_test[1], model.plotpairs_test[0],s=16,marker='.',color='black',alpha=0.5)
        self.ax[1].set_title('Test_MSE='+str(round(model.model_stats['test_avg_loss'],3))+r'$\pm$'+str(round(model.model_stats['test_std_loss'],3)))

        self.add_axis()
        self.fig.tight_layout()

class x_to_assay_plot(model_plot):
    def add_axis(self):
        for i in range(2):
            self.ax[i].set_xlabel('Predicted Assay Score')
            self.ax[i].set_ylabel('True Assay Score')
            self.ax[i].set_xlim([0,1])
            self.ax[i].set_ylim([0,1])


    def __init__(self,model):
        super().__init__()
        _,_,_,img1=self.ax[0].hist2d(model.plotpairs_cv[1], model.plotpairs_cv[0], bins=(np.linspace(0,1,21), np.linspace(0,1,21)), cmap=plt.cm.jet,cmin=1)
        self.colorbar(img1)
        self.ax[0].set_title('CV_MSE='+str(round(model.model_stats['cv_avg_loss'],3))+r'$\pm$'+str(round(model.model_stats['cv_std_loss'],3)))
        _,_,_,img2=self.ax[1].hist2d(model.plotpairs_test[1], model.plotpairs_test[0], bins=(np.linspace(0,1,21), np.linspace(0,1,21)), cmap=plt.cm.jet,cmin=1)
        self.colorbar(img2)
        self.ax[1].set_title('Test_MSE='+str(round(model.model_stats['test_avg_loss'],3))+r'$\pm$'+str(round(model.model_stats['test_std_loss'],3)))

        self.add_axis()
        self.fig.tight_layout()