import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import submodels_module as mb
from sklearn import tree
import numpy as np


def get_node_depths(tree):
    """
    Get the node depths of the decision tree

    >>> d = DecisionTreeClassifier()
    >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    >>> get_node_depths(d.tree_)
    array([0, 1, 1, 2, 2])
    """
    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths) 
    return np.array(depths)





a=mb.seqandassay_to_yield_model([1,8,10],'forest',1)
a.load_model(0)

#sort1 = input 0
#sort8 = input 1
#sort10 = input 2
fig,ax=plt.subplots(1,3,figsize=[6,3],dpi=300)
sorts=["Prot K 37","GFP SHuffle",r'$\beta$'+"-lactamase SHuffle"]
for sort_no in [0,1,2]:

	blac_nodes_info=[]
	for j in range(a._model.model.n_estimators):
		a_tree=a._model.model.estimators_[j].tree_
		node_depth=get_node_depths(a_tree)

		for i in range(a_tree.node_count):
			if a_tree.feature[i]==sort_no:

				#tree left is always x<=treshold. and should be "lower" yield
				left_idx=a_tree.children_left[i]
				right_idx=a_tree.children_right[i]
				# if left value is lower, sign = True
				if a_tree.value[left_idx][0][0]<a_tree.value[right_idx][0][0]:
					sign=True
				else:
					sign=False

				blac_nodes_info.append([node_depth[i],a_tree.threshold[i],sign])


	blac_nodes_info=np.array(blac_nodes_info)


	data=[]
	for i in range(5):
		per_depth=blac_nodes_info[blac_nodes_info[:,0]==i][:,1]
		if len(per_depth)>0:
			data.append(blac_nodes_info[blac_nodes_info[:,0]==i][:,1])
		else:
			data.append(np.nan)
	all_cutoffs=np.concatenate(data[:])
	data.insert(0,all_cutoffs)
	violin_parts=ax[sort_no].violinplot(data,positions=[0,1,2,3,4,5],showextrema=False,widths=.9)
	for pc in violin_parts['bodies']:
	    pc.set_color('k')
	ax[sort_no].set_xticks([0,1,2,3,4,5])
	ax[sort_no].set_xticklabels(['All','0','1','2','3','4'])
	ax[sort_no].tick_params(axis='both', which='major', labelsize=6)
	ax[sort_no].set_ylim([0,1])
	ax[sort_no].set_xlabel('Depth of Node',fontsize=6)
	ax[sort_no].set_ylabel('Threshold of Node',fontsize=6)
	ax[sort_no].set_title(sorts[sort_no],fontsize=6)


fig.tight_layout()
fig.savefig('./sort_forest_cutoffs.png')
plt.close()
