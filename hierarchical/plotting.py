# hierarchical/plotting.py

"""
plotting functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.patches as mpatches
# category_to_indices is used by show_evaluation
from .category import category_to_indices 


sns.set_theme(
    context="paper",
    style="white",
    palette="colorblind",
    font="DejaVu Sans",
    font_scale=1.75,
)

def cos_heatmap(mats, titles=None, figsize=(19, 8),
                labels=None, # These are the tick labels
                cmap=None, use_absvals=False, save_to=None,
                xticklabels_rotation=90, yticklabels_fontsize=8, xticklabels_fontsize=8):
    fig = plt.figure(figsize=figsize)
    
    num_mats = len(mats)
    left_margin = 0.25 if labels and num_mats > 0 and any(labels) else 0.10 # More space if y-labels
    right_margin = 0.85 # To leave space for colorbar
    gs_wspace = 0.20 if labels and num_mats > 1 and any(labels) else 0.10 # More space if y-labels on first impact others
    gs_bottom = 0.15 if labels and any(labels) and xticklabels_rotation != 0 else 0.05 
    gs_top = 0.90 if titles and any(titles) else 0.95

    gs = gridspec.GridSpec(1, num_mats, wspace=gs_wspace, 
                           left=left_margin, right=right_margin, 
                           bottom=gs_bottom, top=gs_top)

    vmin = 0.0 if use_absvals else -1.0 
    vmax = 1.0

    if cmap is None:
        cmap = "viridis" if use_absvals else "vlag" 

    ims = []
    for i in range(num_mats):
        ax = plt.subplot(gs[0, i])
        mat_to_plot = mats[i]
        if isinstance(mat_to_plot, torch.Tensor):
            mat_to_plot = mat_to_plot.cpu().numpy()
            
        current_aspect = 'auto' 
        if mat_to_plot.shape[0] == mat_to_plot.shape[1]:
            current_aspect = 'equal'
                                   
        im = ax.imshow(mat_to_plot, aspect=current_aspect, cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ims.append(im)

        if labels is not None and len(labels) > 0 and \
           len(labels) == mat_to_plot.shape[0] and \
           len(labels) == mat_to_plot.shape[1]:
            
            tick_positions = np.arange(len(labels))
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            
            if i == 0: 
                ax.set_yticklabels(labels, fontsize=yticklabels_fontsize)
            else:
                ax.set_yticklabels([])
            
            ax.set_xticklabels(labels, rotation=xticklabels_rotation, ha="right", fontsize=xticklabels_fontsize)
        else: 
            ax.set_xticks([])
            ax.set_yticks([])
            
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=14 if len(titles) <=2 else 12) 

    if ims: 
        cbar_ax_left = gs.right + 0.02 
        cbar_ax_bottom = gs.bottom 
        cbar_ax_width = 0.03
        cbar_ax_height = gs.top - gs.bottom
        
        if cbar_ax_left + cbar_ax_width > 0.99:
            cbar_ax_left = 0.99 - cbar_ax_width
            if gs.right > cbar_ax_left - 0.02 : 
                 gs.update(right = cbar_ax_left - 0.03)

        cbar_ax = fig.add_axes([cbar_ax_left, cbar_ax_bottom, cbar_ax_width, cbar_ax_height])
        cbar = fig.colorbar(ims[-1], cax=cbar_ax, orientation='vertical')
    
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
    # Return fig to allow explicit plt.show() or plt.close() in the notebook
    return fig


def proj_2d(dir1, dir2, unembed, vocab_list, ax,
              added_inds=None,
              normalize = True,
              orthogonal=False, k=10, fontsize=10,
              alpha=0.2, s=0.5,
              target_alpha = 0.9, target_s = 2,
              xlim = None,
              ylim = None,
              draw_arrows = False,
              arrow1_name = None,
              arrow2_name = None,
              right_topk = True,
              left_topk = True,
              top_topk = True,
              bottom_topk = True,
              xlabel="dir1",
              ylabel="dir2",
              title="2D projection plot"):
    # This is from your originally provided plotting.py, with minor robustness
    original_dir1 = dir1.clone() # Use .clone() to avoid modifying original tensors
    original_dir2 = dir2.clone()
    
    dir1_norm_val = torch.norm(dir1) + 1e-9 # Add epsilon for stability
    dir2_norm_val = torch.norm(dir2) + 1e-9

    if normalize:
        dir1 = dir1 / dir1_norm_val
        dir2 = dir2 / dir2_norm_val
        # Update norms after normalization for arrow calculation if orthogonal=False
        dir1_norm_val = 1.0 
        dir2_norm_val = 1.0

    if orthogonal:
        dir1 = dir1 / dir1_norm_val # Ensure dir1 is unit for projection
        dot_product = torch.dot(dir2, dir1)
        dir2 = dir2 - dot_product * dir1
        dir2 = dir2 / (torch.norm(dir2) + 1e-9) # Normalize the orthogonalized dir2
        
        # Arrows represent projection of original_dirs onto the new (potentially orthogonal) basis
        arrow1 = [(torch.dot(original_dir1, dir1)).cpu().item(), (torch.dot(original_dir1, dir2)).cpu().item()]
        arrow2 = [(torch.dot(original_dir2, dir1)).cpu().item(), (torch.dot(original_dir2, dir2)).cpu().item()]
    elif draw_arrows: # Define arrows even if not strictly orthogonal for plotting purposes
        # Simplified: arrows point along the (potentially just normalized) dir1 and dir2
        # Their lengths will represent their original magnitudes projected onto these chosen axes.
        arrow1 = [(torch.dot(original_dir1, dir1)).cpu().item(), 0] # Assumes dir2 is somewhat "y-axis"
        arrow2 = [0, (torch.dot(original_dir2, dir2)).cpu().item()] # Assumes dir1 is somewhat "x-axis"
        # A more general approach if not orthogonal might just scale dir1, dir2 by some factor
        # arrow1 = (dir1 * (xlim[1]*0.3 if xlim else 5)).cpu().numpy() # Example scaling
        # arrow2 = (dir2 * (ylim[1]*0.3 if ylim else 5)).cpu().numpy()


    proj1 = unembed @ dir1
    proj2 = unembed @ dir2
    
    ax.scatter(proj1.cpu().numpy(), proj2.cpu().numpy(),
               alpha=alpha, color="gray", s=s)
    
    def _add_labels_for_largest(proj_coords, largest_bool):
        if proj_coords.ndim > 1: proj_coords = proj_coords.squeeze()
        if proj_coords.nelement() == 0: return
        k_eff = min(k, len(proj_coords))
        indices = torch.topk(proj_coords, k=k_eff, largest=largest_bool).indices
        for idx in indices:
            if idx < len(vocab_list) and vocab_list[idx] and "$" not in vocab_list[idx]:
                ax.text(proj1[idx].item(), proj2[idx].item(), vocab_list[idx], fontsize=fontsize)
    
    if right_topk: _add_labels_for_largest(proj1, largest=True)
    if left_topk: _add_labels_for_largest(proj1, largest=False)
    if top_topk: _add_labels_for_largest(proj2, largest=True)
    if bottom_topk: _add_labels_for_largest(proj2, largest=False)

    if added_inds:
        colors = iter(["b", "r", "green", "orange", "skyblue", "pink", "yellowgreen", "brown", "cyan", "olive", "purple", "lime"])
        legend_handles = []
        for label, indices_list in added_inds.items():
            if not indices_list: continue
            color = next(colors)
            valid_indices = [idx for idx in indices_list if idx < proj1.shape[0]]
            if not valid_indices: continue
            ax.scatter(proj1[valid_indices].cpu().numpy(), proj2[valid_indices].cpu().numpy(),
                       alpha=target_alpha, color=color, s=target_s)
            legend_handles.append(mpatches.Patch(color=color, label=label))
        if legend_handles: ax.legend(handles=legend_handles, loc='lower left')

    min_proj1_val = proj1.min().item() if proj1.nelement() > 0 else -1.0
    max_proj1_val = proj1.max().item() if proj1.nelement() > 0 else 1.0
    min_proj2_val = proj2.min().item() if proj2.nelement() > 0 else -1.0
    max_proj2_val = proj2.max().item() if proj2.nelement() > 0 else 1.0

    if xlim is not None: ax.set_xlim(xlim); ax.hlines(0, xlim[0], xlim[1], colors="black", alpha=0.3, linestyles="dashed")
    else: ax.hlines(0, min_proj1_val, max_proj1_val, colors="black", alpha=0.3, linestyles="dashed")
    if ylim is not None: ax.set_ylim(ylim); ax.vlines(0, ylim[0], ylim[1], colors="black", alpha=0.3, linestyles="dashed")
    else: ax.vlines(0, min_proj2_val, max_proj2_val, colors="black", alpha=0.3, linestyles="dashed")
        
    if draw_arrows and 'arrow1' in locals() and 'arrow2' in locals(): # Ensure arrows are defined
        ax.arrow(0, 0, arrow1[0], arrow1[1], head_width=0.5, head_length=0.5, width=0.1, fc='blue', ec='blue', linestyle='dashed', alpha=0.6, length_includes_head=True)
        if arrow1_name: ax.text(arrow1[0]*0.6, arrow1[1]*0.6 - abs(max_proj2_val-min_proj2_val)*0.05, arrow1_name, fontsize=fontsize, bbox=dict(facecolor='blue', alpha=0.2))
        ax.arrow(0, 0, arrow2[0], arrow2[1], head_width=0.5, head_length=0.5, width=0.1, fc='red', ec='red', linestyle='dashed', alpha=0.6, length_includes_head=True)
        if arrow2_name: ax.text(arrow2[0]*0.6 - abs(max_proj1_val-min_proj1_val)*0.05, arrow2[1]*0.6 + abs(max_proj2_val-min_proj2_val)*0.05, arrow2_name, fontsize=fontsize, bbox=dict(facecolor='red', alpha=0.2))
    
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    # No explicit fig return needed if ax is passed in and modified


def proj_2d_single_diff(higher, subcat1, subcat2, unembed, vocab_list, ax, **kwargs):
    # This is your full implementation from the provided file
    dir1 = higher
    dir2 = subcat2 - subcat1

    original_higher = higher.clone()
    original_subcat1 = subcat1.clone()
    original_subcat2 = subcat2.clone()
    original_dir1 = dir1.clone()
    original_dir2 = dir2.clone()
    
    normalize = kwargs.get('normalize', True)
    orthogonal = kwargs.get('orthogonal', False) # Default to False if not specified

    if normalize:
        dir1 = dir1 / (torch.norm(dir1) + 1e-9)
        dir2 = dir2 / (torch.norm(dir2) + 1e-9)
    if orthogonal:
        dir1_norm_basis = dir1 / (torch.norm(dir1) + 1e-9) # dir1 for basis must be unit
        dir2 = dir2 - (torch.dot(dir2, dir1_norm_basis)) * dir1_norm_basis
        dir2 = dir2 / (torch.norm(dir2) + 1e-9)
        dir1 = dir1_norm_basis # Ensure dir1 is the unit vector used for ortho

        arrow1 = [(torch.dot(original_dir1, dir1)).cpu().item(), (torch.dot(original_dir1, dir2)).cpu().item()]
        arrow2 = [(torch.dot(original_dir2, dir1)).cpu().item(), (torch.dot(original_dir2, dir2)).cpu().item()]
    elif kwargs.get('draw_arrows', False): # Define arrows for plotting if not orthogonal
        arrow1 = [(torch.dot(original_dir1, dir1)).cpu().item(), 0] 
        arrow2 = [(torch.dot(original_dir2, dir1)).cpu().item(), (torch.dot(original_dir2, dir2)).cpu().item()]


    # Call the main proj_2d logic, but override arrow drawing if custom logic is needed here
    # For now, let proj_2d handle it based on computed dir1, dir2 and passed arrow kwargs
    # The arrow positioning might be tricky; your original code had specific positioning for this func.
    
    # Re-pass all kwargs to the generic proj_2d
    # The arrow start point for dir2 is different here.
    proj_2d(dir1, dir2, unembed, vocab_list, ax, **kwargs) # This will draw arrows from origin

    # Custom arrow drawing for proj_2d_single_diff if `draw_arrows` is True
    # The second arrow (dir2 = subcat2 - subcat1) should conceptually start at subcat1's projection.
    if kwargs.get('draw_arrows', False) and 'arrow1' in locals() and 'arrow2' in locals(): # Check if arrows were computed
        # Clear previous arrows if proj_2d drew them from origin incorrectly for this context
        # This is tricky; ideally proj_2d wouldn't draw arrows if it's a helper.
        # For now, we might overplot or need to make proj_2d more flexible.
        
        # Arrow 1 (higher concept)
        ax.arrow(0, 0, arrow1[0], arrow1[1], head_width=0.5, head_length=0.5,
                 width=0.1, fc='blue', ec='blue',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if kwargs.get('arrow1_name'):
            ax.text(arrow1[0]/2, arrow1[1]/2 - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05 , kwargs.get('arrow1_name'), 
                    fontsize=kwargs.get('fontsize',10), bbox=dict(facecolor='blue', alpha=0.2))

        # Arrow 2 (difference vector: subcat2 - subcat1) should start at subcat1's projection
        # Project original_subcat1 onto the new basis (dir1, dir2)
        subcat1_proj_x = (torch.dot(original_subcat1, dir1)).cpu().item()
        subcat1_proj_y = (torch.dot(original_subcat1, dir2)).cpu().item()

        # Arrow2 components are projections of (original_subcat2 - original_subcat1) onto new basis
        # arrow2[0] is (original_dir2 @ dir1), arrow2[1] is (original_dir2 @ dir2)
        ax.arrow(subcat1_proj_x, subcat1_proj_y, 
                 arrow2[0], arrow2[1], # These are components of (subcat2-subcat1) in new basis
                 head_width=0.5, head_length=0.5,
                 width=0.1,  fc='red', ec='red',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if kwargs.get('arrow2_name'):
            # Adjust text position relative to the arrow's midpoint
            mid_x = subcat1_proj_x + arrow2[0]/2
            mid_y = subcat1_proj_y + arrow2[1]/2
            ax.text(mid_x - (ax.get_xlim()[1]-ax.get_xlim()[0])*0.1 , mid_y + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 
                    kwargs.get('arrow2_name'), fontsize=kwargs.get('fontsize',10),
                    bbox=dict(facecolor='red', alpha=0.2))
    # No explicit fig return needed if ax is passed in and modified


def proj_2d_double_diff(higher1, higher2, subcat1, subcat2, unembed, vocab_list, ax, **kwargs):
    # This is your full implementation from the provided file
    dir1 = higher2 - higher1
    dir2 = subcat2 - subcat1

    original_higher1 = higher1.clone()
    original_higher2 = higher2.clone()
    original_subcat1 = subcat1.clone()
    original_subcat2 = subcat2.clone()
    original_dir1 = dir1.clone()
    original_dir2 = dir2.clone()
    
    normalize = kwargs.get('normalize', True)
    orthogonal = kwargs.get('orthogonal', False)

    if normalize:
        dir1 = dir1 / (torch.norm(dir1) + 1e-9)
        dir2 = dir2 / (torch.norm(dir2) + 1e-9)
    if orthogonal:
        dir1_norm_basis = dir1 / (torch.norm(dir1) + 1e-9)
        dir2 = dir2 - (torch.dot(dir2, dir1_norm_basis)) * dir1_norm_basis
        dir2 = dir2 / (torch.norm(dir2) + 1e-9)
        dir1 = dir1_norm_basis

        arrow1 = [(torch.dot(original_dir1, dir1)).cpu().item(), (torch.dot(original_dir1, dir2)).cpu().item()]
        arrow2 = [(torch.dot(original_dir2, dir1)).cpu().item(), (torch.dot(original_dir2, dir2)).cpu().item()]
    elif kwargs.get('draw_arrows', False):
        arrow1 = [(torch.dot(original_dir1, dir1)).cpu().item(), 0] 
        arrow2 = [0, (torch.dot(original_dir2, dir2)).cpu().item()]


    # Call generic proj_2d for scatter and basic labels
    proj_2d(dir1, dir2, unembed, vocab_list, ax, **kwargs) # This will draw arrows from origin

    # Custom arrow drawing for proj_2d_double_diff
    if kwargs.get('draw_arrows', False) and 'arrow1' in locals() and 'arrow2' in locals():
        # Arrow 1 (higher2 - higher1) starts at projection of higher1
        higher1_proj_x = (torch.dot(original_higher1, dir1)).cpu().item()
        higher1_proj_y = (torch.dot(original_higher1, dir2)).cpu().item()
        
        ax.arrow(higher1_proj_x, higher1_proj_y, 
                 arrow1[0], arrow1[1], # Components of (higher2-higher1) in new basis
                 head_width=0.5, head_length=0.5, width=0.1, fc='blue', ec='blue',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if kwargs.get('arrow1_name'):
            mid_x1 = higher1_proj_x + arrow1[0]*0.2
            mid_y1 = higher1_proj_y + arrow1[1]*0.2 - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05
            ax.text(mid_x1, mid_y1, kwargs.get('arrow1_name'), fontsize=kwargs.get('fontsize',10),
                    bbox=dict(facecolor='blue', alpha=0.2))

        # Arrow 2 (subcat2 - subcat1) starts at projection of subcat1
        subcat1_proj_x = (torch.dot(original_subcat1, dir1)).cpu().item()
        subcat1_proj_y = (torch.dot(original_subcat1, dir2)).cpu().item()

        ax.arrow(subcat1_proj_x, subcat1_proj_y,
                 arrow2[0], arrow2[1], # Components of (subcat2-subcat1) in new basis
                 head_width=0.5, head_length=0.5, width=0.1,  fc='red', ec='red',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if kwargs.get('arrow2_name'):
            mid_x2 = subcat1_proj_x + arrow2[0]/2 + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
            mid_y2 = subcat1_proj_y + arrow2[1]/2
            ax.text(mid_x2, mid_y2, kwargs.get('arrow2_name'), fontsize=kwargs.get('fontsize',10),
                    bbox=dict(facecolor='red', alpha=0.2))
    # No explicit fig return needed if ax is passed in


def show_evaluation(vec_reps_split, sorted_keys, vocab_dict, version='lda', save_to=None, 
                    data_type_name="WordNet Hierarchy"):
    
    ind, original, shuffled = ({'train': [], 'test': [], 'random': []} for _ in range(3))
    
    g = vec_reps_split['g'].cpu() 
    shuffled_g = vec_reps_split['shuffled_g'].cpu()

    num_samples = 200000
    if g.shape[0] == 0: 
        print("Warning (show_evaluation): 'g' matrix is empty."); return plt.figure(figsize=(18,5)) # Return empty fig
    if g.shape[0] < num_samples: num_samples = g.shape[0]
            
    torch.random.manual_seed(100)
    all_indices = torch.randperm(g.shape[0])[:num_samples] 
    ind['random'] = all_indices 

    active_plot_keys = [] 

    for node in sorted_keys:
        if not (node in vec_reps_split['original']['split'] and \
                node in vec_reps_split['shuffled']['split'] and \
                node in vec_reps_split['train_lemmas'] and \
                node in vec_reps_split['test_lemmas'] and \
                version in vec_reps_split['original']['split'][node] and \
                vec_reps_split['original']['split'][node][version].nelement() > 0 and \
                version in vec_reps_split['shuffled']['split'][node] and \
                vec_reps_split['shuffled']['split'][node][version].nelement() > 0
               ):
            continue 
        active_plot_keys.append(node)
        original_dir = vec_reps_split['original']['split'][node][version].cpu()
        original_dir_norm_sq = torch.norm(original_dir)**2
        if original_dir_norm_sq < 1e-9: original_dir_norm_sq = 1e-9 
        shuffled_dir = vec_reps_split['shuffled']['split'][node][version].cpu()
        shuffled_dir_norm_sq = torch.norm(shuffled_dir)**2
        if shuffled_dir_norm_sq < 1e-9: shuffled_dir_norm_sq = 1e-9

        ind['train'] = category_to_indices(vec_reps_split['train_lemmas'][node], vocab_dict)
        ind['test'] = category_to_indices(vec_reps_split['test_lemmas'][node], vocab_dict)

        for key in ['train', 'test', 'random']: 
            indices_to_use = ind[key]
            if not isinstance(indices_to_use, list): indices_to_use = indices_to_use.tolist()
            valid_indices = [i for i in indices_to_use if i < g.shape[0]]
            if not valid_indices:
                original[key].append(np.array([np.nan])); shuffled[key].append(np.array([np.nan]))
                continue
            original_projs = (g[valid_indices] @ (original_dir / original_dir_norm_sq))
            shuffled_projs = (shuffled_g[valid_indices] @ (shuffled_dir / shuffled_dir_norm_sq))
            original[key].append(original_projs.numpy()); shuffled[key].append(shuffled_projs.numpy())
    
    if not active_plot_keys:
        print("Warning (show_evaluation): No data points to plot. Returning empty figure.")
        fig, _ = plt.subplots(1, 2, figsize=(18,5)); return fig

    inds_plot = range(len(active_plot_keys))
    colors = {'train': 'green', 'test': 'blue', 'random': 'orange'}
    fig, ax_subplots = plt.subplots(1, 2, figsize=(18, 5))
    
    plotted_on_ax0 = False
    for key in ['train', 'test', 'random']:
        if len(original.get(key, [])) == len(active_plot_keys):
            proj_mean = [np.nanmean(proj) if proj.size > 0 else np.nan for proj in original[key]]
            proj_std = [np.nanstd(proj) if proj.size > 0 else np.nan for proj in original[key]]
            valid_plot_points = [(i, m, s) for i, (m,s) in enumerate(zip(proj_mean, proj_std)) if not (np.isnan(m) or np.isnan(s))]
            if valid_plot_points:
                plot_inds, plot_means, plot_stds = zip(*valid_plot_points)
                ax_subplots[0].plot(plot_inds, plot_means, alpha=0.8, color=colors[key], linewidth=1, label=key)
                ax_subplots[0].errorbar(plot_inds, plot_means, yerr=plot_stds, color=colors[key], capsize=3, ecolor=colors[key], alpha=0.1, fmt='none')
                plotted_on_ax0 = True
    ax_subplots[0].set_title('Original Unembeddings'); ax_subplots[0].set_ylim(-1, 2)
    if not plotted_on_ax0: ax_subplots[0].text(0.5, 0.5, "No data for original plot", ha='center', va='center', transform=ax_subplots[0].transAxes)

    plotted_on_ax1 = False
    for key in ['train', 'test', 'random']:
        if len(shuffled.get(key, [])) == len(active_plot_keys):
            proj_mean = [np.nanmean(proj) if proj.size > 0 else np.nan for proj in shuffled[key]]
            proj_std = [np.nanstd(proj) if proj.size > 0 else np.nan for proj in shuffled[key]]
            valid_plot_points = [(i, m, s) for i, (m,s) in enumerate(zip(proj_mean, proj_std)) if not (np.isnan(m) or np.isnan(s))]
            if valid_plot_points:
                plot_inds, plot_means, plot_stds = zip(*valid_plot_points)
                ax_subplots[1].plot(plot_inds, plot_means, alpha=0.8, color=colors[key], linewidth=1, label=key)
                ax_subplots[1].errorbar(plot_inds, plot_means, yerr=plot_stds, color=colors[key], capsize=3, ecolor=colors[key], alpha=0.1, fmt='none')
                plotted_on_ax1 = True
    ax_subplots[1].set_title('Shuffled Unembeddings'); ax_subplots[1].set_ylim(-1, 2)
    if not plotted_on_ax1: ax_subplots[1].text(0.5, 0.5, "No data for shuffled plot", ha='center', va='center', transform=ax_subplots[1].transAxes)

    if plotted_on_ax0 or plotted_on_ax1:
        handles, labels_legend = ax_subplots[0].get_legend_handles_labels()
        if not handles: handles, labels_legend = ax_subplots[1].get_legend_handles_labels()
        if handles: fig.legend(handles, labels_legend, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3)
    
    fig.supxlabel(rf'Binary Features in {data_type_name}', x=0.5, y=0.08 if plotted_on_ax0 or plotted_on_ax1 else 0.01, fontsize=17)
    plt.tight_layout(rect=[0, 0.05 if plotted_on_ax0 or plotted_on_ax1 else 0, 1, 0.95])
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    return fig
