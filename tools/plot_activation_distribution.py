"""Activation-distribution analysis used for Figures 2 and 3 of the paper.

Hooks every quantized layer of a full-precision SR network, records the
activation histogram of each calibration sample, and plots the dense region,
the outlier region and the resulting breakpoint.

Run it from inside ``src/`` with the same arguments as ``main.py``.
"""

import torch
import data
import model
import utility
from option import args
from trainer import Trainer

import time
import datetime

import numpy as np
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
import seaborn as sns
from sympy import symbols
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

checkpoint = utility.checkpoint(args)
def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

def get_hadK(n, transpose=False):
    hadK, K = None, None
    assert is_pow2(n)
    K = 1
    return hadK, K

def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()

def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float32)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)
def plot_activations(args, name, abs_act: torch.Tensor, filename):
    # set up the figure and axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # plot activation
    abs_act = abs_act.squeeze()
    abs_act = abs_act.view(abs_act.size(0), -1)
    n_tokens, hidden_dim = abs_act.shape
    x = np.arange(1, hidden_dim + 1)
    y = np.arange(1, n_tokens + 1)
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_surface(X, Y, abs_act.detach().cpu().numpy(), cmap='coolwarm',
                          rstride=1, cstride=1, linewidth=0.5,
                          antialiased=True, zorder=1)
    # fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # plot separation line between prompt & query

    # ax1.set_title(f'Layer {name.split(".")[-1]}')
    ax1.set_xlabel('HW')
    ax1.set_ylabel('Channle')
    # ax1.set_zlabel('Absolute Value')

    jpg_path = os.path.join("../"+str(filename), f'{name}.jpg')
    plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()

def plot_activations_duquant(args, name, abs_act, filename):
    abs_act = abs_act.squeeze()
    abs_act = abs_act.view(abs_act.size(0), -1)
    fig = plt.figure()
    abs_act = abs_act.detach().cpu()
    max_activation_after = torch.max(abs_act, dim=1).values.numpy()
    min_activation_after = torch.min(abs_act, dim=1).values.numpy()
    plt.figure(figsize=(8, 6))

    # Plot Activation Change
    #plt.plot(max_activation_before, label="Max (before smooth)", color="orange")
    #plt.plot(max_activation_after, label="Max (after hadamard)", color="blue")
    channels = np.arange(abs_act.shape[0])
    plt.fill_between(channels, min_activation_after, max_activation_after,
                 color="blue", alpha=0.5, label="Before Hadamard (min-max range)")
    plt.title("Activation Change with Hadamard")
    plt.xlabel("Channel ID")
    plt.ylabel("Activation Value")
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    jpg_path = os.path.join("../"+str(filename), f'{name}_dequant.jpg')
    plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()

def hadamard_matrix(size):
    if size & (size -1) != 0:
        raise ValueError("Hadamard matrix size must be power of 2")
    return torch.tensor(scipy.linalg.hadamard(size), dtype = torch.float32)/(size ** 0.5)


def set_seed(seed):
    # Python standard-library RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch CPU RNG
    torch.manual_seed(seed)

    # CUDA RNG
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # all GPUs

    # make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    _loader = data.Data(args)
    _model = model.Model(args, checkpoint)
    _model.eval()
    test_images = _loader.loader_train
    feature_maps = {}

    def hook_fn(module, input, output, layer_name):
        feature_maps[layer_name] = input

    # Register a forward hook on every layer of the model.
    hooks = []
    for name, layer in _model.named_modules():
        if "conv" in name:
            hooks.append(layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name)))
    lr, _, idx_scale = next(iter(test_images))
    sr_t, _, _, _ = _model(lr.to("cuda"), idx_scale)
    for hook in hooks:
        hook.remove()
    ratation = random_hadamard_matrix(64, sr_t.device)
    filename = "hist2"
    high_density_ranges = []
    for name, act in tqdm(feature_maps.items()):
        act1 = act[0][0]
        # Per-region colours
        #colors = ['#3E5D70', '#3E5D70', '#3E5D70', 'c', 'm']#['#3E5D70', '#E57259', '#EEA672', 'c', 'm']
        # colors = ['#FFB6C1', '#FFD700', '#ADFF2F', '#87CEFA', '#FF69B4']  # LightPink, Gold, GreenYellow, LightSkyBlue, HotPink
        # Draw all histograms in the same figure
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        #fig.suptitle('Distributions with KDE Contour Lines (Ignoring 0 Range)', fontsize=16)
        """
        for i in range(3):

            counts, bins, patches = axes[i].hist(act1[i].detach().cpu().numpy().flatten(), bins=50, density=True, alpha=1, color=colors[i])
            #high_density_bins = bins[counts > 0.005]
            density_threshold = counts.mean() * 0.5
            high_density_bins = bins[:-1][counts > density_threshold]
            if len(high_density_bins) > 0:
                high_density_min, high_density_max = high_density_bins.min(), high_density_bins.max()
                high_density_ranges.append((high_density_min, high_density_max))
            axes[i].set_title(f'Sample {i+1} (Range: [{int(act1[i].detach().cpu().numpy().flatten().min())}, {int(act1[i].detach().cpu().numpy().flatten().max())}], Skew: {round(skew(act1[i].detach().cpu().numpy().flatten()), 2)})', fontsize=10)
            axes[i].set_ylabel('Density')
        """

        global_min = min(act1[i].detach().cpu().numpy().min() for i in range(3))
        global_max = max(act1[i].detach().cpu().numpy().max() for i in range(3))

        # Range of each region
        neg_quant_min, neg_quant_max = global_min, -50  # negative outlier region
        mid_quant_min, mid_quant_max = -50, 50          # dense region
        pos_quant_min, pos_quant_max = 50, global_max   # positive outlier region

        # Number of quantization points per region
        num_quant_points_neg = 4    # negative outlier region
        num_quant_points_mid = 8   # dense region
        num_quant_points_pos = 4    # positive outlier region

        # Generate the quantization points of the three regions
        neg_quant_points = np.linspace(neg_quant_min, neg_quant_max, num_quant_points_neg)
        mid_quant_points = np.linspace(mid_quant_min, mid_quant_max, num_quant_points_mid)
        pos_quant_points = np.linspace(pos_quant_min, pos_quant_max, num_quant_points_pos)

        # Plot the histogram together with the quantization points
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

        high_density_color = '#E57259'
        low_density_color_negative = '#EEA672'
        low_density_color_positive = '#87CEFA'

        for i in range(3):
            # Flatten the activations for each sample
            data1 = act1[i].detach().cpu().numpy().flatten()
            counts, bins, patches = axes[i].hist(data1, bins=50, density=True, alpha=1)

            # Density threshold
            density_threshold = counts.mean() * 0.5

            # Colour each bar by region and density
            for j, count in enumerate(counts):
                bin_center = (bins[j] + bins[j+1]) / 2
                if -50 <= bin_center <= 50:
                    patches[j].set_facecolor(high_density_color)
                elif bin_center < -50:
                    patches[j].set_facecolor(low_density_color_negative)
                elif bin_center > 50:
                    patches[j].set_facecolor(low_density_color_positive)
            # Merge the differently coloured qp markers into a single legend entry
            qp_legend_blue = mlines.Line2D([], [], color='w', marker='o', markersize=6, linestyle='None',
                                    markerfacecolor='blue', markeredgecolor='blue', label='qp')
            qp_legend_purple = mlines.Line2D([], [], color='w', marker='o', markersize=6, linestyle='None',
                                            markerfacecolor='purple', markeredgecolor='purple')
            qp_legend_green = mlines.Line2D([], [], color='w', marker='o', markersize=6, linestyle='None',
                                            markerfacecolor='green', markeredgecolor='green')
            # Draw the quantization points
            axes[i].scatter(neg_quant_points, np.zeros_like(neg_quant_points), color='blue', label='qp', zorder=5)
            axes[i].scatter(mid_quant_points, np.zeros_like(mid_quant_points), color='purple', zorder=5, label='qp')
            axes[i].scatter(pos_quant_points, np.zeros_like(pos_quant_points), color='green', zorder=5, label='qp')

            # Draw the clipping bounds
            axes[i].axvline(x=-50, color='black', linestyle='--', linewidth=3, label=symbols('-bp'), )
            axes[i].axvline(x=50, color='black', linestyle=':', linewidth=3, label=symbols('bp'))
            axes[i].axvline(x=global_min, color='red', linestyle='--', linewidth=3, label=symbols('l_a'))
            axes[i].axvline(x=global_max, color='blue', linestyle='--', linewidth=3, label=symbols('u_a'))
            # Title and axis labels
            data_skew = skew(data1)
            axes[i].set_title(f'Sample {i+1} (Range: [{int(data1.min())}, {int(data1.max())}], Skew: {round(data_skew, 2)})', fontsize=20, fontweight='bold')
            axes[i].set_ylabel('Density', fontsize=20, fontweight='bold')
            axes[i].set_xlabel('Activation Value', fontsize=20, fontweight='bold')
            axes[i].tick_params(axis='both', which='major', labelsize=20)
            axes[i].tick_params(axis='both', which='major', labelsize=20, width=2)  # thicker ticks
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontweight('bold')  # bold tick labels

            qp_legend = (qp_legend_blue, qp_legend_purple, qp_legend_green)
        #plt.xlabel("Value")
        """
        handles = [
            qp_legend,  # merged qp legend entry
            mlines.Line2D([], [], color='black', linestyle='--', linewidth=3, label=symbols('-bp')),
            mlines.Line2D([], [], color='black', linestyle=':', linewidth=3, label=symbols('bp')),
            mlines.Line2D([], [], color='red', linestyle='--', linewidth=3, label=symbols('l_a')),
            mlines.Line2D([], [], color='blue', linestyle='--', linewidth=3, label=symbols('u_a'))
        ]
        axes[0].legend(handles=handles, loc='upper left', bbox_to_anchor=(0.7, 1), fontsize=16)
        """
        import matplotlib.font_manager as fm  # font manager

        bold_font = fm.FontProperties(weight='bold', size=16)  # bold font
        axes[0].legend(
            handles=[
                qp_legend,  # add the merged qp entry to the handles
                mlines.Line2D([], [], color='black', linestyle='--', linewidth=3, label=symbols('-bp')),
                mlines.Line2D([], [], color='black', linestyle=':', linewidth=3, label=symbols('bp')),
                mlines.Line2D([], [], color='red', linestyle='--', linewidth=3, label=symbols('l_a')),
                mlines.Line2D([], [], color='blue', linestyle='--', linewidth=3, label=symbols('u_a'))
            ],
            labels=['qp', symbols('-bp'), symbols('bp'), symbols('l_a'), symbols('u_a')],  # custom labels
            loc='upper left', bbox_to_anchor=(0.7, 1), fontsize=16, prop=bold_font, handler_map={tuple: HandlerTuple(ndivide=None)}
        )
        pp = PdfPages(os.path.join("../"+str(filename), f'{name}_dequant.pdf'))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #rect=[0, 0.03, 1, 0.95]
        plt.show()
        jpg_path = os.path.join("../"+str(filename), f'{name}_dequant.jpg')
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=800)
        plt.close()
        pp.close()

if __name__ == '__main__':
    main()