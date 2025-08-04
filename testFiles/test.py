import numpy as np
import scipy.io
import matplotlib.pyplot as plt



print("\n--- Running Analysis for Part 3.2a ---")
def test():
    try:
        mat_data = scipy.io.loadmat('echart.mat')
        echart_img = mat_data['echart']
        img_row = echart_img[146, :]
        n_row = np.arange(len(img_row))
        
        bdiffh = np.array([1, -1])
        filtered_row = np.convolve(img_row, bdiffh, 'same')

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axs.stem(n_row, img_row, linefmt='k-', markerfmt='ko', basefmt=' ', markersize=2)
        axs.set_title('Figure 4: Filtering of Image Row 147 from echart.mat')
        axs.set_ylabel('Original Pixel Values')
        axs.set_ylim(-0.1, 1.1)
        axs.grid(True)
        axs.stem(n_row, filtered_row, linefmt='m-', markerfmt='mo', basefmt=' ', markersize=4)
        axs.set_ylabel('Filtered Output')
        axs.set_xlabel('Pixel Column Index (n)')
        axs.grid(True)
        plt.tight_layout()
        plt.savefig('figure_4.png')
        plt.show()
        print("Figure 4 saved as figure_4.png")

        neg_impulses = np.where(filtered_row == -1)
        pos_impulses = np.where(filtered_row == 1)
        
        start_edge_idx = neg_impulses
        end_edge_idx = pos_impulses
        width_pixels = end_edge_idx - start_edge_idx
        
        print(f"White-to-black transitions (negative impulses) at indices: {neg_impulses}")
        print(f"Black-to-white transitions (positive impulses) at indices: {pos_impulses}")
        print(f"\nAnalysis of the middle bar of the 'E':")
        print(f"  - Start edge (white-to-black) is at pixel index: {start_edge_idx}")
        print(f"  - End edge (black-to-white) is at pixel index: {end_edge_idx}")
        print(f"  - Calculated width of the black bar: {end_edge_idx} - {start_edge_idx} = {width_pixels} pixels")
    except FileNotFoundError:
        print("\nERROR: 'echart.mat' not found. Please place it in the same directory.")


test()
