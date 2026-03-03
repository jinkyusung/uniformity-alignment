import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import loss  # Imports the loss.py module

def get_robust_mean(times):
    """
    Calculates the mean of the provided time list after removing outliers.
    Uses the Interquartile Range (IQR) method.
    
    Args:
        times: A list or numpy array of time measurements.
        
    Returns:
        The mean value of the filtered data.
    """
    times = np.array(times)
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = np.percentile(times, 25)
    q3 = np.percentile(times, 75)
    iqr = q3 - q1
    
    # Define bounds (standard 1.5 * IQR rule)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter data
    clean_times = times[(times >= lower_bound) & (times <= upper_bound)]
    
    # Return mean of clean data (fallback to original mean if all filtered)
    if len(clean_times) == 0:
        return np.mean(times)
    return np.mean(clean_times)

def benchmark_grid(N_list, d_list, device, iterations=50):
    """
    Runs a grid search benchmark over N (Batch Size) and d (Dimension).
    Collects raw times, removes outliers, and computes the mean.
    """
    # Initialize result matrices
    shape = (len(N_list), len(d_list))
    T_align = np.zeros(shape)
    T_uniform = np.zeros(shape)
    T_wasserstein = np.zeros(shape)

    print(f"Starting Benchmark on {device}")
    print(f"Iterations: {iterations} (Outliers will be removed)")
    print("-" * 50)
    
    # Iterate over the grid
    for i, N in enumerate(N_list):
        N = int(N)
        for j, d in enumerate(d_list):
            d = int(d)
            
            print(f"  Measuring N={N:<4}, d={d:<4}...", end='\r')
            
            # Generate random data
            x = torch.randn(N, d).to(device)
            y = torch.randn(N, d).to(device)

            # Warm-up (Crucial for CUDA kernel initialization)
            loss.alignment_loss(x, y, 2)
            loss.uniformity_loss(x, 2)
            loss.quadratic_wasserstein_loss(x)
            if device.type == 'cuda': torch.cuda.synchronize()

            # Lists to store raw times for this specific (N, d)
            raw_align = []
            raw_uni = []
            raw_wass = []

            # Loop 50 times
            for _ in range(iterations):
                # 1. Alignment Loss
                start = time.time()
                _ = loss.alignment_loss(x, y, 2)
                if device.type == 'cuda': torch.cuda.synchronize()
                raw_align.append(time.time() - start)

                # 2. Uniformity Loss
                start = time.time()
                _ = loss.uniformity_loss(x, 2)
                if device.type == 'cuda': torch.cuda.synchronize()
                raw_uni.append(time.time() - start)

                # 3. Wasserstein Loss
                start = time.time()
                _ = loss.quadratic_wasserstein_loss(x)
                if device.type == 'cuda': torch.cuda.synchronize()
                raw_wass.append(time.time() - start)

            # Compute robust mean (remove outliers) and store
            T_align[i, j] = get_robust_mean(raw_align)
            T_uniform[i, j] = get_robust_mean(raw_uni)
            T_wasserstein[i, j] = get_robust_mean(raw_wass)
            
    print("\nBenchmark Complete.")
    return T_align, T_uniform, T_wasserstein

def save_surface_plots(N_grid, d_grid, T_align, T_uniform, T_wasserstein, filename="benchmark_surface_plot.png"):
    """
    Generates 3D surface plots and saves them to a file instead of showing them.
    """
    print(f"Generating plots and saving to '{filename}'...")
    
    # Convert Tensors to numpy arrays for plotting
    if isinstance(N_grid, torch.Tensor): N_grid = N_grid.numpy()
    if isinstance(d_grid, torch.Tensor): d_grid = d_grid.numpy()

    # Create meshgrid (X=Dimension, Y=Batch Size)
    X, Y = np.meshgrid(d_grid, N_grid)

    fig = plt.figure(figsize=(20, 6))
    
    def plot_single_surface(ax, Z, title):
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension (d)', fontsize=10)
        ax.set_ylabel('Batch Size (N)', fontsize=10)
        ax.set_zlabel('Time (s)', fontsize=10)
        
        # Adjust view angle for best visibility
        ax.view_init(elev=30, azim=220) 
        
        # Add color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.ax.tick_params(labelsize=8)

    # Plot 1: Alignment Loss
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_single_surface(ax1, T_align, "Alignment Loss\n(Element-wise)")

    # Plot 2: Uniformity Loss
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    plot_single_surface(ax2, T_uniform, "Uniformity Loss\n(Pairwise $O(N^2)$)")

    # Plot 3: Wasserstein Loss
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_single_surface(ax3, T_wasserstein, "Wasserstein Loss\n(Robust to N, Sensitive to d)")

    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free memory
    print("Done.")

if __name__ == "__main__":
    # Define Grid
    # N: 16 ~ 2048 (10 steps)
    # d: 16 ~ 128 (10 steps)
    N_grid = torch.linspace(16, 2048, 10).long()
    d_grid = torch.linspace(16, 128, 10).long()
    
    print(f"Batch Sizes (N): {N_grid.tolist()}")
    print(f"Dimensions (d): {d_grid.tolist()}")

    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run Benchmark with 50 iterations per grid point
    times_align, times_uni, times_wass = benchmark_grid(
        N_grid, d_grid, device, iterations=50
    )
    
    # Save Visualization
    save_surface_plots(N_grid, d_grid, times_align, times_uni, times_wass)