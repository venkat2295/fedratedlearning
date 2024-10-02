import numpy as np
import struct
import gzip
import os
import matplotlib.pyplot as plt

def read_idx(filename):
    """Read IDX file format used for MNIST dataset.
    
    Args:
        filename (str): Path to the IDX file
        
    Returns:
        numpy.ndarray: Numpy array containing the data
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def read_gz_idx(filename):
    """Read gzipped IDX file format.
    
    Args:
        filename (str): Path to the gzipped IDX file
        
    Returns:
        numpy.ndarray: Numpy array containing the data
    """
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def load_mnist(data_dir, use_gz=False):
    """Load MNIST data from ubyte or ubyte.gz files.
    
    Args:
        data_dir (str): Directory containing the MNIST files
        use_gz (bool): Whether to use gzipped files
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    read_func = read_gz_idx if use_gz else read_idx
    ext = '.gz' if use_gz else ''
    
    try:
        # Load training data
        x_train = read_func(os.path.join(data_dir, f'train-images-idx3-ubyte{ext}'))
        y_train = read_func(os.path.join(data_dir, f'train-labels-idx1-ubyte{ext}'))
        
        # Load test data
        x_test = read_func(os.path.join(data_dir, f't10k-images-idx3-ubyte{ext}'))
        y_test = read_func(os.path.join(data_dir, f't10k-labels-idx1-ubyte{ext}'))
        
        return (x_train, y_train), (x_test, y_test)
    
    except FileNotFoundError as e:
        print(f"Error: Required MNIST files not found in {data_dir}")
        print("Expected files (with or without .gz):")
        print("  - train-images-idx3-ubyte")
        print("  - train-labels-idx1-ubyte")
        print("  - t10k-images-idx3-ubyte")
        print("  - t10k-labels-idx1-ubyte")
        raise e

def view_mnist_data(data_dir, use_gz=False):
    """Views the MNIST data from ubyte or ubyte.gz files.
    
    Args:
        data_dir (str): Directory containing the MNIST files
        use_gz (bool): Whether to use gzipped files
    """
    try:
        # Load the MNIST data
        (x_train, y_train), (x_test, y_test) = load_mnist(data_dir, use_gz)
        
        # Print the shape of the training and test sets
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Sample MNIST Images')
        
        # Display the first 10 images
        for i in range(10):
            row = i // 5
            col = i % 5
            axs[row, col].imshow(x_train[i], cmap='gray')
            axs[row, col].set_title(f'Label: {y_train[i]}')
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return (x_train, y_train), (x_test, y_test)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
   
    data_dir = "./data/MNIST/raw"  
    
    # Try with regular ubyte files first
    try:
        data = view_mnist_data(data_dir, use_gz=False)
    except FileNotFoundError:
        # If regular files not found, try gzipped files
        print("Trying gzipped files...")
        data = view_mnist_data(data_dir, use_gz=True)