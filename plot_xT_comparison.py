#!/usr/bin/env python3
"""
plot_xT_comparison.py - Generate comparison plot showing Input, Reconstruction, and Varying stochastic subcode
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import argparse

def load_and_resize_image(image_path, size=(128, 128)):
    """Load and resize image to specified size."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

def extract_close_up(image, crop_size=(32, 32), position='center'):
    """Extract a close-up region from the image."""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    if position == 'center':
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
    elif position == 'eye':  # For face images, focus on eye region
        start_h = h // 4
        start_w = w // 3
    elif position == 'mouth':  # For face images, focus on mouth region
        start_h = int(h * 0.6)
        start_w = (w - crop_w) // 2
    else:
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
    
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    
    # Ensure bounds are within image
    start_h = max(0, start_h)
    start_w = max(0, start_w)
    end_h = min(h, end_h)
    end_w = min(w, end_w)
    
    return image[start_h:end_h, start_w:end_w]

def compute_mean_and_std(images):
    """Compute mean and standard deviation across multiple images."""
    if len(images) == 0:
        return np.zeros((128, 128, 3)), np.zeros((128, 128, 3))
    
    images_array = np.stack(images, axis=0).astype(np.float32)
    mean_img = np.mean(images_array, axis=0)
    std_img = np.std(images_array, axis=0)
    
    # Normalize std for visualization
    std_img = (std_img / np.max(std_img) * 255).astype(np.uint8)
    mean_img = np.clip(mean_img, 0, 255).astype(np.uint8)
    
    return mean_img, std_img

def create_comparison_plot(input_paths, reconstruction_paths, varying_paths, output_path='xT_comparison.png'):
    """Create the comparison plot with 3 rows of images."""
    
    # Number of rows (subjects)
    n_rows = len(input_paths)
    if n_rows == 0:
        print("No input images provided!")
        return
    
    # Number of varying stochastic images
    n_varying = len(varying_paths[0]) if varying_paths else 6
    
    # Total columns: Input + Reconstruction + Varying images + Close-up + Mean + Std
    n_cols = 2 + n_varying + 2  # Input, Recon, 6 varying, close-up, mean, std
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.1))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Process each row (subject)
    for row in range(n_rows):
        # Load input image
        input_img = load_and_resize_image(input_paths[row])
        axes[row, 0].imshow(input_img)
        axes[row, 0].set_title('Input' if row == 0 else '')
        axes[row, 0].axis('off')
        
        # Load reconstruction image
        recon_img = load_and_resize_image(reconstruction_paths[row])
        axes[row, 1].imshow(recon_img)
        axes[row, 1].set_title('Reconstruction\n($z_{sem}, x_T$)' if row == 0 else '')
        axes[row, 1].axis('off')
        
        # Load varying stochastic images
        varying_images = []
        for col in range(n_varying):
            if col < len(varying_paths[row]):
                varying_img = load_and_resize_image(varying_paths[row][col])
                varying_images.append(varying_img)
            else:
                varying_img = np.zeros((128, 128, 3), dtype=np.uint8)
            
            axes[row, 2 + col].imshow(varying_img)
            if row == 0:
                if col == n_varying // 2:
                    axes[row, 2 + col].set_title('Varying stochastic subcode ($z_{sem}, x_T^i$)')
                else:
                    axes[row, 2 + col].set_title('')
            axes[row, 2 + col].axis('off')
        
        # Create close-up from the middle varying image
        # if len(varying_images) > 0:
        #     close_up_img = varying_images[len(varying_images)//2]
        #     close_up_region = extract_close_up(close_up_img, crop_size=(32, 32), 
        #                                      position='eye' if row == 0 else 'mouth')
        #     # Resize close-up to match other images
        #     close_up_pil = Image.fromarray(close_up_region)
        #     close_up_resized = close_up_pil.resize((128, 128), Image.NEAREST)
        #     close_up_array = np.array(close_up_resized)
        # else:
        #     close_up_array = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # axes[row, 2 + n_varying].imshow(close_up_array)
        # axes[row, 2 + n_varying].set_title('Close-up' if row == 0 else '')
        # axes[row, 2 + n_varying].axis('off')
        
        # Compute mean and standard deviation
        if len(varying_images) > 0:
            mean_img, std_img = compute_mean_and_std(varying_images)
            Image.fromarray(mean_img).save(f'results/xT_comparison/mean_{row}.png')
            Image.fromarray(std_img.mean(axis=2).astype(np.uint8), mode='L').save(f'results/xT_comparison/std_{row}.png')
        else:
            mean_img = np.zeros((128, 128, 3), dtype=np.uint8)
            std_img = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Display mean
        axes[row, 2 + n_varying].imshow(mean_img)
        axes[row, 2 + n_varying].set_title('Mean' if row == 0 else '')
        axes[row, 2 + n_varying].axis('off')
        
        # Display standard deviation (as grayscale)
        std_gray = np.mean(std_img, axis=2)
        axes[row, 2 + n_varying + 1].imshow(std_gray, cmap='gray')
        axes[row, 2 + n_varying + 1].set_title('Standard\ndeviation' if row == 0 else '')
        axes[row, 2 + n_varying + 1].axis('off')
    
    # Adjust layout with reduced spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.001, hspace=0.1)  # Reduced spacing between images
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Comparison plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate xT comparison plot')
    parser.add_argument('--input-paths', nargs='+', required=True,
                       help='Paths to input images (one per row)')
    parser.add_argument('--reconstruction-paths', nargs='+', required=True,
                       help='Paths to reconstruction images (one per row)')
    parser.add_argument('--varying-paths', nargs='+', action='append', required=True,
                       help='Paths to varying stochastic images (use multiple times for each row)')
    parser.add_argument('--output', type=str, default='xT_comparison.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    # Validate input
    n_rows = len(args.input_paths)
    if len(args.reconstruction_paths) != n_rows:
        print("Error: Number of reconstruction images must match number of input images")
        return
    
    if len(args.varying_paths) != n_rows:
        print("Error: Number of varying image sets must match number of input images")
        return
    
    # Create the comparison plot
    create_comparison_plot(
        input_paths=args.input_paths,
        reconstruction_paths=args.reconstruction_paths,
        varying_paths=args.varying_paths,
        output_path=args.output
    )

if __name__ == "__main__":
    # Example usage (uncomment and modify paths as needed):

    # Example for 2 subjects, each with 6 varying images
    input_paths = [
        'imgs_align/622.jpg',
        'imgs_align/869.jpg',
        'imgs_align/21576.jpg',
    ]
    
    reconstruction_paths = [
        'results/xT_comparison/622/reconstruction_10.png',
        'results/xT_comparison/869/reconstruction_10.png',
        'results/xT_comparison/21576/reconstruction_10.png'
    ]
    
    varying_paths = [
        [  # Subject 1 varying images
            'results/xT_comparison/622/output_10_seed_1.png',
            'results/xT_comparison/622/output_10_seed_2.png',
            'results/xT_comparison/622/output_10_seed_3.png',
            'results/xT_comparison/622/output_10_seed_4.png',
        ],
        [  # Subject 2 varying images
            'results/xT_comparison/869/output_images_celeba_pred_male_10_seed_1.png',
            'results/xT_comparison/869/output_images_celeba_pred_male_10_seed_2.png',
            'results/xT_comparison/869/output_images_celeba_pred_male_10_seed_3.png',
            'results/xT_comparison/869/output_images_celeba_pred_male_10_seed_4.png',
        ],
        [  # Subject 3 varying images
            'results/xT_comparison/21576/output_images_celeba_pred_10_seed_1.png',
            'results/xT_comparison/21576/output_images_celeba_pred_10_seed_2.png',
            'results/xT_comparison/21576/output_images_celeba_pred_10_seed_3.png',
            'results/xT_comparison/21576/output_images_celeba_pred_10_seed_4.png',
        ]
    ]
    
    create_comparison_plot(input_paths, reconstruction_paths, varying_paths, 'results/xT_comparison/xT_comparison.png')

    main()
