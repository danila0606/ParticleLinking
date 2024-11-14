import os
import numpy as np
import pandas as pd
import trackpy as tp
from PIL import Image

# Define folders for the input image stacks
output_folder_static = 'black_image_stack_spherical_particles_full_z_correct_size_tiff_static'
output_folder_rotated = 'black_image_stack_spherical_particles_full_z_correct_size_tiff_rotated_z_axis'

# Load particle data from CSV
particle_data_static = pd.read_csv('particle_data_static.csv')
particle_data_rotated = pd.read_csv('particle_data_rotated.csv')

# Initialize lists to store located particles
located_particles_static = []
located_particles_rotated = []

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.tiff'):
            img = Image.open(os.path.join(folder, filename))
            images.append(np.array(img))
    return images

# Load static images
static_images = load_images_from_folder(output_folder_static)
# Load rotated images
rotated_images = load_images_from_folder(output_folder_rotated)

# Function to locate particles using trackpy
def locate_particles(images):
    # Create a list to store the results
    results = []
    
    # Loop through each frame and locate particles
    for i, image in enumerate(images):
        # Locate particles in the current frame
        particles = tp.locate(image, diameter=25, threshold=10, minmass=10)
        particles['frame'] = i  # Add the frame number to the particles
        results.append(particles)

    # Concatenate all results into a single DataFrame
    return pd.concat(results, ignore_index=True)

# Locate particles in static images
located_particles_static = locate_particles(static_images)
# Locate particles in rotated images
located_particles_rotated = locate_particles(rotated_images)

# Save located particle data to CSV
located_particles_static.to_csv('located_particles_static.csv', index=False)
located_particles_rotated.to_csv('located_particles_rotated.csv', index=False)

# Compare results with the CSV of static particle data
def compare_particles(static_data, located_data):
    static_data = static_data[['x', 'y', 'z']]  # Select relevant columns
    located_data = located_data[['x', 'y', 'z']]  # Select relevant columns
    
    # Merge on x, y, z to find matches
    comparison = pd.merge(static_data, located_data, on=['x', 'y', 'z'], how='outer', indicator=True)
    
    # Count matches, unmatched in static data, and unmatched in located data
    matches = comparison[comparison['_merge'] == 'both']
    unmatched_static = comparison[comparison['_merge'] == 'left_only']
    unmatched_located = comparison[comparison['_merge'] == 'right_only']
    
    return matches, unmatched_static, unmatched_located

# Compare the located particles with the static particle data
matches_static, unmatched_static, unmatched_rotated = compare_particles(particle_data_static, located_particles_static)
matches_rotated, unmatched_static_rotated, unmatched_located = compare_particles(particle_data_rotated, located_particles_rotated)

# Print results
print(f"Matches in static images: {len(matches_static)}")
print(f"Unmatched in static particles: {len(unmatched_static)}")
print(f"Unmatched in located particles: {len(unmatched_rotated)}")
print()
print(f"Matches in rotated images: {len(matches_rotated)}")
print(f"Unmatched in static particles: {len(unmatched_static_rotated)}")
print(f"Unmatched in located particles: {len(unmatched_located)}")

# Save comparison results to CSV
matches_static.to_csv('matches_static.csv', index=False)
unmatched_static.to_csv('unmatched_static.csv', index=False)
unmatched_rotated.to_csv('unmatched_rotated.csv', index=False)

matches_rotated.to_csv('matches_rotated.csv', index=False)
unmatched_static_rotated.to_csv('unmatched_static_rotated.csv', index=False)
unmatched_located.to_csv('unmatched_located.csv', index=False)
