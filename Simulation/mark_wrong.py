import numpy as np
import pandas as pd
import os
import cv2

# Load particle data from CSV (static and linked comparison data)
static_particles = pd.read_csv('particle_data_static.csv')
linked_comparison = pd.read_csv('linked_particle_comparison.csv')

# Define image sizes
image_size_xy = 600  # Image width and height
image_size_z = 60    # Image depth (number of layers)

# Calculate the radius of each particle from its mass
static_particles['radius'] = ((3 * static_particles['mass']) / (4 * np.pi)) ** (1/3)

# Identify correctly linked, wrongly linked, and unlinked particles
correctly_linked_particles = linked_comparison[linked_comparison['correct_link'] == True]
wrongly_linked_particles = linked_comparison[linked_comparison['correct_link'] == False]
linked_ids = set(linked_comparison['id_static'])
unlinked_particles = static_particles[~static_particles['id'].isin(linked_ids)]

# Create the output directory
output_dir = 'generated_particle_image_stack_with_unlinked'
os.makedirs(output_dir, exist_ok=True)

# Generate image stack with marked particles
for z_index in range(image_size_z):
    # Create a blank RGB image
    image = np.zeros((image_size_xy, image_size_xy, 3), dtype=np.uint8)

    # Mark correctly linked particles (white color)
    for _, row in correctly_linked_particles.iterrows():
        static_particle = static_particles[static_particles['id'] == row['id_static']].iloc[0]
        x, y, z, radius = static_particle['x'], static_particle['y'], static_particle['z'], static_particle['radius']
        
        # Only mark particles that are within the current z slice
        if abs(z - z_index) < radius:
            visible_radius = int(np.sqrt(radius**2 - (z - z_index)**2))
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < image_size_xy and 0 <= y_int < image_size_xy:
                cv2.circle(image, (x_int, y_int), visible_radius, (255, 255, 255), thickness=-1)

    # Mark wrongly linked particles (red color)
    for _, row in wrongly_linked_particles.iterrows():
        static_particle = static_particles[static_particles['id'] == row['id_static']].iloc[0]
        x, y, z, radius = static_particle['x'], static_particle['y'], static_particle['z'], static_particle['radius']
        
        # Only mark particles that are within the current z slice
        if abs(z - z_index) < radius:
            visible_radius = int(np.sqrt(radius**2 - (z - z_index)**2))
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < image_size_xy and 0 <= y_int < image_size_xy:
                cv2.circle(image, (x_int, y_int), visible_radius, (0, 0, 255), thickness=-1)

    # Mark unlinked particles (blue color)
    for _, row in unlinked_particles.iterrows():
        x, y, z, radius = row['x'], row['y'], row['z'], row['radius']

        # Only mark particles that are within the current z slice
        if abs(z - z_index) < radius:
            visible_radius = int(np.sqrt(radius**2 - (z - z_index)**2))
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < image_size_xy and 0 <= y_int < image_size_xy:
                cv2.circle(image, (x_int, y_int), visible_radius, (0, 255, 0), thickness=-1)

    # Save the image slice
    output_path = os.path.join(output_dir, f'slice_{z_index:02d}.png')
    cv2.imwrite(output_path, image)

print(f"Image stack with marked particles saved in '{output_dir}'")
