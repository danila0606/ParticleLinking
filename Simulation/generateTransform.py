import numpy as np
import os
import pandas as pd
from PIL import Image

# Define image sizes
image_size_xy = 2000  # Set the size of the image (400x400)
image_size_z = 1   # Set the size of the z-axis

# Number of particles
num_particles = 2000

# Create folders to save the 2D image slices for original and rotated particles
output_folder_static = 'black_image_stack_spherical_particles_full_z_correct_size_tiff_static'
output_folder_rotated = 'black_image_stack_spherical_particles_full_z_correct_size_tiff_rotated_z_axis'
os.makedirs(output_folder_static, exist_ok=True)
os.makedirs(output_folder_rotated, exist_ok=True)

# Generate random 3D positions for particles with x, y, and z in the range [0, image_size_xy] for x and y, [0, image_size_z] for z
np.random.seed(0)
x = np.random.uniform(low=100, high=image_size_xy - 1100, size=num_particles)
np.random.seed(1)
y = np.random.uniform(low=100, high=image_size_xy - 1100, size=num_particles)
np.random.seed(2)
z = np.random.uniform(low=0, high=image_size_z, size=num_particles)  # Now the z-range is [0, image_size_z]

# Generate random sizes (radius) for each particle
np.random.seed(0)
sizes = np.random.uniform(low=3, high=4, size=num_particles)  # Particle sizes (radius in 3D)

# Calculate mass (volume) of each particle
mass = (4/3) * np.pi * (sizes ** 3)  # Volume of a sphere
particle_ids = np.arange(num_particles)

# Create a DataFrame for particles' data
particle_data = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z,
    'mass': mass,
    'id': particle_ids
})

# Save particle data to CSV for static particles
particle_data.to_csv('particle_data_static.csv', index=False)

z_slices = np.arange(0, image_size_z)  # Now z-slices span the full 0 to image_size_z range

# Function to calculate displacement fields for Mode I
def calculate_displacement_crack(x, y, K_I, mu, kappa):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # Angle in radians
    
    u_x = (K_I / (2 * mu)) * np.sqrt(r / (2 * np.pi)) * np.cos(theta / 2) * (kappa - 1 + 2 * np.sin(theta / 2)**2)
    u_y = (K_I / (2 * mu)) * np.sqrt(r / (2 * np.pi)) * np.sin(theta / 2) * (kappa + 1 - 2 * np.cos(theta / 2)**2)
    
    return u_x, u_y

def calculate_displacement_linear(x, y, K_I, mu, kappa):  
    u_x = 1.0 * x / 10.0
    # u_y = 1.0 * y / 10.0
    u_y = np.zeros_like(u_x)
    
    return u_x, u_y

# Crack tip and Mode I parameters (these would be based on your specific problem)
K_I = 2  # Stress intensity factor (example value)
mu = 1.0   # Shear modulus (example value)
nu = 0.3   # Poisson's ratio
kappa = 3 - 4 * nu  # For plane strain

# Apply displacement to particles using Mode I crack deformation
u_x, u_y = calculate_displacement_linear(x, y - image_size_xy / 2, K_I, mu, kappa)
x_deformed = x + u_x
y_deformed = y + u_y

valid_particles = (
    (x_deformed >= 0) & (x_deformed < image_size_xy) & 
    (y_deformed >= 0) & (y_deformed < image_size_xy) & 
    (z >= 0) & (z < image_size_z)
)

# Save the deformed particle data to CSV
particle_data_deformed = pd.DataFrame({
    'x': x_deformed[valid_particles],
    'y': y_deformed[valid_particles],
    'z': z[valid_particles],
    'mass': mass[valid_particles],
    'id': particle_ids[valid_particles],
    'ux': u_x[valid_particles],
    'uy': u_y[valid_particles]
})
# Save particle data to CSV for rotated particles
particle_data_deformed.to_csv('particle_data_deformed.csv', index=False)

# Function to save images
def save_images(output_folder, is_deformed=False):
    for i, z_slice in enumerate(z_slices):
        # Create a blank black image
        image = np.zeros((image_size_xy, image_size_xy), dtype=np.uint8)  # Create a single-channel (grayscale) image
        # Loop through each particle to determine its appearance in this slice
        for xi, yi, zi, si in zip(x_deformed if is_deformed else x,
                                   y_deformed if is_deformed else y,
                                   z, sizes):
            # Calculate the distance of the current slice from the center of the particle
            distance_from_center = abs(z_slice - zi)

            if (xi < 0) & (xi >= image_size_xy) & \
                (yi < 0) & (yi >= image_size_xy) & \
                (zi < 0) & (zi >= image_size_z) :
                continue
            
            # If the distance is less than the radius, the particle should appear in this slice
            if distance_from_center < si:
                # Calculate the radius of the circle in this slice using the formula
                slice_radius = np.sqrt(si**2 - distance_from_center**2)

                # Convert coordinates to integer and ensure they are within the image bounds
                xi_int = int(np.clip(xi, 0, image_size_xy - 1))
                yi_int = int(np.clip(yi, 0, image_size_xy - 1))

                # Create a mask for the circle
                for x_offset in range(-int(slice_radius), int(slice_radius) + 1):
                    for y_offset in range(-int(slice_radius), int(slice_radius) + 1):
                        if x_offset**2 + y_offset**2 <= slice_radius**2:
                            x_pixel = xi_int + x_offset
                            y_pixel = yi_int + y_offset
                            if 0 <= x_pixel < image_size_xy and 0 <= y_pixel < image_size_xy:
                                image[y_pixel, x_pixel] = 255  # Set pixel to white

        # Save the image as a TIFF file
        if is_deformed:
            filename = os.path.join(output_folder_rotated, f'slice_deformed_z_{i:02d}.tiff')
        else:
            filename = os.path.join(output_folder_static, f'slice_static_{i:02d}.tiff')
        
        img = Image.fromarray(image)  # Convert numpy array to PIL image
        img.save(filename, format='TIFF')

# Save static images
save_images(output_folder_static, is_deformed=False)
# Save rotated images
save_images(output_folder_rotated, is_deformed=True)

print(f"30 layers of {image_size_xy}x{image_size_xy} TIFF images with static particles saved to '{output_folder_static}' folder.")
print(f"30 layers of {image_size_xy}x{image_size_xy} TIFF images with spherical particles rotated around the z-axis saved to '{output_folder_rotated}' folder.")
print("Particle data saved to 'particle_data_static.csv' and 'particle_data_rotated.csv'.")
