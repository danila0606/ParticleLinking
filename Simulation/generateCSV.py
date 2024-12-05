import numpy as np
import os
import pandas as pd
import random

# Configuration Parameters
IMAGE_SIZE_XY = 2000        # Size of the XY plane
IMAGE_SIZE_Z = 1            # Size of the Z-axis
NUM_PARTICLES = 20       # Total number of particles
ADD_BLINKING = True         # Toggle blinking behavior
BLINKING_NUM = 2          # Number of particles that blink per iteration
MEMORY = 2                  # Number of iterations a blinked particle remains invisible
ITERATIONS = 10             # Number of deformation iterations
DEFORMATION_FUNCTION = 'linear'  # Options: 'linear', 'crack'
OUTPUT_DIR = 'particle_data_iterations'  # Directory to save CSV files

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Blinking Behavior
def initialize_blinking(num_particles, blinking_num):
    """
    Initialize blinking flags and cooldown timers for particles.
    
    Parameters:
        num_particles (int): Total number of particles.
        blinking_num (int): Number of particles that can blink per iteration.
        
    Returns:
        blink_flags (np.ndarray): Boolean array indicating which particles can blink.
        blink_cooldown (np.ndarray): Integer array tracking cooldown iterations for each particle.
    """
    if ADD_BLINKING:
        # Initially, all particles are eligible to blink
        blink_flags = np.array([True] * num_particles)
        blink_cooldown = np.zeros(num_particles, dtype=int)
    else:
        # If blinking is disabled, no particles can blink
        blink_flags = np.array([False] * num_particles)
        blink_cooldown = np.zeros(num_particles, dtype=int)
    return blink_flags, blink_cooldown

# Generate Random 3D Positions
def generate_positions(num_particles, image_size_xy, image_size_z):
    """
    Generate random 3D positions for particles.
    
    Parameters:
        num_particles (int): Total number of particles.
        image_size_xy (int): Size of the XY plane.
        image_size_z (int): Size of the Z-axis.
        
    Returns:
        x, y, z (np.ndarray): Arrays of x, y, z coordinates.
    """
    np.random.seed(0)
    x = np.random.uniform(low=100, high=image_size_xy - 1100, size=num_particles)
    np.random.seed(1)
    y = np.random.uniform(low=100, high=image_size_xy - 1100, size=num_particles)
    np.random.seed(2)
    z = np.random.uniform(low=0, high=image_size_z, size=num_particles)
    return x, y, z

# Generate Particle Sizes and Mass
def generate_sizes_and_mass(num_particles):
    """
    Generate random sizes and compute mass for each particle.
    
    Parameters:
        num_particles (int): Total number of particles.
        
    Returns:
        sizes, mass (np.ndarray): Arrays of sizes and masses.
    """
    np.random.seed(0)
    sizes = np.random.uniform(low=3, high=4, size=num_particles)  # Radius in 3D
    mass = (4 / 3) * np.pi * (sizes ** 3)  # Volume of a sphere
    return sizes, mass

# Save Static Particle Data
def save_static_particles(x, y, z, mass, particle_ids, output_path):
    """
    Save static particle data to a CSV file.
    
    Parameters:
        x, y, z (np.ndarray): Coordinates of particles.
        mass (np.ndarray): Mass of particles.
        particle_ids (np.ndarray): IDs of particles.
        output_path (str): File path to save the CSV.
    """
    particle_data = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'mass': mass,
        'id': particle_ids
    })
    particle_data.to_csv(output_path, index=False)
    print(f"Static particle data saved to '{output_path}'.")

# Displacement Functions
def calculate_displacement_crack(x, y, K_I, mu, kappa):
    """
    Calculate displacement fields for Mode I crack deformation.
    
    Parameters:
        x, y (np.ndarray): Coordinates relative to crack tip.
        K_I (float): Stress intensity factor.
        mu (float): Shear modulus.
        kappa (float): Parameter related to Poisson's ratio.
        
    Returns:
        u_x, u_y (np.ndarray): Displacement components.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # Angle in radians
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        u_x = (K_I / (2 * mu)) * np.sqrt(r / (2 * np.pi)) * np.cos(theta / 2) * (kappa - 1 + 2 * np.sin(theta / 2)**2)
        u_y = (K_I / (2 * mu)) * np.sqrt(r / (2 * np.pi)) * np.sin(theta / 2) * (kappa + 1 - 2 * np.cos(theta / 2)**2)
        u_x = np.nan_to_num(u_x)
        u_y = np.nan_to_num(u_y)
    
    return u_x, u_y

def calculate_displacement_linear(x, y, iteration, scaling_factor=1.0):
    """
    Calculate linear displacement.
    
    Parameters:
        x, y (np.ndarray): Coordinates.
        iteration (int): Current iteration number.
        scaling_factor (float): Scaling factor for displacement.
        
    Returns:
        u_x, u_y (np.ndarray): Displacement components.
    """
    # u_x = scaling_factor * x / 10.0 * (iteration + 1)  # Progressive displacement
    u_x = np.ones_like(x)
    u_y = np.zeros_like(u_x)  # No displacement in y-direction
    return u_x, u_y

# Apply Deformation Iteratively with Blinking and Memory
def apply_deformation(
    x_original, y_original, z, mass, particle_ids, blink_flags, blink_cooldown,
    iterations, deformation_function='linear', output_dir='particle_data_iterations',
    blinking_num=100, memory=1
):
    """
    Apply deformation to particles over multiple iterations, managing blinking with memory.
    
    Parameters:
        x_original, y_original (np.ndarray): Original coordinates of particles.
        z (np.ndarray): Z-coordinates of particles.
        mass (np.ndarray): Mass of particles.
        particle_ids (np.ndarray): IDs of particles.
        blink_flags (np.ndarray): Boolean array indicating which particles can blink.
        blink_cooldown (np.ndarray): Array tracking cooldown iterations for each particle.
        iterations (int): Number of deformation iterations.
        deformation_function (str): Type of deformation ('linear' or 'crack').
        output_dir (str): Directory to save CSV files.
        blinking_num (int): Number of particles that blink per iteration.
        memory (int): Number of iterations a blinked particle remains invisible.
    """
    # Crack Tip and Mode I Parameters (Adjust as needed)
    K_I = 2      # Stress intensity factor
    mu = 1.0     # Shear modulus
    nu = 0.3     # Poisson's ratio
    kappa = 3 - 4 * nu  # For plane strain
    
    # Initialize current positions
    x_current = np.copy(x_original)
    y_current = np.copy(y_original)
    
    for iteration in range(iterations):
        print(f"Processing iteration {iteration + 1}/{iterations}...")
        
        # Decrement cooldown timers
        blink_cooldown = np.maximum(blink_cooldown - 1, 0)
        
        # Determine eligible particles to blink (can blink and not in cooldown)
        eligible_to_blink = blink_flags & (blink_cooldown == 0)
        eligible_indices = np.where(eligible_to_blink)[0]
        
        # If there are not enough eligible particles, adjust blinking_num
        current_blinking_num = min(blinking_num, len(eligible_indices))
        
        # Randomly select particles to blink
        blinked_indices = np.random.choice(eligible_indices, size=current_blinking_num, replace=False) if current_blinking_num > 0 else np.array([], dtype=int)
        
        # Set cooldown for blinked particles
        blink_cooldown[blinked_indices] = memory
        
        # Create a mask for particles visible in this iteration
        blink_mask = np.ones(NUM_PARTICLES, dtype=bool)
        blink_mask[blinked_indices] = False  # Blinked particles are invisible
        
        # Apply deformation
        if deformation_function == 'linear':
            u_x, u_y = calculate_displacement_linear(
                x_current, y_current - IMAGE_SIZE_XY / 2, iteration, scaling_factor=1.0
            )
        elif deformation_function == 'crack':
            u_x, u_y = calculate_displacement_crack(x_current, y_current - IMAGE_SIZE_XY / 2, K_I, mu, kappa)
        else:
            raise ValueError(f"Unknown deformation function: {deformation_function}")
        
        x_deformed = x_current + u_x
        y_deformed = y_current + u_y
        
        # Validate Deformed Positions
        valid_particles = (
            (x_deformed >= 0) & (x_deformed < IMAGE_SIZE_XY) &
            (y_deformed >= 0) & (y_deformed < IMAGE_SIZE_XY) &
            (z >= 0) & (z < IMAGE_SIZE_Z)
        )
        
        # Combine Validity and Blink Mask
        final_mask = valid_particles & blink_mask
        
        # Create DataFrame for Deformed Particles
        particle_data_deformed = pd.DataFrame({
            'x': x_deformed[final_mask],
            'y': y_deformed[final_mask],
            'z': z[final_mask],
            'mass': mass[final_mask],
            'id': particle_ids[final_mask],
            'ux': u_x[final_mask],
            'uy': u_y[final_mask],
            'iteration': iteration + 1
        })
        
        # Define Filename and Save
        filename = os.path.join(output_dir, f'particle_t_{iteration + 1}.csv')
        particle_data_deformed.to_csv(filename, index=False)
        
        # Update current positions for next iteration
        x_current = x_deformed
        y_current = y_deformed
        
        # Optional: Print blinking information
        print(f"Blinked {current_blinking_num} particles in iteration {iteration + 1}.")
    
    print(f"All {iterations} iterations completed. Deformed particle data saved in '{output_dir}'.")

# Main Execution Flow
def main():
    # Initialize Blinking Flags and Cooldowns
    blink_flags, blink_cooldown = initialize_blinking(NUM_PARTICLES, BLINKING_NUM)
    
    # Generate Positions, Sizes, and Mass
    x, y, z = generate_positions(NUM_PARTICLES, IMAGE_SIZE_XY, IMAGE_SIZE_Z)
    sizes, mass = generate_sizes_and_mass(NUM_PARTICLES)
    particle_ids = np.arange(NUM_PARTICLES)
    
    # Save Static Particles
    static_output_path = os.path.join(OUTPUT_DIR, 'particle_t_0.csv')
    save_static_particles(x, y, z, mass, particle_ids, static_output_path)
    
    # Apply Deformation Iteratively with Blinking and Memory
    apply_deformation(
        x, y, z, mass, particle_ids, blink_flags, blink_cooldown, ITERATIONS,
        deformation_function=DEFORMATION_FUNCTION, output_dir=OUTPUT_DIR,
        blinking_num=BLINKING_NUM, memory=MEMORY
    )

if __name__ == "__main__":
    main()
