import pandas as pd
import trackpy as tp

# Load particle data from CSV (static and deformed, with particle IDs)
static_particles = pd.read_csv('particle_data_static.csv')
deformed_particles = pd.read_csv('particle_data_deformed.csv')

# Add 'frame' information for linking
static_particles['frame'] = 0
deformed_particles['frame'] = 1

# Concatenate both static and deformed particle data for linking
combined_particles = pd.concat([static_particles, deformed_particles], ignore_index=True)

# Perform particle linking using trackpy
linked_particles = tp.link(combined_particles, search_range=60)
# linked_particles = tp.link(combined_particles, pos_columns=['x', 'y'], search_range=(800, 800), adaptive_stop = 0.01, adaptive_step = 0.99)

# Now we compare the linked results with the ground truth
# The ground truth for static and deformed particle IDs are already in the input CSV files.
# We assume the 'id' column in both static and deformed particles is the ground truth.

# Merge the linked results with the original data based on the 'particle' column assigned by trackpy
linked_static = linked_particles[linked_particles['frame'] == 0][['particle', 'id']].rename(columns={'id': 'id_static'})
linked_deformed = linked_particles[linked_particles['frame'] == 1][['particle', 'id']].rename(columns={'id': 'id_deformed'})

# Merge the linked static and deformed particles on 'particle' to get pairs of linked IDs
linked_comparison = pd.merge(linked_static, linked_deformed, on='particle')

# Compare the linked results with the ground truth by checking if the IDs match
linked_comparison['correct_link'] = linked_comparison['id_static'] == linked_comparison['id_deformed']

# Calculate the number of correctly linked particles
correct_links = linked_comparison['correct_link'].sum()
total_links = len(linked_comparison)

# Calculate accuracy
accuracy = correct_links / total_links * 100

# Output the results
print(f"Total particles linked: {total_links}")
print(f"Correctly linked particles: {correct_links}")
print(f"Linking accuracy: {accuracy:.2f}%")

# Optionally save the comparison results to a CSV for analysis
linked_comparison.to_csv('linked_particle_comparison.csv', index=False)