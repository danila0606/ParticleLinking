import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
static_particles = pd.read_csv('real_particle_static.csv')
deformed_particles = pd.read_csv('real_particle_deformed.csv')
# static_particles = pd.read_csv('particle_data_static.csv')
# deformed_particles = pd.read_csv('particle_data_deformed.csv')

connections = pd.read_csv('linked_particle_comparison.csv')

# Ensure the columns are properly named
assert 'x' in static_particles.columns and 'y' in static_particles.columns, "Static particles CSV must have 'x' and 'y' columns."
assert 'x' in deformed_particles.columns and 'y' in deformed_particles.columns, "Deformed particles CSV must have 'x' and 'y' columns."
assert 'id_static' in connections.columns and 'id_deformed' in connections.columns, "Connections CSV must have 'id_static' and 'id_deformed' columns."
assert 'correct_link' in connections.columns, "Connections CSV must have a 'boolean' column."

# Start plotting
plt.figure(figsize=(10, 8))

# Plot static particles
plt.scatter(static_particles['x'], static_particles['y'], color='blue', label='Static Particles', zorder=2, s=1)

# Plot deformed particles
plt.scatter(deformed_particles['x'], deformed_particles['y'], color='red', label='Deformed Particles', zorder=2, s=2)

# Draw lines connecting particles based on the connections file
for _, row in connections.iterrows():
    static_id = int(row['id_static'])
    deformed_id = int(row['id_deformed'])
    is_true = row['correct_link']

    static_x, static_y = static_particles.loc[static_id, ['x', 'y']]
    deformed_x, deformed_y = deformed_particles.loc[deformed_id, ['x', 'y']]

    # Draw line with color based on the boolean value
    line_color = 'green' if is_true else 'purple'
    # if not is_true :
    plt.plot([static_x, deformed_x], [static_y, deformed_y], color=line_color, zorder=1)

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Static and Deformed Particles with Connections')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Show the plot
plt.show()