import pandas as pd
import matplotlib.pyplot as plt


CSV_TRACKS = 'linked_data.csv'
SAVE_PLOT_FILENAME = 'tracks.png'
TIME_COLUMN = 'time'


if __name__ == "__main__":
    df = pd.read_csv(CSV_TRACKS)

    max_time = df[TIME_COLUMN].max()

    # Create a scatter plot for the particles
    plt.figure(figsize=(10, 8))

    # Plot all particles with color coding based on 'time'
    for _, row in df.iterrows():
        color = 'blue' if row[TIME_COLUMN] == 0 else ('red' if row[TIME_COLUMN] == max_time else 'black')
        plt.scatter(row['x'], row['y'], color=color, s=3, alpha=1.0)

    # Connect particles with the same 'particle' value in time order
    for particle_id in df['particle'].unique():
        if particle_id < 0:
            continue  # Skip particles with ID < 0
        particle_data = df[df['particle'] == particle_id].sort_values(by=TIME_COLUMN)
        plt.plot(particle_data['x'], particle_data['y'], color='black', alpha=1.0)

    # Add plot labels and show
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle Trajectories')
    plt.grid(True)
    plt.savefig(SAVE_PLOT_FILENAME, bbox_inches='tight')
    plt.show()