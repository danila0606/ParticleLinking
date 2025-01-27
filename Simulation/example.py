from RAFTlink import REL_RAFT_link
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np

def choose_start_GUI(data1, data2) :
    selected_particles = [-1, -1]

    def on_click_left(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            distances = ((data1['x'] - x) ** 2 + (data1['y'] - y) ** 2).pow(0.5)
            selected_particles[0] = distances.idxmin()
            print(f"Selected particle from t=0: ID={selected_particles[0]}")

    def on_click_right(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            distances = ((data2['x'] - x) ** 2 + (data2['y'] - y) ** 2).pow(0.5)
            selected_particles[1] = distances.idxmin()
            print(f"Selected particle from t=1: ID={selected_particles[1]}")

    def on_key(event):
        if event.key == "enter":
            plt.close("all")
            print("Selected Particles:")
            print(f"t=0 Particle ID: {selected_particles[0]}")
            print(f"t=1 Particle ID: {selected_particles[1]}")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(data1['x'], data1['y'], color='blue', label='File 1 Particles', zorder=2, s=2)
    ax1.set_title('Particles from time 0 (Left)')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)
    fig1.canvas.mpl_connect('button_press_event', on_click_left)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(data2['x'], data2['y'], color='red', label='File 2 Particles', zorder=2, s=2)
    ax2.set_title('Particles from time 1 (Right)')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True)
    fig2.canvas.mpl_connect('button_press_event', on_click_right)

    fig1.canvas.mpl_connect('key_press_event', on_key)
    fig2.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return selected_particles

def get_csv_filenames(folder) :
    num_pattern = re.compile(r'\d+')
    filenames = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    sorted_filenames = sorted(
        filenames,
        key=lambda x: int(num_pattern.search(x).group()) if num_pattern.search(x) else float('inf')
    )
    
    return sorted_filenames

def save_track_path_to_image(df, image_path) :
    df = df[df['trace'] >= 0]
    unique_frames = df['time'].unique()
    
    for frame in unique_frames:
        df_current = df[df['time'] == frame]

        x_min, x_max = df_current['x'].min(), df_current['x'].max()
        y_min, y_max = df_current['y'].min(), df_current['y'].max()

        plt.figure(figsize=(6, 6))

        plt.xlim(x_min - 1, x_max + 1)
        plt.ylim(y_min - 1, y_max + 1)

        plt.scatter(df_current['x'], df_current['y'], c=df_current['trace'], cmap='Reds', s=15)

        plt.colorbar(label='Linked Order')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Particles at Frame {frame}')
        plt.grid(True)
        plt.savefig(image_path + '/trace_{i}_{j}'.format(i=frame,j=frame+1), bbox_inches='tight')
        

def check_linking_accuracy(tracked_data, static_time, dynamic_time) :
    linked_static = tracked_data[tracked_data['time'] == static_time][['particle', 'id']].rename(columns={'id': f'id_static'})
    linked_static_filtered = linked_static[linked_static['particle'] != -1]
    linked_deformed = tracked_data[tracked_data['time'] == dynamic_time][['particle', 'id']].rename(columns={'id': f'id_deformed'})
    linked_deformed_filtered = linked_deformed[linked_deformed['particle'] != -1]

    # Merge particles on 'particle' to get pairs of linked IDs
    linked_comparison = pd.merge(linked_static_filtered, linked_deformed_filtered, on='particle', how='inner')

    linked_comparison['correct_link'] = linked_comparison['id_static'] == linked_comparison['id_deformed']
    # Calculate the number of correctly linked particles
    correct_links = linked_comparison['correct_link'].sum()
    total_links = len(linked_comparison)

    # Calculate accuracy
    accuracy = correct_links / total_links * 100

    # Output the results
    print(f"Total tracks: {total_links}")
    print(f"Correctly found tracks: {correct_links}")
    print(f"Linking accuracy: {accuracy:.2f}%")

if __name__ == "__main__":

    CSV_FOLDER = 'big_disp_data'
    # CSV_FOLDER = 'memory_particle_data'
    
    DIM = 3
    COLUMN_NAMES = ['time', 'z', 'y', 'x']

    USE_IDS_FROM_GUI = False
    first_ids = []
    my_predictors = [[np.array([458, 635, 205]), np.array([458, 665, 205])]]
    SAMPLE_RATIO = 0.03
    SAMPLING_SEARCH_RADIUS_COEF = 2.5

    ERROR_FUNCTION = 'STRAIN' # 'L2' or 'STRAIN'
    SIGMA_THRESHOLD = 3.0

    MAX_DISP = 20
    N_CONSIDER = 15
    N_USE = 10

    SAVE_TRACE = True
    TRACE_PATH = 'trace'

    CHECK_LINKING_ACCURACY = False
    LINKING_DATA_FILENAME = 'linked_data.csv'

    MEMORY=0

    csv_file_names_list = get_csv_filenames(CSV_FOLDER)

    data = []
    for i in range(len(csv_file_names_list)) :
        tmp_data = pd.read_csv(CSV_FOLDER + '/' + csv_file_names_list[i])
        tmp_data[COLUMN_NAMES[0]] = i
        data.append(tmp_data)
    
    combined_data = pd.concat(data, ignore_index=True)

    if USE_IDS_FROM_GUI :
        for i in range(len(csv_file_names_list) - 1) :
            first_ids.append(choose_start_GUI(data[i], data[i + 1]))

    tracked_data = REL_RAFT_link(combined_data, DIM, maxdisp=MAX_DISP, \
                                 column_names=COLUMN_NAMES,
                                first_ids=first_ids, my_predictors=my_predictors, \
                                sample_ratio=SAMPLE_RATIO, sample_search_range_coef=SAMPLING_SEARCH_RADIUS_COEF, \
                                n_consider=N_CONSIDER, n_use=N_USE, \
                                error_f=ERROR_FUNCTION, sigma_threshold=SIGMA_THRESHOLD, \
                                memory=MEMORY)

    if SAVE_TRACE :
        if not os.path.exists(TRACE_PATH):
            os.makedirs(TRACE_PATH)

        save_track_path_to_image(tracked_data, TRACE_PATH)

    if CHECK_LINKING_ACCURACY :
        check_linking_accuracy(tracked_data, static_time=0, dynamic_time=len(csv_file_names_list)-1)

    tracked_data.to_csv(LINKING_DATA_FILENAME, index=False)
