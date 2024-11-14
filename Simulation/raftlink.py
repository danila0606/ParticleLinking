import numpy as np
import pandas as pd

def flow_tracker(pts1, pts2, maxdisp, n_consider, n_use, **kwargs):
    # Set displacement bounds from kwargs
    maxdisp_x = kwargs.get('x', maxdisp)
    maxdisp_y = kwargs.get('y', maxdisp)
    maxdisp_z = kwargs.get('z', maxdisp)

    n_pts1 = pts1.shape[0]
    n_pts2 = pts2.shape[0]

    # Find the nearest n_consider neighbours for pts1 and pts2
    near_neighb_inds_pts1 = np.zeros((n_pts1, n_consider), dtype=int)
    near_neighb_inds_pts2 = np.zeros((n_pts2, n_consider), dtype=int)

    for i in range(n_pts1):
        dists = np.sum((pts1 - pts1[i])**2, axis=1)
        inds = np.argpartition(dists, n_consider + 1)[:n_consider + 1]
        near_neighb_inds_pts1[i] = inds[inds != i][:n_consider]

    for i in range(n_pts2):
        dists = np.sum((pts2 - pts2[i])**2, axis=1)
        inds = np.argpartition(dists, n_consider + 1)[:n_consider + 1]
        near_neighb_inds_pts2[i] = inds[inds != i][:n_consider]

    N = pts1.shape[1]
    nn = []

    for i in range(n_pts1):
        # Find nearby points in pts2 that satisfy displacement constraints
        if N == 1:
            inds_near = (np.sum((pts2 - pts1[i])**2, axis=1) < maxdisp**2) & ((pts2[:, 0] - pts1[i, 0])**2 < maxdisp_x**2)
        elif N == 2:
            inds_near = (np.sum((pts2 - pts1[i])**2, axis=1) < maxdisp**2) & \
                        ((pts2[:, 0] - pts1[i, 0])**2 < maxdisp_x**2) & \
                        ((pts2[:, 1] - pts1[i, 1])**2 < maxdisp_y**2)
        else:
            inds_near = (np.sum((pts2 - pts1[i])**2, axis=1) < maxdisp**2) & \
                        ((pts2[:, 0] - pts1[i, 0])**2 < maxdisp_x**2) & \
                        ((pts2[:, 1] - pts1[i, 1])**2 < maxdisp_y**2) & \
                        ((pts2[:, 2] - pts1[i, 2])**2 < maxdisp_z**2)

        inds_near = np.where(inds_near)[0]

        if len(inds_near) > 0:
            pm = []
            for j in inds_near:
                # Relative positions of nearest neighbours
                ri = pts1[near_neighb_inds_pts1[i]] - pts1[i]
                rj = pts2[near_neighb_inds_pts2[j]] - pts2[j]
                # Calculate the squared distance matrix for relative particle points
                dij = np.sum(ri**2, axis=1)[:, None] + np.sum(rj**2, axis=1) - 2 * np.dot(ri, rj.T)
                # Cost is the sum of distances between n_use points
                pm.append(np.sum(np.sqrt(np.partition(np.min(dij, axis=1), n_use)[:n_use])))

            # Find the minimum penalty and corresponding point in pts2
            penalty = min(pm)
            ind_nn2 = inds_near[np.argmin(pm)]
            nn.append([i, ind_nn2, penalty])

    nn = np.array(nn)


    # Remove duplicate points and sort by penalty
    if len(nn) > 0:
        nn = nn[np.lexsort((nn[:, 2], nn[:, 1]))]
        _, unique_inds = np.unique(nn[:, 1], return_index=True)
        nn = nn[unique_inds]
        nn = nn[np.argsort(nn[:, 0])]

    # Return indices of particle pairs
    idx = nn[:, :2].astype(int) if len(nn) > 0 else np.empty((0, 2), dtype=int)
    return idx


def track_RAFT(xyzt, maxdisp, **kwargs):
    # Set optional parameters from kwargs
    min_track_length = kwargs.get('min_track_length', 2)
    dim = kwargs.get('dim', xyzt.shape[1] - 1)
    n_consider = kwargs.get('n_consider', 10)
    n_use = min(kwargs.get('n_use', 8), n_consider)
    maxdisp_x = kwargs.get('x', maxdisp)
    maxdisp_y = kwargs.get('y', maxdisp)
    maxdisp_z = kwargs.get('z', maxdisp)

    # Validate input parameters
    if min_track_length < 2 or dim < 1 or not isinstance(min_track_length, int) or not isinstance(dim, int):
        raise ValueError("min_track_length must be an integer >= 2 and dim must be an integer >= 1")

    # Extract unique time points and trackable time indices
    times = xyzt[:, -1]
    unique_times = np.unique(times)
    trackable_times = unique_times[np.where(np.diff(unique_times) == 1)[0]]

    if len(trackable_times) == 0:
        raise ValueError("No consecutive time points found for tracking")

    # Track particles between consecutive time points
    res = []
    for t in trackable_times:
        pts1 = xyzt[times == t, :dim]
        pts2 = xyzt[times == t + 1, :dim]
        idx = flow_tracker(pts1, pts2, maxdisp, n_consider, n_use, x=maxdisp_x, y=maxdisp_y, z=maxdisp_z)
        res.append(idx)

    # Combine tracked indices into a complete tracking structure
    # all_tracks = []
    # for i, t in enumerate(trackable_times):
    #     if i == 0:
    #         id_tot = res[i]
    #     else:
    #         new_tracks = []
    #         for track in id_tot:
    #             matching = res[i][res[i][:, 0] == track[-1]]
    #             if len(matching) > 0:
    #                 new_tracks.append(np.append(track, matching[0, 1]))
    #         id_tot = np.array(new_tracks)

    # # Filter out tracks that are shorter than min_track_length
    # valid_tracks = id_tot[np.sum(id_tot != 0, axis=1) >= min_track_length]
    
    xyzti = np.column_stack((xyzt, np.full((xyzt.shape[0], 1), -1)))
    id = 0
    num_static_points = xyzt[times == 0].shape[0]
    for i in range (0, res[0].shape[0]) :
        p1 = res[0][i, 0]
        p2 = res[0][i, 1]
        xyzti[p1, 4] = id
        xyzti[num_static_points + p2, 4] = id
        id = id + 1

    df = pd.DataFrame(xyzti, columns=['x', 'y', 'z', 't', 'id'])
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    df['t'] = df['t'].astype(int)
    df['id'] = df['id'].astype(int)


    # # Create final tracking structure
    # trks = []
    # for i, t in enumerate(unique_times):
    #     pts = xyzt[times == t]
    #     track_indices = valid_tracks[:, i] if i < valid_tracks.shape[1] else []
    #     for track_id in track_indices:
    #         if track_id != 0:
    #             trks.append(np.append(pts[track_id], i + 1))

    # for t in unique_times:
    #     pts = xyzt[times == t]
    #     for i in range(0, pts.shape[0]) :
            
    

    # print("dim:",valid_tracks)
    # trks = np.array(trks)
    # trks = trks[np.lexsort((trks[:, -1], trks[:, -2]))]

    return res[0]




def link_particles_and_compare(static_csv, deformed_csv, maxdisp):
    # Load the static and deformed particle data from CSV files
    static_data = pd.read_csv(static_csv)
    deformed_data = pd.read_csv(deformed_csv)

    # Add time column (0 for static, 1 for deformed)
    static_data['time'] = 0
    deformed_data['time'] = 1

    # Concatenate static and deformed data
    combined_data = pd.concat([static_data, deformed_data], ignore_index=True)
    combined_data['particle'] = -1

    # Convert combined data to numpy array for tracking
    xyzt = combined_data[['x', 'y', 'z', 'time']].to_numpy()
    # Track particles using track_raft function
    tracked_data = track_RAFT(xyzt, maxdisp=maxdisp, min_track_length=2, dim=3)

    for i, pair in enumerate(tracked_data):
        # Update 'particle' for the row with 'time' == 0 and index 'pair[0]'
        combined_data.loc[(combined_data['time'] == 0) & (combined_data.index == pair[0]), 'particle'] = i
        
        # Handle the 'time' == 1 case
        # Get the filtered DataFrame where 'time' == 1
        filtered_data = combined_data[combined_data['time'] == 1]
        
        # Get the original index of the i-th row in the filtered DataFrame
        if pair[1] < len(filtered_data):
            original_index = filtered_data.index[pair[1]]
            combined_data.loc[original_index, 'particle'] = i

    combined_data = combined_data.loc[combined_data['particle'] != -1]

    # Merge the linked results with the original data based on the 'particle' column assigned by trackpy
    linked_static = combined_data[combined_data['time'] == 0][['particle', 'id']].rename(columns={'id': 'id_static'})
    linked_deformed = combined_data[combined_data['time'] == 1][['particle', 'id']].rename(columns={'id': 'id_deformed'})

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


# Example usage:
# link_particles_and_compare('static_particles.csv', 'deformed_particles.csv', maxdisp=10)



# Example usage:
link_particles_and_compare('particle_data_static.csv', 'particle_data_deformed.csv', maxdisp=100)