import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# from tqdm import tqdm


def get_near_inds(coord, pts, maxdisp, coef = 1.0, **kwargs) :
    maxdisp   = coef * maxdisp
    maxdisp_x = coef * kwargs.get('x', maxdisp)
    maxdisp_y = coef * kwargs.get('y', maxdisp)
    maxdisp_z = coef * kwargs.get('z', maxdisp)

    N = pts.shape[1]

    if N == 1:
        inds_near = (np.sum((pts - coord)**2, axis=1) < maxdisp**2) & ((pts[:, 0] - coord[0])**2 < maxdisp_x**2)
    elif N == 2:
        inds_near = (np.sum((pts - coord)**2, axis=1) < maxdisp**2) & \
                    ((pts[:, 0] - coord[0])**2 < maxdisp_x**2) & \
                    ((pts[:, 1] - coord[1])**2 < maxdisp_y**2)
    else:
        inds_near = (np.sum((pts - coord)**2, axis=1) < maxdisp**2) & \
                    ((pts[:, 0] - coord[0])**2 < maxdisp_x**2) & \
                    ((pts[:, 1] - coord[1])**2 < maxdisp_y**2) & \
                    ((pts[:, 2] - coord[2])**2 < maxdisp_z**2)
    
    return np.where(inds_near)[0]


def sample_start_point(pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2, n_use, tries, maxdisp, sample_search_range_coef, **kwargs) :
    n_pts1 = pts1.shape[0]
    n_pts2 = pts2.shape[0]

    sample_candidates = random.sample(range(0, n_pts1), tries)

    errors = []
    dest_ids = []
    for i in sample_candidates:
        inds_near = get_near_inds(pts1[i], pts2, maxdisp, sample_search_range_coef, **kwargs)

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
            dest_ids.append(inds_near[np.argmin(pm)])
            errors.append(penalty)
        else :
            errors.append(float("inf"))
            dest_ids.append(-1)

    errors = np.array(errors)
    return sample_candidates[np.argmin(errors)], dest_ids[np.argmin(errors)], np.min(errors)



class LinkInfo :
    def __init__(self, src_id, dest_id, error, banned_dest_ids):
        self.src_id = src_id
        self.dest_id = dest_id
        self.error = error
        self.banned_dest_ids = banned_dest_ids


def reliability_flow_tracker(pts1, pts2, maxdisp, n_consider, n_use, sample_ratio = 0.1, sample_search_range_coef = 3.5, **kwargs):
    n_pts1 = pts1.shape[0]
    n_pts2 = pts2.shape[0]

    # Find the nearest n_consider neighbours for pts1 and pts2
    near_neighb_inds_pts1 = np.zeros((n_pts1, n_consider), dtype=int)
    near_neighb_inds_pts2 = np.zeros((n_pts2, n_consider), dtype=int)

    for i in range(n_pts1):
        dists = np.sum((pts1 - pts1[i])**2, axis=1)
        inds = np.argsort(dists)[:n_consider + 1]
        near_neighb_inds_pts1[i] = inds[inds != i][:n_consider]

    for i in range(n_pts2):
        dists = np.sum((pts2 - pts2[i])**2, axis=1)
        inds = np.argsort(dists)[:n_consider + 1]
        near_neighb_inds_pts2[i] = inds[inds != i][:n_consider]


    #Sampling
    tries = int(sample_ratio * n_pts1)
    start_id, dest_id, error = sample_start_point(pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2, n_use, tries, maxdisp, sample_search_range_coef, **kwargs)
    print("Sampling done! ", start_id, dest_id, error)

    source_pts_stack = []
    # source_id, error, banned_dest_ids
    source_pts_stack.append(LinkInfo(start_id, dest_id, error, []))
    if (dest_id == -1) :
        raise ValueError("Bad sampling, try to change params!")
    linked_source_pts = np.full(n_pts1, -1)
    linked_dest_pts = np.full(n_pts2, -1)
    
    linked_source_pts[start_id] = dest_id
    linked_dest_pts[dest_id] = start_id

    predict_stack_p = 0
    while (len(source_pts_stack) < n_pts1) :
        predict_linked_pt_info = source_pts_stack[predict_stack_p]
        if (predict_linked_pt_info.dest_id == -2) :
            predict_stack_p = predict_stack_p - 1
            if (predict_stack_p < 0) :
                raise ValueError("Can't good predictor, try to change params!")
            continue

        predict_pt_neighbours = near_neighb_inds_pts1[predict_linked_pt_info.src_id]

        next_src_pt_id = -1
        for neighbour_id in predict_pt_neighbours :
            if (linked_source_pts[neighbour_id] == -1) : # or (linked_source_pts[neighbour_id] == -2)
                next_src_pt_id = neighbour_id
                break

        # if (next_src_pt_id == -1) :
        #     raise ValueError("Can't find free neighbour for {predict_linked_pt_info[0]}, try to change params!")

        # TODODODODODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo
        # Dont raise, because sometime generates far away predict particles which has no free neighbour particles, in this case 
        # take free particles, find close neigbour with dest_id != -2 or -1, take this prediction
        if (next_src_pt_id == -1) :
            predict_stack_p = predict_stack_p - 1
            if (predict_stack_p < 0) :
                for i in range(0, linked_source_pts.shape[0]) :
                    if (linked_source_pts[i] == -1) :
                        next_src_pt_id = i
                        dists_tmp = np.sum((pts1 - pts1[next_src_pt_id])**2, axis=1)
                        inds_tmp = np.argsort(dists_tmp)
                        neighbours_free_pt = inds_tmp[inds_tmp != i]
                        for k in range (neighbours_free_pt.shape[0]) :
                            if (linked_source_pts[neighbours_free_pt[k]] >= 0) :
                                predict_linked_pt_info = LinkInfo(neighbours_free_pt[k], linked_source_pts[neighbours_free_pt[k]], 0, [])
                                break

                        break
                if (next_src_pt_id == -1) : # Smth wrong!!!
                    raise ValueError("Can't find next free particle, try to change params!")
            else :
                continue
        
    # Find nearby points in pts2 that satisfy displacement constraints
        prediction_uvz = (pts2[predict_linked_pt_info.dest_id] - pts1[predict_linked_pt_info.src_id])
        # OLD WAY
        inds_near = get_near_inds(pts1[next_src_pt_id] + prediction_uvz, pts2, maxdisp, **kwargs) # add prediction
        # NEW WAY
        if (len(inds_near) == 0) :
            dists_tmp = np.sum((pts2 - (pts1[next_src_pt_id] + prediction_uvz))**2, axis=1)
            inds_near = np.argsort(dists_tmp)[:n_consider + 1]
            inds_near = inds_near[inds_near != i][:n_consider]
        if len(inds_near) > 0:
            pm = []
            for j in inds_near:
                # Relative positions of nearest neighbours
                ri = pts1[near_neighb_inds_pts1[next_src_pt_id]] - pts1[next_src_pt_id]
                rj = pts2[near_neighb_inds_pts2[j]] - pts2[j]
                # Calculate the squared distance matrix for relative particle points
                dij = np.sum(ri**2, axis=1)[:, None] + np.sum(rj**2, axis=1) - 2 * np.dot(ri, rj.T)
                # Cost is the sum of distances between n_use points
                pm.append(np.sum(np.sqrt(np.partition(np.min(dij, axis=1), n_use)[:n_use])))

            # Find the minimum penalty and corresponding point in pts2
            dest_id = -1

            while(dest_id == -1) :
                penalty = min(pm)
                ind_nn2 = inds_near[np.argmin(pm)]
                if (linked_dest_pts[ind_nn2] == -1) :
                    dest_id = ind_nn2
                    break
                else :
                    bad_linked_src_id = linked_dest_pts[ind_nn2]
                    for link_info_id in range(0, len(source_pts_stack)) :# add pointer in linked_source_pts
                        link_info = source_pts_stack[link_info_id]
                        if (link_info.src_id == bad_linked_src_id) :
                            error = link_info.error
                            break
                    
                    if (error <= penalty) :
                        np.delete(inds_near, np.argmin(pm))
                        del pm[np.argmin(pm)]
                        if (len(pm) == 0) :
                            dest_id = -2
                            break
                            # raise ValueError("Can't find good destination candidates for the particle, try to change params!")
                        continue
                    else :
                        dest_id = ind_nn2
                        for k in range(link_info_id, len(source_pts_stack)) :
                            linked_source_pts[source_pts_stack[k].src_id] = -1
                            linked_dest_pts[source_pts_stack[k].dest_id] = -1
                        # linked_source_pts[bad_linked_src_id] = -1
                        del source_pts_stack[link_info_id : len(source_pts_stack)]
            
            linked_source_pts[next_src_pt_id] = dest_id
            if (dest_id != -2) :
                linked_dest_pts[dest_id] = next_src_pt_id

            predict_stack_p = len(source_pts_stack)
            source_pts_stack.append(LinkInfo(next_src_pt_id, dest_id, penalty, []))

        else :
            raise ValueError("Can't find neighbours for the particle, try to change params!")
    
    trace = []
    for link_info in source_pts_stack :
        trace.append(np.array(pts1[link_info.src_id][:2]))

    return np.column_stack((np.arange(linked_source_pts.shape[0]), linked_source_pts)), trace


def track_reliability_RAFT(id_xyzt, maxdisp, **kwargs):
    # Set optional parameters from kwargs
    xyzt = id_xyzt[['x', 'y', 'z', 'time']].to_numpy()

    dim = kwargs.get('dim', xyzt.shape[1] - 1)
    n_consider = kwargs.get('n_consider', 10)
    n_use = min(kwargs.get('n_use', 8), n_consider)


    # Extract unique time points and trackable time indices
    times = xyzt[:, -1]
    unique_times = np.unique(times)
    trackable_times = unique_times[np.where(np.diff(unique_times) == 1)[0]]

    if len(trackable_times) == 0:
        raise ValueError("No consecutive time points found for tracking")

    pts1 = xyzt[times == trackable_times[0], :dim]
    pts2 = xyzt[times == trackable_times[0] + 1, :dim]
    res, trace = reliability_flow_tracker(pts1, pts2, maxdisp, n_consider, n_use, **kwargs)

    xyzti = np.column_stack((xyzt, np.full((xyzt.shape[0], 1), -1)))
    id = 0
    num_static_points = xyzt[times == 0].shape[0]
    for i in range (0, res.shape[0]) :
        p1 = res[i, 0]
        p2 = res[i, 1]
        if (p1 < 0) :
            raise ValueError("Something wrong happened!")
        xyzti[p1, 4] = id
        if (p2 < 0) :
            xyzti[num_static_points + p2, 4] = -1
        else :
            xyzti[num_static_points + p2, 4] = id            
        id = id + 1

    df = pd.DataFrame(xyzti, columns=['x', 'y', 'z', 'time', 'particle'])
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    df['time'] = df['time'].astype(int)
    df['particle'] = df['particle'].astype(int)

    return pd.concat([id_xyzt[['id']], df], axis = 1), trace

def save_track_path_to_image(trace, image_path) :
    min_x = min(p[0] for p in trace)
    max_x = max(p[0] for p in trace)
    min_y = min(p[1] for p in trace)
    max_y = max(p[1] for p in trace)

    # Create a black image with adjusted size
    image_size = (int(max_y - min_y) + 1, int(max_x - min_x) + 1)
    image = np.zeros(image_size)
    n = len(trace)
    # Create a color map based on the index
    indices = np.arange(n)
    colors = plt.cm.Reds(indices / n)  # Use 'viridis' colormap; normalize by dividing by `n`

    # Plot the particles on the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')  # Display the black background

    # Plot each particle with its color based on the index
    for i, particle in enumerate(trace):
        plt.scatter(particle[0] - min_x, particle[1] - min_y, color=colors[i], s=50)  # Adjust `s` for marker size

    # Add a colorbar to show the mapping of index to color
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0, n))
    plt.colorbar(sm, label='Particle Index')

    plt.title("Particles Placed on a Black Image")
    plt.axis('off')  # Hide axis if desired
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()


def link_particles_and_compare(static_csv, deformed_csv, maxdisp):
    # Load the static and deformed particle data from CSV files
    static_data = pd.read_csv(static_csv)
    deformed_data = pd.read_csv(deformed_csv)

    # Add time column (0 for static, 1 for deformed)
    static_data['time'] = 0
    deformed_data['time'] = 1

    # Concatenate static and deformed data
    combined_data = pd.concat([static_data, deformed_data], ignore_index=True)

    # Track particles using track_raft function
    tracked_data, trace = track_reliability_RAFT(combined_data, maxdisp=maxdisp, dim=2)
    save_track_path_to_image(trace, 'trace.png')


    linked_static = tracked_data[tracked_data['time'] == 0][['particle', 'id']].rename(columns={'id': 'id_static'})
    linked_deformed = tracked_data[tracked_data['time'] == 1][['particle', 'id']].rename(columns={'id': 'id_deformed'})

    # Merge the linked static and deformed particles on 'particle' to get pairs of linked IDs
    linked_comparison = pd.merge(linked_static, linked_deformed, on='particle')
    # print(linked_comparison)

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
link_particles_and_compare('particle_data_static.csv', 'particle_data_deformed.csv', maxdisp=25)