import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class ReliabilityRAFTSolver :
    class LinkInfo:
        def __init__(self, id=-1, error=float('inf')):
            self.id = id
            self.error = error

    def __init__(self, dim, init_prediction_method, maxdisp, sample_ratio = 0.1, sample_search_range_coef = 3.5, first_ids = None, **kwargs) :
        if init_prediction_method not in ['SampleRandom', 'UseProvided']:
            raise ValueError(f"Invalid prediction method: {init_prediction_method}")
        
        self.init_prediction_method = init_prediction_method
        self.sample_search_range_coef = sample_search_range_coef
        self.sample_ratio = sample_ratio
        self.maxdisp = maxdisp

        self.maxdisp_x = kwargs.get('x', maxdisp)
        self.maxdisp_y = kwargs.get('y', maxdisp)
        self.maxdisp_z = kwargs.get('z', maxdisp)

        self.dim = dim
        self.n_consider = kwargs.get('n_consider', 10)
        self.n_use = min(kwargs.get('n_use', 8), self.n_consider)

        self.first_ids = first_ids

        self.drop_stack = False # True
        self.predictors_consider = 5

    def track_reliability_RAFT(self, id_xyzt):
        # Set optional parameters from kwargs
        xyzt = id_xyzt[['x', 'y', 'z', 'time']].to_numpy()

        # Extract unique time points and trackable time indices
        times = xyzt[:, -1]
        unique_times = np.unique(times)
        trackable_times = unique_times[np.where(np.diff(unique_times) == 1)[0]]

        if len(trackable_times) == 0:
            raise ValueError("No consecutive time points found for tracking")

        pts1 = xyzt[times == trackable_times[0], :self.dim]
        pts2 = xyzt[times == trackable_times[0] + 1, :self.dim]
        res, trace = self.__reliability_flow_tracker__(pts1, pts2)

        xyzti = np.column_stack((xyzt, np.full((xyzt.shape[0], 1), -1)))
        id = 0
        num_static_points = xyzt[times == 0].shape[0]
        for i in range (0, res.shape[0]) :
            p1 = res[i, 0]
            p2 = res[i, 1]
            if (p1 < 0) :
                raise ValueError("Something wrong happened!")
            xyzti[p1, 4] = id
            # if (p2 < 0) :
            #     xyzti[num_static_points + p2, 4] = -1
            # else :
            if (p2 > 0) :
                xyzti[num_static_points + p2, 4] = id            
            id = id + 1

        df = pd.DataFrame(xyzti, columns=['x', 'y', 'z', 'time', 'particle'])
        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)
        df['z'] = df['z'].astype(float)
        df['time'] = df['time'].astype(int)
        df['particle'] = df['particle'].astype(int)

        return pd.concat([id_xyzt[['id']], df], axis = 1), trace

    def __find_nearest_neighbors__(self, pts, n_neighbors):
        dists = np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.argsort(dists, axis=1)[:, :n_neighbors]
    
    def __find_unlinked_source__(self, linked_source, pts1):
        for i, link in enumerate(linked_source):
            if link.id == -1:
                dists = np.sum((pts1 - pts1[i])**2, axis=1)
                neighbors = np.argsort(dists)
                for neighbor in neighbors:
                    if linked_source[neighbor].id >= 0 and dists[neighbor] > self.maxdisp * self.sample_search_range_coef:
                        return i
        return -1
    
    def __find_close_predictors__(self, linked_source, pts1, src_id, predictors_consider) :
        inds_tmp = np.argsort(np.sum((pts1 - pts1[src_id])**2, axis=1))
        predictor_ids = inds_tmp[inds_tmp != src_id]
        predictors_infos = []
        for neighbour in predictor_ids :
            if (len(predictors_infos) > self.predictors_consider) :
                break
            if (linked_source[neighbour].id >= 0) :
                predictors_infos.append(linked_source[neighbour])

        return predictors_infos

    def __reliability_flow_tracker__(self, pts1, pts2):
        n_pts1, n_pts2 = pts1.shape[0], pts2.shape[0]

        # Find the nearest n_consider neighbours for pts1 and pts2
        near_neighb_inds_pts1 = self.__find_nearest_neighbors__(pts1, self.n_consider)
        near_neighb_inds_pts2 = self.__find_nearest_neighbors__(pts2, self.n_consider)

        #Sampling
        start_id, dest_id, error = self.__sample_start_point__(pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)
        if dest_id == -1:
            raise ValueError("Bad sampling, try to change params!")

        linked_source_pts = [self.LinkInfo() for _ in range(n_pts1)]
        linked_dest_pts = np.full(n_pts2, -1, dtype=int)
        errors = np.full(n_pts1, -1.0)

        linked_source_pts[start_id] = self.LinkInfo(dest_id, error)
        errors[start_id] = error
        linked_dest_pts[dest_id] = start_id

        source_pts_stack = [start_id]
        cur_stack_p = 0

        while len(source_pts_stack) < n_pts1 :
            src_id = source_pts_stack[cur_stack_p]
            last_linked_pt_info = linked_source_pts[src_id]
            if (last_linked_pt_info.id == -2) :
                cur_stack_p -= 1
                if (cur_stack_p < 0) :
                    raise ValueError("Can't good predictor, try to change params!")
                continue

            last_pt_neighbours = near_neighb_inds_pts1[src_id]
            next_src_pt_id = next((n for n in last_pt_neighbours if linked_source_pts[n].id == -1), -1)

            if next_src_pt_id == -1 :
                cur_stack_p -= 1
                if cur_stack_p < 0 :
                    next_src_pt_id = self.__find_unlinked_source__(linked_source_pts, pts1)
                    if (next_src_pt_id == -1) : # Smth wrong!!!
                        raise ValueError("Can't find next free particle, try to change params!")
                else :
                    continue
            
            predictors_infos = [linked_source_pts[n] for n in near_neighb_inds_pts1[next_src_pt_id] if linked_source_pts[n].id >= 0]

            if (len(predictors_infos) == 0) :
                cur_stack_p -= 1
                if (cur_stack_p < 0) : # taking any closest predictor
                    predictors_infos = self.__find_close_predictors__(linked_source_pts, pts1, next_src_pt_id, self.predictors_consider)
                else :
                    continue
            
            # sort by disp, take the middle one
            predictors_infos = self.__get_reasonable_predictors__(predictors_infos)
            prediction = np.mean([pts2[p.id] - pts1[linked_dest_pts[p.id]] for p in predictors_infos], axis=0)
            
            inds_near = self.__get_near_inds__(pts1[next_src_pt_id] + prediction, pts2) # add prediction
            if inds_near.size == 0 :
                inds_near = np.argsort(np.sum((pts2 - (pts1[next_src_pt_id] + prediction))**2, axis=1))[:self.n_consider]
                inds_near = inds_near[inds_near != next_src_pt_id]
            
            if inds_near.size == 0 :
                raise ValueError("Can't find neighbours for the particle, try to change params!")
            
            pm = self.__eval_penalties__(next_src_pt_id, inds_near, pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)
            dest_id = -1
            
            while(dest_id == -1) :
                penalty = min(pm)
                min_idx = inds_near[np.argmin(pm)]

                if (linked_dest_pts[min_idx] == -1) :
                    dest_id = min_idx
                    break
                else :
                    bad_linked_src_id = linked_dest_pts[min_idx]
                    
                    if (linked_source_pts[bad_linked_src_id].error <= penalty) :
                        np.delete(inds_near, np.argmin(pm))
                        del pm[np.argmin(pm)]
                        if (len(pm) == 0) :
                            dest_id = -2
                            break
                    else :
                        dest_id = min_idx
                        stack_id = next((n for n in range(0, len(source_pts_stack)) if source_pts_stack[n] == bad_linked_src_id), -1)
                        linked_source_pts[bad_linked_src_id] = self.LinkInfo()
                        errors[bad_linked_src_id] = -1
                        del source_pts_stack[stack_id]

            if (self.__is_big_error__(errors, penalty)) :
                dest_id = -2

            if (dest_id != -2) :
                linked_dest_pts[dest_id] = next_src_pt_id
                errors[next_src_pt_id] = penalty
            else :
                errors[next_src_pt_id] = -1

            linked_source_pts[next_src_pt_id] = self.LinkInfo(dest_id, penalty)

            cur_stack_p = len(source_pts_stack)
            source_pts_stack.append(next_src_pt_id)


        trace = [pts1[src_id, :2] for src_id in source_pts_stack]

        links_arr = np.array([info.id for info in linked_source_pts])
        return np.column_stack((np.arange(len(linked_source_pts)), links_arr)), trace
    
    def __is_big_error__(self, errors, error, num_sigma = 3.0, min_errors_to_consider = 10) :
        if (error < 0) :
            return True

        valid_errors = errors[errors > 0]
        if len(valid_errors) < min_errors_to_consider:
            return False

        mean, std = valid_errors.mean(), valid_errors.std()
        return abs(error - mean) > num_sigma * std
    
    def __get_reasonable_predictors__(self, predictors_infos) :
        if (len(predictors_infos) == 0) :
            raise ValueError("Predictors list is emply!")
        
        predictors = sorted(predictors_infos, key=lambda x: x.error)[:self.predictors_consider]
        return predictors[1:-1] if len(predictors) >= 3 else predictors[:1]
    
    def __remove_outliers__(self, errors, num_sigma = 3) :
        mean, std = errors.mean(), errors.std()

        outlier_indices = np.where(np.abs(errors - mean) > num_sigma * std)[0]
        good_errors = errors
        good_errors[outlier_indices] = 0

        return good_errors
    
    # L2 error
    def __eval_penalty__(self, src_id, dst_id, pts1, near_pts1, pts2, near_pts2):
        ri = pts1[near_pts1[src_id]] - pts1[src_id]
        rj = pts2[near_pts2[dst_id]] - pts2[dst_id]
        dij = np.sum(ri**2, axis=1)[:, None] + np.sum(rj**2, axis=1) - 2 * ri.dot(rj.T)
        errors = np.sqrt(np.partition(dij.min(axis=1), self.n_use)[:self.n_use])
        good_errors = self.__remove_outliers__(errors)
        return good_errors.sum()
    
    def __eval_penalties__(self, src_id, inds_near, pts1, near_pts1, pts2, near_pts2):
        if len(inds_near) == 0 :
            raise ValueError("Indices array is empty!")
        return [self.__eval_penalty__(src_id, j, pts1, near_pts1, pts2, near_pts2) for j in inds_near]
    
    def __get_near_inds__(self, coord, pts, sample_start_point=False) :
        if (sample_start_point) :
            coef = self.sample_search_range_coef
        else :
            coef = 1.0

        maxdisp   = coef * self.maxdisp
        maxdisp_x = coef * self.maxdisp_x
        maxdisp_y = coef * self.maxdisp_y
        maxdisp_z = coef * self.maxdisp_z

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


    def __sample_start_point__(self, pts1, near_pts1, pts2, near_pts2) :
        if self.init_prediction_method == 'UseProvided' and self.first_ids:
            src, dst = self.first_ids
            error = self.__eval_penalty__(src, dst, pts1, near_pts1, pts2, near_pts2)
            return src, dst, error

        # Random Sampling
        tries = int(self.sample_ratio * pts1.shape[0])
        random.seed(0)
        sample_candidates = random.sample(range(pts1.shape[0]), tries)
        errors = []
        dest_ids = []
        for i in sample_candidates:
            inds_near = self.__get_near_inds__(pts1[i], pts2, sample_start_point=True)
            if inds_near.size:
                penalties = self.__eval_penalties__(i, inds_near, pts1, near_pts1, pts2, near_pts2)
                dest_ids.append(inds_near[penalties.argmin()] if penalties.size else -1)
                errors.append(penalties.min() if penalties.size else float("inf"))
            else:
                dest_ids.append(-1)
                errors.append(float("inf"))

        errors = np.array(errors)
        min_idx = errors.argmin()
        return sample_candidates[min_idx], dest_ids[min_idx], errors[min_idx]
        
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

def link_particles_and_compare(static_csv, deformed_csv, sample_random = False):
    # Load the static and deformed particle data from CSV files
    static_data = pd.read_csv(static_csv)
    deformed_data = pd.read_csv(deformed_csv)

    # Add time column (0 for static, 1 for deformed)
    static_data['time'] = 0
    deformed_data['time'] = 1

    # Concatenate static and deformed data
    combined_data = pd.concat([static_data, deformed_data], ignore_index=True)

    # TODO: GUI choose of start src_id
    if not sample_random :
        method = 'UseProvided'
        first_ids = choose_start_GUI(static_data, deformed_data)
    else :
        method = 'SampleRandom'
        first_ids = []

    # Track particles using track_raft function
    solver = ReliabilityRAFTSolver(3, method, maxdisp=15, sample_ratio=0.03, sample_search_range_coef=2.5, first_ids=first_ids, \
                                   n_consider=16, n_use=10)

    tracked_data, trace = solver.track_reliability_RAFT(combined_data)
    save_track_path_to_image(trace, 'trace.png')


    linked_static = tracked_data[tracked_data['time'] == 0][['particle', 'id']].rename(columns={'id': 'id_static'})
    linked_deformed = tracked_data[tracked_data['time'] == 1][['particle', 'id']].rename(columns={'id': 'id_deformed'})

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
link_particles_and_compare('hard_real_particle_static.csv', 'hard_real_particle_deformed.csv')
# link_particles_and_compare('real_particle_static.csv', 'real_particle_deformed.csv')
# link_particles_and_compare('particle_data_static.csv', 'particle_data_deformed.csv')