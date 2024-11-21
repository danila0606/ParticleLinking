import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# from tqdm import tqdm


class ReliabilityRAFTSolver :
    def __init__(self, dim, init_prediction_method, maxdisp, sample_ratio = 0.1, sample_search_range_coef = 3.5, first_ids = [], **kwargs) :
        if (init_prediction_method != "SampleRandom" and init_prediction_method != "UseProvided") :
            raise Exception("There is not prediction method with name " + init_prediction_method)
        
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
    
    class LinkInfo :
        def __init__(self, id, error, banned_dest_ids):
            self.id = id
            self.error = error
            self.banned_dest_ids = banned_dest_ids


    def __reliability_flow_tracker__(self, pts1, pts2):
        n_pts1 = pts1.shape[0]
        n_pts2 = pts2.shape[0]

        # Find the nearest n_consider neighbours for pts1 and pts2
        near_neighb_inds_pts1 = np.zeros((n_pts1, self.n_consider), dtype=int)
        near_neighb_inds_pts2 = np.zeros((n_pts2, self.n_consider), dtype=int)

        for i in range(n_pts1):
            dists = np.sum((pts1 - pts1[i])**2, axis=1)
            inds = np.argsort(dists)[:self.n_consider + 1]
            near_neighb_inds_pts1[i] = inds[inds != i][:self.n_consider]

        for i in range(n_pts2):
            dists = np.sum((pts2 - pts2[i])**2, axis=1)
            inds = np.argsort(dists)[:self.n_consider + 1]
            near_neighb_inds_pts2[i] = inds[inds != i][:self.n_consider]


        #Sampling
        start_id, dest_id, error = self.__sample_start_point__(pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)
        print("Sampling done! ", start_id, dest_id, error)

        source_pts_stack = []
        linked_source_pts = [self.LinkInfo(-1, float('inf'), []) for i in range(n_pts1)]
        linked_dest_pts = np.full(n_pts2, -1)
        errors = [-1] * n_pts1
        # source_id, error, banned_dest_ids
        if (dest_id == -1) :
            raise ValueError("Bad sampling, try to change params!")

        source_pts_stack.append(start_id)
        linked_source_pts[start_id] = self.LinkInfo(dest_id, error, [])
        errors[start_id] = error
        linked_dest_pts[dest_id] = start_id

        cur_stack_p = 0

        while (len(source_pts_stack) < n_pts1) :
            # print(len(source_pts_stack))
            last_linked_src_id = source_pts_stack[cur_stack_p]
            last_linked_pt_info = linked_source_pts[last_linked_src_id]
            if (last_linked_pt_info.id == -2) :
                cur_stack_p = cur_stack_p - 1
                if (cur_stack_p < 0) :
                    raise ValueError("Can't good predictor, try to change params!")
                continue

            last_pt_neighbours = near_neighb_inds_pts1[last_linked_src_id]

            next_src_pt_id = -1
            for neighbour_id in last_pt_neighbours :
                if (linked_source_pts[neighbour_id].id == -1) : # or (linked_source_pts[neighbour_id] == -2)
                    next_src_pt_id = neighbour_id
                    break

            # if (next_src_pt_id == -1) :
            #     raise ValueError("Can't find free neighbour for {last_linked_pt_info[0]}, try to change params!")

            # TODODODODODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo
            # Dont raise, because sometime generates far away predict particles which has no free neighbour particles, in this case 
            # take free particles, find close neigbour with dest_id != -2 or -1, take this prediction
            if (next_src_pt_id == -1) :
                cur_stack_p = cur_stack_p - 1
                if (cur_stack_p < 0) :
                    for i in range(0, len(linked_source_pts)) :
                        if (linked_source_pts[i].id == -1) :
                            next_src_pt_id = i
                            dists_tmp = np.sum((pts1 - pts1[next_src_pt_id])**2, axis=1)
                            inds_tmp = np.argsort(dists_tmp)
                            neighbours_free_pt = inds_tmp[inds_tmp != next_src_pt_id]
                            for neighbour in neighbours_free_pt :
                                if (linked_source_pts[neighbour].id >= 0) :
                                    if (dists_tmp[neighbour] > self.maxdisp * self.sample_search_range_coef) :
                                        break

                            break
                    if (next_src_pt_id == -1) : # Smth wrong!!!
                        raise ValueError("Can't find next free particle, try to change params!")
                else :
                    continue
            else :
                neighbours_free_pt = near_neighb_inds_pts1[next_src_pt_id]

            predictors_infos = []
            for neighbour in neighbours_free_pt :
                if (linked_source_pts[neighbour].id >= 0) :
                    predictors_infos.append(linked_source_pts[neighbour])
 

            if (len(predictors_infos) == 0) :
                cur_stack_p = cur_stack_p - 1
                if (cur_stack_p < 0) : # taking any closest predictor
                    dists_tmp = np.sum((pts1 - pts1[next_src_pt_id])**2, axis=1)
                    inds_tmp = np.argsort(dists_tmp)
                    predictor_ids = inds_tmp[inds_tmp != next_src_pt_id]
                    for neighbour in predictor_ids :
                        if (len(predictors_infos) > self.predictors_consider) :
                            break
                        if (linked_source_pts[neighbour].id >= 0) :
                            predictors_infos.append(linked_source_pts[neighbour])
                else :
                    continue
            
            # sort by disp, take the middle one!!!!!!
            # print("Before: ", len(predictors_infos))
            predictors_infos = self.__get_reasonable_predictors__(predictors_infos)
            # print("After: ", len(predictors_infos))

            prediction_uvz = np.zeros(self.dim)
            for p_info in predictors_infos :
                prediction_uvz = prediction_uvz + (pts2[p_info.id] - pts1[linked_dest_pts[p_info.id]])
            prediction_uvz = prediction_uvz / len(predictors_infos)
            
            # OLD WAY
            inds_near = self.__get_near_inds__(pts1[next_src_pt_id] + prediction_uvz, pts2) # add prediction
            # NEW WAY
            if (len(inds_near) == 0) :
                dists_tmp = np.sum((pts2 - (pts1[next_src_pt_id] + prediction_uvz))**2, axis=1)
                inds_near = np.argsort(dists_tmp)[:self.n_consider + 1]
                inds_near = inds_near[inds_near != i][:self.n_consider]
            if len(inds_near) > 0:
                pm = pm = self.__eval_penalties__(next_src_pt_id, inds_near, pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)

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
                        link_info = linked_source_pts[bad_linked_src_id]
                        
                        if (link_info.error <= penalty) :
                            np.delete(inds_near, np.argmin(pm))
                            del pm[np.argmin(pm)]
                            if (len(pm) == 0) :
                                dest_id = -2
                                break
                                # raise ValueError("Can't find good destination candidates for the particle, try to change params!")
                            continue
                        else :
                            dest_id = ind_nn2
                            stack_id = self.__find_src_in_stack__(source_pts_stack, bad_linked_src_id)
                            if (self.drop_stack) :
                                for k in range(stack_id, len(source_pts_stack)) :
                                    src_id_to_drop = source_pts_stack[k]
                                    linked_source_pts[src_id_to_drop] = self.LinkInfo(-1, float('inf'), [])
                                    errors[src_id_to_drop] = -1
                                    linked_dest_pts[linked_source_pts[src_id_to_drop].id] = -1

                                del source_pts_stack[stack_id : len(source_pts_stack)]
                            else :
                                linked_source_pts[bad_linked_src_id] = self.LinkInfo(-1, float('inf'), [])
                                errors[bad_linked_src_id] = -1
                                del source_pts_stack[stack_id]

                if (self.__is_big_error__(errors, penalty)) :
                    dest_id = -2

                if (dest_id != -2) :
                    linked_dest_pts[dest_id] = next_src_pt_id
                    errors[next_src_pt_id] = penalty
                else :
                    errors[next_src_pt_id] = -1

                linked_source_pts[next_src_pt_id] = self.LinkInfo(dest_id, penalty, [])

                cur_stack_p = len(source_pts_stack)
                source_pts_stack.append(next_src_pt_id)

            else :
                raise ValueError("Can't find neighbours for the particle, try to change params!")

        trace = []
        for src_id in source_pts_stack :
            trace.append(np.array(pts1[src_id][:2]))

        links_arr = np.array([info.id for info in linked_source_pts])
        return np.column_stack((np.arange(len(linked_source_pts)), links_arr)), trace
    
    def __find_src_in_stack__(self, infos_stack, src_id) :
        for k in range(0, len(infos_stack)) :
            if infos_stack[k] == src_id :
                return k
            
        return -1
    
    def __is_big_error__(self, errors, error, num_sigma = 4.0, min_errors_to_consider = 10) :
        if (error < 0) :
            return True

        filtered_errors = [e for e in errors if e > 0]

        if (len(filtered_errors) < min_errors_to_consider) :
            return False

        mean_error = np.mean(filtered_errors)
        std_error = np.std(filtered_errors)

        if (abs(error - mean_error) > num_sigma * std_error) :
            return True
        
        return False
    
    def __get_reasonable_predictors__(self, predictors_infos) :
        if (len(predictors_infos) == 0) :
            raise ValueError("Predictors list is emply!")
        
        predictors_infos = sorted(predictors_infos, key=lambda x: x.error)[:self.predictors_consider]
        if (len(predictors_infos) < 3) :
            return [predictors_infos[0]]
        
        # return predictors_infos[0:3]
        return predictors_infos[1:-1]
    
    def __remove_outliers__(self, errors, num_sigma = 3) :
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        threshold = mean_error + num_sigma * std_error

        outlier_indices = np.where(np.abs(errors - mean_error) > threshold)[0]
        good_errors = errors
        good_errors[outlier_indices] = 0

        return good_errors
    
    def __eval_penalties__(self, src_id, inds_near, pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2):
        if len(inds_near) == 0 :
            raise ValueError("Indices array is empty!")
        
        pm = []
        for j in inds_near:
            # Relative positions of nearest neighbours
            ri = pts1[near_neighb_inds_pts1[src_id]] - pts1[src_id]
            rj = pts2[near_neighb_inds_pts2[j]] - pts2[j]
            # Calculate the squared distance matrix for relative particle points
            dij = np.sum(ri**2, axis=1)[:, None] + np.sum(rj**2, axis=1) - 2 * np.dot(ri, rj.T)
            # Cost is the sum of distances between n_use points
            errors = np.sqrt(np.partition(np.min(dij, axis=1), self.n_use)[:self.n_use])
            good_errors = self.__remove_outliers__(errors)
            pm.append(np.sum(good_errors))
        
        return pm
    
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


    def __sample_start_point__(self, pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2) :
        n_pts1 = pts1.shape[0]
        n_pts2 = pts2.shape[0]

        if (self.init_prediction_method == "SampleRandom") :
            tries = int(self.sample_ratio * n_pts1)
            random.seed(0)
            sample_candidates = random.sample(range(0, n_pts1), tries)
        else :
            sample_candidates = self.first_ids

        errors = []
        dest_ids = []
        for i in sample_candidates:
            inds_near = self.__get_near_inds__(pts1[i], pts2, sample_start_point=True)

            if len(inds_near) > 0:
                pm = self.__eval_penalties__(i, inds_near, pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)

                # Find the minimum penalty and corresponding point in pts2
                penalty = min(pm)
                dest_ids.append(inds_near[np.argmin(pm)])
                errors.append(penalty)
            else :
                errors.append(float("inf"))
                dest_ids.append(-1)

        errors = np.array(errors)
        return sample_candidates[np.argmin(errors)], dest_ids[np.argmin(errors)], np.min(errors)
        
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


def link_particles_and_compare(static_csv, deformed_csv):
    # Load the static and deformed particle data from CSV files
    static_data = pd.read_csv(static_csv)
    deformed_data = pd.read_csv(deformed_csv)

    # Add time column (0 for static, 1 for deformed)
    static_data['time'] = 0
    deformed_data['time'] = 1

    # Concatenate static and deformed data
    combined_data = pd.concat([static_data, deformed_data], ignore_index=True)

    # Track particles using track_raft function
    solver = ReliabilityRAFTSolver(3, "SampleRandom", maxdisp=15, sample_ratio=0.03, sample_search_range_coef=2.5, first_ids=[1, 3, 5, 8, 15], \
                                   n_consider=16, n_use=10)

    tracked_data, trace = solver.track_reliability_RAFT(combined_data)
    # save_track_path_to_image(trace, 'trace.png')


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
link_particles_and_compare('real_particle_static.csv', 'real_particle_deformed.csv')
# link_particles_and_compare('particle_data_static.csv', 'particle_data_deformed.csv')
