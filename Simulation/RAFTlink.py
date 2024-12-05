import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import functools
import os
import re

CSV_FOLDER = 'particle_data_iterations'
RANDOM_SAMPLE = False

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

    def __insert_id__(self, xyzti, time, src_id, id) :
        slice_condition = xyzti[:, 3] == time
        sliced_data = xyzti[slice_condition]
        sliced_data[src_id, 4] = id
        xyzti[slice_condition] = sliced_data

    def __add_tracks__(self, xyzti, res, static_id, id, static_time, times) :
        self.__insert_id__(xyzti, static_time, static_id, id)

        prev_id = static_id
        for t in range (static_time, len(times) - 1) :
            if t == 0 : 
                cur_id = static_id
            else :
                cur_id = res[t - 1][prev_id]
                res[t - 1][prev_id] = -1 # avoid duplicating
            
            if res[t][cur_id] < 0:
                return

            self.__insert_id__(xyzti, t + 1, res[t][cur_id], id)
            prev_id = cur_id

        res[t][prev_id] = -1

    def track_reliability_RAFT(self, id_xyzt):
        # Set optional parameters from kwargs
        xyzt = id_xyzt[['x', 'y', 'z', 'time']].to_numpy()

        # Extract unique time points and trackable time indices
        times = xyzt[:, -1]
        unique_times = np.unique(times)
        trackable_times = unique_times[np.where(np.diff(unique_times) == 1)[0]]

        if len(trackable_times) == 0:
            raise ValueError("No consecutive time points found for tracking")

        res = []
        for time in range(len(unique_times) - 1) :
            pts1 = xyzt[times == unique_times[time], :self.dim]
            pts2 = xyzt[times == unique_times[time + 1], :self.dim]
            res.append(self.__reliability_flow_tracker__(pts1, pts2, time))
        
        # print(res)
        xyzti = np.column_stack((xyzt, np.full((xyzt.shape[0], 1), -1)))
        id = 0
        # print(xyzti)
        # print("after")
        # self.__insert_id__(xyzti, 1, 4, 5)
        # print(xyzti)

        for time in range(len(unique_times) - 1) :
            cur_link = res[time]
            for i in range (cur_link.shape[0]) :
                if cur_link[i] >= 0 :
                    self.__add_tracks__(xyzti, res, i, id, time, unique_times)
                    id += 1

        df = pd.DataFrame(xyzti, columns=['x', 'y', 'z', 'time', 'particle'])
        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)
        df['z'] = df['z'].astype(float)
        df['time'] = df['time'].astype(int)
        df['particle'] = df['particle'].astype(int)

        return pd.concat([id_xyzt[['id']], df], axis = 1)

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

    def __reliability_flow_tracker__(self, pts1, pts2, time):
        n_pts1, n_pts2 = pts1.shape[0], pts2.shape[0]

        # Find the nearest n_consider neighbours for pts1 and pts2
        near_neighb_inds_pts1 = self.__find_nearest_neighbors__(pts1, self.n_consider)
        near_neighb_inds_pts2 = self.__find_nearest_neighbors__(pts2, self.n_consider)

        #Sampling
        start_id, dest_id, error = self.__sample_start_point__(pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2, time)
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

        links_arr = np.array([info.id for info in linked_source_pts])
        return links_arr
        # return np.column_stack((np.arange(len(linked_source_pts)), links_arr))
    
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


    def __sample_start_point__(self, pts1, near_pts1, pts2, near_pts2, time) :
        if self.init_prediction_method == 'UseProvided' and self.first_ids:
            src, dst = self.first_ids[time]
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

def link_particles_and_compare(csv_file_names_list):

    data = []
    for i in range(len(csv_file_names_list)) :
        tmp_data = pd.read_csv(CSV_FOLDER + '/' + csv_file_names_list[i])
        tmp_data['time'] = i
        data.append(tmp_data)
    
    combined_data = pd.concat(data, ignore_index=True)

    first_ids = []
    if not RANDOM_SAMPLE :
        method = 'UseProvided'
        for i in range(len(csv_file_names_list) - 1) :
            first_ids.append(choose_start_GUI(data[i], data[i + 1]))

        solver = ReliabilityRAFTSolver(3, method, maxdisp=15, first_ids=first_ids, n_consider=10, n_use=8)
    else :
        method = 'SampleRandom'
        solver = ReliabilityRAFTSolver(3, method, maxdisp=15, sample_ratio=0.03, sample_search_range_coef=2.5, n_consider=16, n_use=10)

    tracked_data = solver.track_reliability_RAFT(combined_data)
    # print(tracked_data)

    links = []
    for t in range(len(csv_file_names_list)) :
        links.append(tracked_data[tracked_data['time'] == t][['particle', 'id']].rename(columns={'id': f'id_{t}'}))

    # Merge particles on 'particle' to get pairs of linked IDs
    linked_comparison = functools.reduce(lambda  left,right: pd.merge(left, right, on=['particle']), links)

    linked_comparison['correct_link'] = (linked_comparison.nunique(axis=1) == 1).astype(int)
    # Calculate the number of correctly linked particles
    correct_links = linked_comparison['correct_link'].sum()
    total_links = len(linked_comparison)

    # Calculate accuracy
    accuracy = correct_links / total_links * 100

    # Output the results
    print(f"Total tracks: {total_links}")
    print(f"Correctly found tracks: {correct_links}")
    print(f"Linking accuracy: {accuracy:.2f}%")

    # Optionally save the comparison results to a CSV for analysis
    tracked_data.to_csv('linked_data.csv', index=False)


def get_csv_filenames(folder) :
    num_pattern = re.compile(r'\d+')
    filenames = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    sorted_filenames = sorted(
        filenames,
        key=lambda x: int(num_pattern.search(x).group()) if num_pattern.search(x) else float('inf')
    )
    
    return sorted_filenames


link_particles_and_compare(get_csv_filenames(CSV_FOLDER))
# link_particles_and_compare('real_particle_static.csv', 'real_particle_deformed.csv')
# link_particles_and_compare('particle_data_static.csv', 'particle_data_deformed.csv')
