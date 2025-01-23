import numpy as np
import pandas as pd
import random

NOT_TOUCHED = -1
NOT_LINKED = -2

class ReliabilityRAFTSolver :
    class LinkInfo:
        def __init__(self, id=NOT_TOUCHED, error=float('inf')):
            self.id = id
            self.error = error

    class ResultLinks:
        def __init__ (self, time_static, time_deformed, pts1, pts2, links, errors = None, trace = None):
            self.time_static   = time_static
            self.time_deformed = time_deformed
            self.pts1          = pts1
            self.pts2          = pts2
            self.links         = links
            self.errors        = errors
            self.trace         = trace

    def __init__(self, dim, maxdisp, \
                 sample_ratio = 0.1, sample_search_range_coef = 3.5, first_ids = None, my_predictors = None, \
                 error_f='L2', sigma_threshold = 3.0, \
                 memory=1, **kwargs) :
        
        if error_f not in ['L2', 'STRAIN']:
            raise ValueError(f"Invalid prediction method: {error_f}")
        
        self.error_f = error_f
        self.sigma_threshold = sigma_threshold

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
        self.my_predictors = my_predictors

        self.predictors_consider = 5

        self.memory = memory


    def __get_links_with_time__(self, all_links, time) :
        links = []
        for link in all_links :
            if (link.time_static == time) :
                links.append(link)

        links = sorted(links, key=lambda x: x.time_deformed)
        return links

    def track_reliability_RAFT(self, data_frame):
        # Set optional parameters from kwargs
        xyzt = data_frame[['x', 'y', 'z', 'time']].to_numpy()

        # Extract unique time points and trackable time indices
        times = xyzt[:, NOT_TOUCHED]
        unique_times = np.unique(times)
        trackable_times = unique_times[np.where(np.diff(unique_times) == 1)[0]]

        if len(trackable_times) == 0:
            raise ValueError("No consecutive time points found for tracking")

        res = []
        for time in range(len(unique_times) - 1) :
            pts1 = xyzt[times == unique_times[time], :self.dim]
            pts2 = xyzt[times == unique_times[time + 1], :self.dim]
            res = self.__reliability_flow_tracker__(pts1, pts2, time, res)
        
        xyztiet = np.column_stack((xyzt, np.full((xyzt.shape[0], 1), NOT_TOUCHED), np.full((xyzt.shape[0], 1), NOT_TOUCHED), np.full((xyzt.shape[0], 1), NOT_TOUCHED)))

        id = 0
        for time in range(len(unique_times) - 1) :
            static_condition = xyztiet[:, 3] == time
            static_data = xyztiet[static_condition]
            links = self.__get_links_with_time__(res, time)
            for cur_link in links :
                deformed_condition = xyztiet[:, 3] == cur_link.time_deformed
                deformed_data = xyztiet[deformed_condition]
                for src_id in range(0, len(cur_link.links)) :
                    if (cur_link.links[src_id] == NOT_LINKED) :
                        continue
                    
                    if static_data[src_id, 4] == NOT_TOUCHED :
                        static_data[src_id, 4] = id
                        id += 1

                    deformed_data[cur_link.links[src_id], 4] = static_data[src_id, 4]

                xyztiet[deformed_condition] = deformed_data
                
                if (cur_link.time_deformed == time + 1) :
                    static_data[:, 5] = cur_link.errors
                    order_id = 0
                    for order in cur_link.trace :
                        if cur_link.links[order] > 0 :
                            static_data[order, 6] = order_id
                            order_id += 1
            
            xyztiet[static_condition] = static_data

        df = data_frame.copy()
        df['particle'] = xyztiet[:, 4].astype(int)
        df['errors'] = xyztiet[:, 5]
        df['trace'] = xyztiet[:, 6].astype(int)

        return df

    def __find_nearest_neighbors__(self, pts, n_neighbors):
        dists = np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.argsort(dists, axis=1)[:, :n_neighbors]
    
    def __find_unlinked_source__(self, linked_source, pts1):
        for i, link in enumerate(linked_source):
            if link.id == NOT_TOUCHED:
                dists = np.sum((pts1 - pts1[i])**2, axis=1)
                neighbors = np.argsort(dists)
                for neighbor in neighbors:
                    if linked_source[neighbor].id >= 0 and dists[neighbor] > self.maxdisp * self.sample_search_range_coef:
                        return i
        return NOT_TOUCHED
    
    def __find_close_predictors__(self, linked_source, pts1, src_id) :
        inds_tmp = np.argsort(np.sum((pts1 - pts1[src_id])**2, axis=1))
        predictor_ids = inds_tmp[inds_tmp != src_id]
        predictors_infos = []
        for neighbour in predictor_ids :
            if (len(predictors_infos) > self.predictors_consider) :
                break
            if (linked_source[neighbour].id >= 0) :
                predictors_infos.append(linked_source[neighbour])

        return predictors_infos

    def __reliability_flow_tracker__(self, pts1, pts2, time, prev_results):
        n_pts1, n_pts2 = pts1.shape[0], pts2.shape[0]

        # Find the nearest n_consider neighbours for pts1 and pts2
        near_neighb_inds_pts1 = self.__find_nearest_neighbors__(pts1, self.n_consider)
        near_neighb_inds_pts2 = self.__find_nearest_neighbors__(pts2, self.n_consider)

        #Sampling
        start_id, dest_id, error = self.__sample_start_point__(pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2, time)
        if dest_id == NOT_TOUCHED:
            raise ValueError("Bad sampling, try to change params!")

        linked_source_pts = [self.LinkInfo() for _ in range(n_pts1)]
        linked_dest_pts = np.full(n_pts2, NOT_TOUCHED, dtype=int)
        errors = np.full(n_pts1, -1.0)

        linked_source_pts[start_id] = self.LinkInfo(dest_id, error)
        errors[start_id] = error
        linked_dest_pts[dest_id] = start_id

        trace = [start_id]
        cur_stack_p = 0

        while len(trace) < n_pts1 :
            src_id = trace[cur_stack_p]
            last_linked_pt_info = linked_source_pts[src_id]
            if (last_linked_pt_info.id == NOT_LINKED) :
                cur_stack_p -= 1
                if (cur_stack_p < 0) :
                    raise ValueError("Can't good predictor, try to change params!")
                continue

            last_pt_neighbours = near_neighb_inds_pts1[src_id]
            next_src_pt_id = next((n for n in last_pt_neighbours if linked_source_pts[n].id == NOT_TOUCHED), -1)

            if next_src_pt_id == NOT_TOUCHED :
                cur_stack_p -= 1
                if cur_stack_p < 0 :
                    next_src_pt_id = self.__find_unlinked_source__(linked_source_pts, pts1)
                    if (next_src_pt_id == NOT_TOUCHED) : # Smth wrong!!!
                        raise ValueError("Can't find next free particle, try to change params!")
                else :
                    continue
            
            predictors_infos = [linked_source_pts[n] for n in near_neighb_inds_pts1[next_src_pt_id] if linked_source_pts[n].id >= 0]

            if (len(predictors_infos) == 0) :
                cur_stack_p -= 1
                if (cur_stack_p < 0) : # taking any closest predictor
                    predictors_infos = self.__find_close_predictors__(linked_source_pts, pts1, next_src_pt_id)
                else :
                    continue
            
            # sort by disp, take the middle one
            predictors_infos = self.__get_reasonable_predictors__(predictors_infos)
            prediction = np.mean([pts2[p.id] - pts1[linked_dest_pts[p.id]] for p in predictors_infos], axis=0)
            
            inds_near = self.__get_near_inds__(pts1[next_src_pt_id] + prediction, pts2) # add prediction
            dest_id = NOT_TOUCHED

            if inds_near.size == 0 :
                dest_id = NOT_LINKED
                penalty = NOT_TOUCHED

            if inds_near.size != 0 :
                pm = self.__eval_penalties__(next_src_pt_id, inds_near, pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)
            
            while(dest_id == NOT_TOUCHED) :
                penalty = min(pm)
                min_idx = inds_near[np.argmin(pm)]

                if (linked_dest_pts[min_idx] == NOT_TOUCHED) :
                    dest_id = min_idx
                    break
                else :
                    bad_linked_src_id = linked_dest_pts[min_idx]
                    
                    if (linked_source_pts[bad_linked_src_id].error <= penalty) :
                        np.delete(inds_near, np.argmin(pm))
                        del pm[np.argmin(pm)]
                        if (len(pm) == 0) :
                            dest_id = NOT_LINKED
                            break
                    else :
                        dest_id = min_idx
                        stack_id = trace.index(bad_linked_src_id) if bad_linked_src_id in trace else NOT_TOUCHED
                        if stack_id == NOT_TOUCHED :
                            raise ValueError("Can't find previous wrong link!")
                        linked_source_pts[bad_linked_src_id] = self.LinkInfo()
                        errors[bad_linked_src_id] = NOT_TOUCHED
                        linked_dest_pts[dest_id] = NOT_TOUCHED
                        del trace[stack_id]

            if (self.__is_big_error__(errors, penalty)) :
                dest_id = NOT_LINKED

            if (dest_id != NOT_LINKED) :
                linked_dest_pts[dest_id] = next_src_pt_id
                errors[next_src_pt_id] = penalty
            else :
                errors[next_src_pt_id] = NOT_TOUCHED

            linked_source_pts[next_src_pt_id] = self.LinkInfo(dest_id, penalty)

            cur_stack_p = len(trace)
            trace.append(next_src_pt_id)

        links_arr = [info.id for info in linked_source_pts]
        
        all_links = []
        all_links.append(self.ResultLinks(time, time+1, pts1, pts2, links_arr, errors, trace))
        all_links.extend(prev_results)

        if (self.memory <= 0 or time == 0) :
            return all_links
        
        # Memory
        for i in range (1, self.memory + 1) :
            new_src_links = self.ResultLinks(-1, -1, None, None, [])
            for result_links in prev_results :
                if (result_links.time_static == time - i) :
                    new_src_links = result_links

            if (new_src_links.time_static == -1) :
                raise ValueError("Can't find link with time ", time - i, " !")

            not_linked_src = [i for i, x in enumerate(new_src_links.links) if x == NOT_LINKED]

            if len(not_linked_src) == 0 :
                continue

            near_neighb_inds_pts1 = self.__find_nearest_neighbors__(new_src_links.pts1, self.n_consider) # could be saved somewhere

            result_link = self.ResultLinks(new_src_links.time_static, time+1, new_src_links.pts1, pts2, [NOT_TOUCHED] * new_src_links.pts1.shape[0])
            # iterate through not linked pts
            for next_src_pt_id in not_linked_src :
                neighbours = [n for n in near_neighb_inds_pts1[next_src_pt_id] if new_src_links.links[n] >= 0]
                prediction = self.__get_memory_prediction__(neighbours, new_src_links.time_static, time+1, all_links)
                if prediction is None :
                    continue

                inds_near = self.__get_near_inds__(new_src_links.pts1[next_src_pt_id] + prediction, pts2) # add prediction
                if (len(inds_near) == 0) :
                    continue
                
                pm = self.__eval_penalties__(next_src_pt_id, inds_near, new_src_links.pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)
                dest_id = NOT_TOUCHED

                while(dest_id == NOT_TOUCHED) :
                    penalty = min(pm)
                    min_idx = inds_near[np.argmin(pm)]

                    if (linked_dest_pts[min_idx] == NOT_TOUCHED) :
                        dest_id = min_idx
                        break
                    else :
                        np.delete(inds_near, np.argmin(pm))
                        del pm[np.argmin(pm)]
                        if (len(pm) == 0) :
                            dest_id = NOT_LINKED
                            break

                if (self.__is_big_error__(errors, penalty)) :
                    dest_id = NOT_LINKED

                if (dest_id != NOT_LINKED) :
                    linked_dest_pts[dest_id] = next_src_pt_id

                result_link.links[next_src_pt_id] = dest_id

            all_links.append(result_link)
            
        return all_links
    
    def __get_memory_prediction__(self, start_inds, time_static, time_deformed, all_links) :
        prediction = np.zeros(3)
        cur_inds = start_inds

        for time in range (time_static, time_deformed) :
            cur_links = self.ResultLinks(-1, -1, None, None, [])
            for links in all_links :
                if (links.time_static == time and links.time_deformed == time + 1) :
                    cur_links = links

            if (cur_links.time_static == -1) :
                raise ValueError("Can't find link with static time ", time, " and deformed time ", time_deformed, " !")
            
            if len(cur_inds) == 0 :
                return None
        
            prediction += np.mean([cur_links.pts2[cur_links.links[p]] - cur_links.pts1[p] for p in cur_inds], axis=0)
            cur_inds = [cur_links.links[p] for p in cur_inds if cur_links.links[p] >= 0]
        
        return prediction
    
    def __is_big_error__(self, errors, error, min_errors_to_consider = 10) :
        if (error < 0) :
            return True

        valid_errors = errors[errors > 0]
        if len(valid_errors) < min_errors_to_consider:
            return False

        mean, std = valid_errors.mean(), valid_errors.std()
        return abs(error - mean) > self.sigma_threshold * std
    
    def __get_reasonable_predictors__(self, predictors_infos) :
        if (len(predictors_infos) == 0) :
            raise ValueError("Predictors list is emply!")
        
        predictors = sorted(predictors_infos, key=lambda x: x.error)[:self.predictors_consider]
        return predictors[1:-1] if len(predictors) >= 3 else predictors[:1]
    
    def __remove_outliers__(self, errors) :
        mean, std = errors.mean(), errors.std()

        outlier_indices = np.where(np.abs(errors - mean) > self.sigma_threshold * std)[0]
        good_errors = errors
        good_errors[outlier_indices] = 0

        return good_errors
    
    # L2 error
    def __eval_penalty__(self, src_id, dst_id, pts1, near_pts1, pts2, near_pts2):
        if self.error_f == 'L2' :
            ri = pts1[near_pts1[src_id]] - pts1[src_id]
            rj = pts2[near_pts2[dst_id]] - pts2[dst_id]
            dij = np.sum(ri**2, axis=1)[:, None] + np.sum(rj**2, axis=1) - 2 * ri.dot(rj.T)
            errors = np.sqrt(np.partition(dij.min(axis=1), self.n_use)[:self.n_use])
            good_errors = self.__remove_outliers__(errors)
            return good_errors.sum()
        else : # Strain
            ri = pts1[near_pts1[src_id]] - pts1[src_id]
            rj = pts2[near_pts2[dst_id]] - pts2[dst_id]
            diff = ri[:, np.newaxis, :] - rj[np.newaxis, :, :]
            deltas = np.linalg.norm(diff, axis=2)
            ri_lens = np.linalg.norm(ri, axis=1)
            dij = deltas / ri_lens[:, np.newaxis]
            errors = np.sqrt(np.partition(dij.min(axis=1), self.n_use)[:self.n_use])
            good_errors = self.__remove_outliers__(errors)
            return errors.sum()
    
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
        def is_valid_list(lst):
            return bool(lst) and any(sublist for sublist in lst)

        if (self.first_ids is not None) and is_valid_list(self.first_ids):
            src, dst = self.first_ids[time]
            error = self.__eval_penalty__(src, dst, pts1, near_pts1, pts2, near_pts2)
            return src, dst, error
        elif (self.my_predictors is not None) and is_valid_list(self.my_predictors):
            crd1, crd2 = self.my_predictors[time]
            dists1 = np.sum((pts1[:] - crd1)**2, axis=1)
            dists2 = np.sum((pts2[:] - crd2)**2, axis=1)
            src = np.argsort(dists1)[:1][0]
            dst = np.argsort(dists2)[:1][0]
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
                dest_ids.append(inds_near[np.array(penalties).argmin()] if (len(penalties) > 0) else NOT_TOUCHED)
                errors.append(min(penalties) if (len(penalties) > 0) else float("inf"))
            else:
                dest_ids.append(NOT_TOUCHED)
                errors.append(float("inf"))

        errors = np.array(errors)
        min_idx = errors.argmin()
        return sample_candidates[min_idx], dest_ids[min_idx], errors[min_idx]


def REL_RAFT_link(combined_data, dim, maxdisp, \
                 sample_ratio = 0.1, sample_search_range_coef = 3.5, 
                 first_ids = None, my_predictors = None, \
                 error_f='L2', sigma_threshold = 3.0, \
                 memory=1, 
                 **kwargs) :
    
    solver = ReliabilityRAFTSolver(dim, maxdisp=maxdisp, \
                                    sample_ratio=sample_ratio, sample_search_range_coef=sample_search_range_coef,
                                    first_ids=first_ids, my_predictors=my_predictors, \
                                    error_f=error_f, sigma_threshold=sigma_threshold, \
                                    memory=memory,
                                    **kwargs)
    
    return solver.track_reliability_RAFT(combined_data)
