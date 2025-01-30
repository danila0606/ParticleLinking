import numpy as np
import pandas as pd
import random

NOT_TOUCHED = -1
NOT_LINKED = -2
LINKED_WITH_MEMORY = -3

class ReliabilityRAFTSolver :
    """
    A class that links particles between frames using the "Reliability RAFT"

    Attributes
    ----------
    dim : int
        Considered dimension (2 or 3)
    maxdisp: float
        Maximum allowed offset
    n_consider : int
        The number of neighbors considered to evaluate the candidate's error
    n_use : int
        The number of neighbors used to calculate the candidate error (among the considered ones)
    column_names : list of strings
        The name of the columns in the particle dataset.
        The first string is the name of the time column, then the column names are patricle's coordinates in the order "z, y, x" for 3D or "y, x" for 2D
    first_ids : list[list[int]]
        A list of [src_id, dst_id] pairs in the current pair of frames used to initialize the predictor
        If the list is empty, uses 'my_predictors'
    my_predictors : list[list[numpy array of shape 'dim']]
        A list of [src coordinate, dst coordinate] pairs in the current pair of frames used to initialize the predictor
        If the list is empty, uses random sampling
    sample_ratio : float
        Percentage of particles considered in random sampling
    sample_search_range_coef : float
        Multiplied by 'maxdisp' when randomly sampling the first link for better search
    error_f : str
        The error function that will be used to evaluate the accuracy of linking
    sigma_threshold : float
        The limit beyond which the error is considered too large
    memory : int
        The number of frames considered in the past used to attempt to link the remaining particles in the current frame
    """
    class LinkInfo:
        """
        A linking information of the one src particle

        Attributes
        ----------
        id : int
            An id of dst particle
        error : float
            An error of the link
        """
        def __init__(self, id=NOT_TOUCHED, error=float(NOT_TOUCHED)):
            self.id = id
            self.error = error

    class ResultLinks:
        """
        The linking information of the one src particle

        Attributes
        ----------
        time_static : int
        time_deformed : int
        links : list of integers
            A list of size equal to the number of particles in 'time_static'
            The index in the list is src_id, the corresponding value is dst_id
        errors : list of floats
            A list of size equal to the number of particles in 'time_static'
            The index in the list is src_id, the corresponding value is the linking error
        trace : list of integers
            A list of size equal to the number of particles in 'time_static'
            The list contains the src_id's in the order in which they were linked
        """
        def __init__ (self, time_static, time_deformed, links, errors = None, trace = None):
            self.time_static   = time_static
            self.time_deformed = time_deformed
            self.links         = links
            self.errors        = errors
            self.trace         = trace

    def __init__(self, dim, maxdisp, \
                 n_consider=10, n_use=8, \
                 column_names = ['time', 'z', 'y', 'x'], \
                 sample_ratio = 0.1, sample_search_range_coef = 3.5, first_ids = None, my_predictors = None, \
                 error_f='L2', sigma_threshold = 3.0, \
                 memory=0, **kwargs) :
        
        self.dim = dim

        self.maxdisp = maxdisp
        self.maxdisp_z = kwargs.get('maxdisp_z', maxdisp)
        self.maxdisp_y = kwargs.get('maxdisp_y', maxdisp)
        self.maxdisp_x = kwargs.get('maxdisp_x', maxdisp)

        self.n_consider = n_consider
        self.n_use = n_use

        self.column_names = column_names

        self.sample_ratio = sample_ratio
        self.sample_search_range_coef = sample_search_range_coef
        self.first_ids = first_ids
        self.my_predictors = my_predictors

        if error_f not in ['L2', 'STRAIN']:
            raise ValueError(f"Invalid prediction method: {error_f}")
        self.error_f = error_f
        self.sigma_threshold = sigma_threshold

        self.memory = memory

        self.predictors_consider = 5
        self.min_errors_to_consider = 10
        self.times        = None
        self.unique_times = None

    def __get_links_with_time__(self, all_links, time) :
        links = []
        for link in all_links :
            if (link.time_static == time) :
                links.append(link)

        links = sorted(links, key=lambda x: x.time_deformed)
        return links

    def track_reliability_RAFT(self, data_frame):
        if self.dim == 2 :
            xyzt = data_frame[[self.column_names[1], self.column_names[2], self.column_names[0]]].to_numpy()
        elif self.dim == 3 :
            xyzt = data_frame[[self.column_names[1], self.column_names[2], self.column_names[3], self.column_names[0]]].to_numpy()
        else :
            raise ValueError("Unsupported dim: " + str(self.dim))

        # Extract unique time points
        self.times = xyzt[:, NOT_TOUCHED]
        self.unique_times = np.sort(np.unique(self.times)).astype(int)

        res = []
        for time_i in range(len(self.unique_times) - 1) :
            res = self.__reliability_flow_tracker__(xyzt, time_i, res)
        
        xyztiet = np.column_stack((xyzt, np.full((xyzt.shape[0], 1), NOT_TOUCHED), 
                                         np.full((xyzt.shape[0], 1), NOT_TOUCHED), 
                                         np.full((xyzt.shape[0], 1), NOT_TOUCHED)))

        id = 0
        time_column, particle_column, error_column, trace_column = self.dim, self.dim+1, self.dim+2, self.dim+3
        # Combine all links together
        for time_i in range(len(self.unique_times) - 1) :
            static_condition = xyztiet[:, time_column] == self.unique_times[time_i]
            static_data = xyztiet[static_condition]
            links = self.__get_links_with_time__(res, time_i)
            # Iterate through all links with current static time
            # If memory is zero, then there is only one
            for cur_link in links :
                deformed_condition = xyztiet[:, time_column] == self.unique_times[cur_link.time_deformed]
                deformed_data = xyztiet[deformed_condition]
                for src_id in range(0, len(cur_link.links)) :
                    if (cur_link.links[src_id] == NOT_LINKED or cur_link.links[src_id] == LINKED_WITH_MEMORY) :
                        continue
                    
                    if static_data[src_id, particle_column] == NOT_TOUCHED :
                        # This particle has no number yet
                        static_data[src_id, particle_column] = id
                        id += 1

                    # Assign the same number for the dst particle as for the src particle
                    deformed_data[cur_link.links[src_id], particle_column] = static_data[src_id, particle_column]

                xyztiet[deformed_condition] = deformed_data
                
                if (cur_link.time_deformed == time_i + 1) :
                    static_data[:, error_column] = cur_link.errors
                    order_id = 0
                    for order in cur_link.trace :
                        if cur_link.links[order] > 0 :
                            static_data[order, trace_column] = order_id
                            order_id += 1
            
            xyztiet[static_condition] = static_data

        df = data_frame.copy()
        df['particle'] = xyztiet[:, particle_column].astype(int)
        df['error']   = xyztiet[:, error_column]
        df['trace']    = xyztiet[:, trace_column].astype(int)

        return df

    def __find_nearest_neighbors__(self, pts):
        """
        Finds 'n_consider' ids of the nearest particles for each particle
        """
        dists = np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.argsort(dists, axis=1)[:, :self.n_consider]
    
    def __find_unlinked_source__(self, linked_source, pts1):
        """
        Finds an unlinked particle that has at least one not far away linked neighbor
        """
        for i, link in enumerate(linked_source):
            if link.id == NOT_TOUCHED:
                dists = np.sum((pts1 - pts1[i])**2, axis=1)
                neighbors = np.argsort(dists)
                for neighbor in neighbors:
                    if linked_source[neighbor].id >= 0 and dists[neighbor] > self.maxdisp * self.sample_search_range_coef:
                        return i
        return NOT_TOUCHED
    
    def __find_close_predictors__(self, linked_source, pts1, src_id) :
        """
        Looking for the nearest linked neighbors
        """
        inds_tmp = np.argsort(np.sum((pts1 - pts1[src_id])**2, axis=1))
        predictor_ids = inds_tmp[inds_tmp != src_id]
        predictors_infos = []
        for neighbour in predictor_ids :
            if (len(predictors_infos) > self.predictors_consider) :
                break
            if (linked_source[neighbour].id >= 0) :
                predictors_infos.append(linked_source[neighbour])

        return predictors_infos

    def __reliability_flow_tracker__(self, xyzt, time, prev_results):
        """
        Links particles from a frame[time] to particles from a frame[time+1]
        If memory is not zero and there are unlinked particles in the dst frame, it tries to find src particles in the previous frames
        """
        pts1 = xyzt[self.times == self.unique_times[time], :self.dim]
        pts2 = xyzt[self.times == self.unique_times[time + 1], :self.dim]
        n_pts1, n_pts2 = pts1.shape[0], pts2.shape[0]

        # Find the nearest n_consider neighbours for pts1 and pts2
        near_neighb_inds_pts1 = self.__find_nearest_neighbors__(pts1)
        near_neighb_inds_pts2 = self.__find_nearest_neighbors__(pts2)

        # Sampling first link
        start_id, dst_id, error = self.__sample_start_point__(pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2, time)
        if dst_id == NOT_TOUCHED:
            raise ValueError("Bad sampling, try to change params!")

        linked_src = [self.LinkInfo() for _ in range(n_pts1)] # index - src_id, value - (dst_id, error)
        linked_dst = np.full(n_pts2, NOT_TOUCHED, dtype=int)  # index - dst_id, value - src_id
        errors = np.full(n_pts1, -1.0) # index - src_id, value - error

        # Put the first linked pair
        linked_src[start_id] = self.LinkInfo(dst_id, error)
        errors[start_id] = error
        linked_dst[dst_id] = start_id

        trace = [start_id]
        cur_trace_ptr = 0

        # Stop when all particles were tried to be linked
        while len(trace) < n_pts1 :
            last_linked_src_id = trace[cur_trace_ptr]
            last_linked_pt_info = linked_src[last_linked_src_id]
            if (last_linked_pt_info.id == NOT_LINKED) :
                # The last particle wasn't linked, take previous one
                cur_trace_ptr -= 1
                if (cur_trace_ptr < 0) :
                    raise ValueError("Can't find good predictor to continue, try to change params!")
                continue

            last_linked_pt_neighbours = near_neighb_inds_pts1[last_linked_src_id]
            src_id = next((n for n in last_linked_pt_neighbours if linked_src[n].id == NOT_TOUCHED), -1)

            if src_id == NOT_TOUCHED :
                # All neighbours of the last linked particle are also linked, take previous one
                cur_trace_ptr -= 1
                if cur_trace_ptr < 0 :
                    # Couldn't find an unlinked src particle among the already linked particles, take any unlinked src particle
                    src_id = self.__find_unlinked_source__(linked_src, pts1)
                    if (src_id == NOT_TOUCHED) : # Smth wrong!!!
                        raise ValueError("Can't find next free particle, try to change params!")
                else :
                    continue
            
            # Consider all neighbour particles that are linked
            predictors_infos = [linked_src[n] for n in near_neighb_inds_pts1[src_id] if linked_src[n].id >= 0]

            if (len(predictors_infos) == 0) :
                # No neighbor for the selected src particle is linked, search another src_id
                cur_trace_ptr -= 1
                if (cur_trace_ptr < 0) : # take any closest predictor
                    predictors_infos = self.__find_close_predictors__(linked_src, pts1, src_id)
                else :
                    continue
            
            # Sort by error, take the middle one
            predictors_infos = self.__get_reasonable_predictors__(predictors_infos)
            prediction = np.mean([pts2[p.id] - pts1[linked_dst[p.id]] for p in predictors_infos], axis=0)
            
            # Find all the dst candidates in the predicted area of the maxdisp radius
            dst_candidates = self.__get_near_inds__(pts1[src_id] + prediction, pts2)
            dst_id = NOT_TOUCHED

            if dst_candidates.size == 0 :
                # Not a single candidate in the predicted field
                dst_id = NOT_LINKED
                penalty = NOT_LINKED

            if dst_candidates.size != 0 :
                # Evaluate errors for each possible candidate
                penalties = self.__eval_penalties__(src_id, dst_candidates, pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)
            
            while(dst_id == NOT_TOUCHED) :
                # Choose the candidate with the least error among the remaining
                penalty = min(penalties)
                min_dst_id = dst_candidates[np.argmin(penalties)]

                if (linked_dst[min_dst_id] == NOT_TOUCHED) :
                    # Candidate is not linked yet, so it can be chosen
                    dst_id = min_dst_id
                    break
                else :
                    # Candidate is already linked, error needs to be compared
                    bad_linked_src_id = linked_dst[min_dst_id]
                    
                    if (linked_src[bad_linked_src_id].error <= penalty) :
                        # Previous link is more accurate, continue to search next candidate
                        np.delete(dst_candidates, np.argmin(penalties))
                        del penalties[np.argmin(penalties)]
                        if (len(penalties) == 0) :
                            # There are no more candidates left, not possible to link this src_id
                            dst_id = NOT_LINKED
                    else :
                        # This candidate is more suitable for the current src_id than for the previous one, 
                        # so let's replace the previous link
                        dst_id = min_dst_id
                        bad_link_trace_id = trace.index(bad_linked_src_id) if bad_linked_src_id in trace else NOT_TOUCHED
                        if bad_link_trace_id == NOT_TOUCHED :
                            raise ValueError("Can't find previous wrong link!")
                        linked_src[bad_linked_src_id] = self.LinkInfo()
                        errors[bad_linked_src_id] = NOT_TOUCHED
                        linked_dst[dst_id] = NOT_TOUCHED
                        del trace[bad_link_trace_id]

            if (self.__is_big_error__(errors, penalty)) :
                # The error is too large, this particle cannot be linked.
                dst_id = NOT_LINKED

            if (dst_id != NOT_LINKED) :
                linked_dst[dst_id] = src_id
                errors[src_id] = penalty
            else :
                errors[src_id] = NOT_LINKED

            linked_src[src_id] = self.LinkInfo(dst_id, penalty)

            cur_trace_ptr = len(trace)
            trace.append(src_id)

        links_arr = [info.id for info in linked_src]
        
        all_links = []
        all_links.append(self.ResultLinks(time, time+1, links_arr, errors, trace))
        all_links.extend(prev_results)

        if (self.memory <= 0 or time == 0) :
            return all_links
        
        # Memory
        for i in range (1, self.memory + 1) :
            # Select previous link, which can be considered
            old_links = next(
                (prev_links for prev_links in prev_results 
                if prev_links.time_static == time - i and prev_links.time_deformed - prev_links.time_static == 1), 
                self.ResultLinks(-1, -1, [])
            )
            if old_links.time_static == -1:
                raise ValueError(f"Can't find link with static time {time - i} and deformed time: {time - i + 1}!")

            not_linked_src = [i for i, x in enumerate(old_links.links) if x == NOT_LINKED]

            if len(not_linked_src) == 0 :
                # All particles for this static time are linked
                continue
            
            pts1 = xyzt[self.times == self.unique_times[old_links.time_static], :self.dim]
            near_neighb_inds_pts1 = self.__find_nearest_neighbors__(pts1)

            result_links = self.ResultLinks(old_links.time_static, time+1, [NOT_TOUCHED] * pts1.shape[0])
            # iterate through not linked pts
            for src_id in not_linked_src :
                # Use linked neighbours to calculate prediction
                neighbours = [n for n in near_neighb_inds_pts1[src_id] if old_links.links[n] >= 0]
                prediction = self.__get_memory_prediction__(neighbours, old_links.time_static, time+1, all_links, xyzt)
                if prediction is None :
                    continue

                dst_candidates = self.__get_near_inds__(pts1[src_id] + prediction, pts2)
                if (len(dst_candidates) == 0) :
                    # Not a single candidate in the predicted field
                    continue
                
                penalties = self.__eval_penalties__(src_id, dst_candidates, pts1, near_neighb_inds_pts1, pts2, near_neighb_inds_pts2)
                dst_id = NOT_TOUCHED

                while(dst_id == NOT_TOUCHED) :
                    penalty = min(penalties)
                    min_idx = dst_candidates[np.argmin(penalties)]

                    if (linked_dst[min_idx] == NOT_TOUCHED) :
                        # Candidate is not linked yet, so it can be chosen
                        dst_id = min_idx
                        break
                    else :
                        # Candidate is already linked, continue to search
                        np.delete(dst_candidates, np.argmin(penalties))
                        del penalties[np.argmin(penalties)]
                        if (len(penalties) == 0) :
                            # There are no more candidates left, not possible to link this src_id
                            dst_id = NOT_LINKED
                            break

                if (self.__is_big_error__(errors, penalty)) :
                    dst_id = NOT_LINKED

                if (dst_id != NOT_LINKED) :
                    linked_dst[dst_id] = src_id
                    old_links.links[src_id] = LINKED_WITH_MEMORY

                result_links.links[src_id] = dst_id

            all_links.append(result_links)
            
        return all_links
    
    def __get_memory_prediction__(self, start_inds, time_static, time_deformed, all_links, xyzt) :
        """
        Calculates a prediction using particles that are linked across multiple frames (from time_static to time_deformed)
        """

        prediction = np.zeros(self.dim)
        cur_inds = start_inds

        for time in range (time_static, time_deformed) :
            cur_links = self.ResultLinks(-1, -1, [])
            for links in all_links :
                if (links.time_static == time and links.time_deformed == time + 1) :
                    cur_links = links

            if (cur_links.time_static == -1) :
                raise ValueError("Can't find link with static time ", time, " and deformed time ", time_deformed, " !")
            
            if len(cur_inds) == 0 :
                # Not possible to find at least one particle that is linked through all frames
                return None

            pts1 = xyzt[self.times == self.unique_times[cur_links.time_static  ], :self.dim]
            pts2 = xyzt[self.times == self.unique_times[cur_links.time_deformed], :self.dim]

            prediction += np.mean([pts2[cur_links.links[p]] - pts1[p] for p in cur_inds], axis=0)
            # Take linked subset in the next frame
            cur_inds = [cur_links.links[p] for p in cur_inds if cur_links.links[p] >= 0]
        
        return prediction
    
    def __is_big_error__(self, errors, error) :
        if (error < 0) :
            # Not linked
            return True

        valid_errors = errors[errors > 0]
        if len(valid_errors) < self.min_errors_to_consider:
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
    
    def __eval_penalty__(self, src_id, dst_id, pts1, near_pts1, pts2, near_pts2):
        if self.error_f == 'L2' : # L2 error
            ri = pts1[near_pts1[src_id]] - pts1[src_id]
            rj = pts2[near_pts2[dst_id]] - pts2[dst_id]
            dij = np.sum(ri**2, axis=1)[:, None] + np.sum(rj**2, axis=1) - 2 * ri.dot(rj.T)
            errors = np.sqrt(np.partition(dij.min(axis=1), self.n_use)[:self.n_use])
            # errors = self.__remove_outliers__(errors)
            return errors.sum()
        else : # Strain
            ri = pts1[near_pts1[src_id]] - pts1[src_id]
            rj = pts2[near_pts2[dst_id]] - pts2[dst_id]
            diff = ri[:, np.newaxis, :] - rj[np.newaxis, :, :]
            deltas = np.linalg.norm(diff, axis=2)
            ri_lens = np.linalg.norm(ri, axis=1)
            dij = deltas / ri_lens[:, np.newaxis]
            errors = np.sqrt(np.partition(dij.min(axis=1), min(self.n_use, dij.min(axis=1).shape[0] - 1))[:self.n_use])
            # errors = self.__remove_outliers__(errors)
            return errors.sum()
    
    def __eval_penalties__(self, src_id, dst_candidates, pts1, near_pts1, pts2, near_pts2):
        """ 
        Calculates the src particle linking errors with all dst candidates
        """

        if len(dst_candidates) == 0 :
            raise ValueError("Indices array is empty!")
        return [self.__eval_penalty__(src_id, j, pts1, near_pts1, pts2, near_pts2) for j in dst_candidates]
    
    def __get_near_inds__(self, coord, pts, sample_start_point=False) :
        """ 
        Searches for the id of all particles within the maxdisp radius of a given coordinate
        """

        if (sample_start_point) :
            # When randomly sample first link, we should search in a big area, because there is no prediction yet
            coef = self.sample_search_range_coef
        else :
            coef = 1.0

        maxdisp   = coef * self.maxdisp
        maxdisp_x = coef * self.maxdisp_x
        maxdisp_y = coef * self.maxdisp_y
        maxdisp_z = coef * self.maxdisp_z

        N = pts.shape[1]

        if N == 1:
            inds_near = (np.sum((pts - coord)**2, axis=1) < maxdisp**2) & ((pts[:, 0] - coord[0])**2 < maxdisp_z**2)
        elif N == 2:
            inds_near = (np.sum((pts - coord)**2, axis=1) < maxdisp**2) & \
                        ((pts[:, 0] - coord[0])**2 < maxdisp_y**2) & \
                        ((pts[:, 1] - coord[1])**2 < maxdisp_x**2)
        else:
            inds_near = (np.sum((pts - coord)**2, axis=1) < maxdisp**2) & \
                        ((pts[:, 0] - coord[0])**2 < maxdisp_z**2) & \
                        ((pts[:, 1] - coord[1])**2 < maxdisp_y**2) & \
                        ((pts[:, 2] - coord[2])**2 < maxdisp_x**2)
        
        return np.where(inds_near)[0]


    def __sample_start_point__(self, pts1, near_pts1, pts2, near_pts2, time) :
        """
        Selects the first linked pair for the selected pair of frames
        If the pair or disp is not provided by the user, then selects the best linked pair among the randomly selected ones
        """

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
            # Choose closest particles to the provided coordinates
            src = np.argsort(dists1)[:1][0]
            dst = np.argsort(dists2)[:1][0]
            error = self.__eval_penalty__(src, dst, pts1, near_pts1, pts2, near_pts2)
            return src, dst, error

        # Random Sampling
        tries = int(self.sample_ratio * pts1.shape[0])
        # random.seed(0)
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
                  n_consider=10, n_use=8, \
                  column_names = ['time', 'z', 'y', 'x'], \
                  sample_ratio=0.1, sample_search_range_coef=3.5, \
                  first_ids=None, my_predictors=None, \
                  error_f='L2', sigma_threshold=3.0, \
                  memory=0, \
                  **kwargs) :
    
    solver = ReliabilityRAFTSolver( dim, maxdisp=maxdisp, \
                                    n_consider=n_consider, n_use=n_use, \
                                    column_names=column_names, \
                                    sample_ratio=sample_ratio, sample_search_range_coef=sample_search_range_coef, \
                                    first_ids=first_ids, my_predictors=my_predictors, \
                                    error_f=error_f, sigma_threshold=sigma_threshold, \
                                    memory=memory, \
                                    **kwargs)
    
    return solver.track_reliability_RAFT(combined_data)
