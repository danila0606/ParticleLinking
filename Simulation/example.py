import os
import pandas as pd

from RAFTlink import REL_RAFT_link
from helper import *


if __name__ == "__main__":

    CSV_FOLDER = 'big_disp_data'
    
    COLUMN_NAMES = ['time', 'z', 'y', 'x']

    USE_PREDICTORS_FROM_GUI = False
    PREDICTORS_PATH = './'
    PREDICTORS_FILENAME = 'pred.xml'
    MY_PREDICTORS = []

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
        tmp_data[COLUMN_NAMES[0]] = i+1
        data.append(tmp_data)
    
    combined_data = pd.concat(data, ignore_index=True)

    if USE_PREDICTORS_FROM_GUI :
        MY_PREDICTORS = []
        for i in range(len(csv_file_names_list) - 1) :
            MY_PREDICTORS.append(choose_start_GUI(data[i], data[i + 1], COLUMN_NAMES[1:]))
    elif len(MY_PREDICTORS) == 0 :
        if PREDICTORS_PATH is not None and PREDICTORS_FILENAME is not None :
            MY_PREDICTORS = load_predictors_from_xml(PREDICTORS_PATH + PREDICTORS_FILENAME)
            print(MY_PREDICTORS)
        else :
            raise ValueError("You should provide predictors .xml file or use GUI to select particles!")
        
    if PREDICTORS_PATH is not None and PREDICTORS_FILENAME is not None :
        save_predictors_to_xml(MY_PREDICTORS, PREDICTORS_PATH + PREDICTORS_FILENAME)


    tracked_data = REL_RAFT_link(combined_data, maxdisp=MAX_DISP, \
                                 n_consider=N_CONSIDER, n_use=N_USE, \
                                 column_names=COLUMN_NAMES, \
                                 my_predictors=MY_PREDICTORS, \
                                 sample_ratio=SAMPLE_RATIO, sample_search_range_coef=SAMPLING_SEARCH_RADIUS_COEF, \
                                 error_f=ERROR_FUNCTION, sigma_threshold=SIGMA_THRESHOLD, \
                                 memory=MEMORY)

    if SAVE_TRACE :
        if not os.path.exists(TRACE_PATH):
            os.makedirs(TRACE_PATH)

        save_track_path_to_image(tracked_data, TRACE_PATH)

    if CHECK_LINKING_ACCURACY :
        check_linking_accuracy(tracked_data, static_time=0, dynamic_time=len(csv_file_names_list)-1)

    tracked_data.to_csv(LINKING_DATA_FILENAME, index=False)
