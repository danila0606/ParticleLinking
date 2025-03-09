import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os

from RAFTlink import REL_RAFT_link
from helper import *

def save_settings():
    settings = {
        "CSV_FOLDER": entry_csv_folder.get(),
        "COLUMN_NAMES": entry_column_names.get(),
        "USE_PREDICTORS_FROM_GUI": use_predictors_from_gui.get(),
        "PREDICTORS_PATH": entry_predictors_path.get(),
        "PREDICTORS_FILENAME": entry_predictors_filename.get(),
        "SAMPLE_RATIO": entry_sample_ratio.get(),
        "SAMPLING_SEARCH_RADIUS_COEF": entry_sampling_search_radius_coef.get(),
        "ERROR_FUNCTION": error_function_combobox.get(),
        "SIGMA_THRESHOLD": entry_sigma_threshold.get(),
        "MAX_DISP": entry_max_disp.get(),
        "N_CONSIDER": entry_n_consider.get(),
        "N_USE": entry_n_use.get(),
        "SAVE_TRACE": save_trace.get(),
        "TRACE_PATH": entry_trace_path.get(),
        "CHECK_LINKING_ACCURACY": check_linking_accuracy.get(),
        "LINKING_DATA_FILENAME": entry_linking_data_filename.get(),
        "MEMORY": entry_memory.get(),
    }

    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if file_path:
        with open(file_path, "w") as f:
            json.dump(settings, f, indent=4)
        messagebox.showinfo("Success", "Settings saved successfully!")

def load_settings():
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        with open(file_path, "r") as f:
            settings = json.load(f)

        entry_csv_folder.delete(0, tk.END)
        entry_csv_folder.insert(0, settings.get("CSV_FOLDER", ""))

        entry_column_names.delete(0, tk.END)
        entry_column_names.insert(0, settings.get("COLUMN_NAMES", ""))

        use_predictors_from_gui.set(settings.get("USE_PREDICTORS_FROM_GUI", False))

        entry_predictors_path.delete(0, tk.END)
        entry_predictors_path.insert(0, settings.get("PREDICTORS_PATH", ""))

        entry_predictors_filename.delete(0, tk.END)
        entry_predictors_filename.insert(0, settings.get("PREDICTORS_FILENAME", ""))

        entry_sample_ratio.delete(0, tk.END)
        entry_sample_ratio.insert(0, settings.get("SAMPLE_RATIO", "0.03"))

        entry_sampling_search_radius_coef.delete(0, tk.END)
        entry_sampling_search_radius_coef.insert(0, settings.get("SAMPLING_SEARCH_RADIUS_COEF", "2.5"))

        error_function_combobox.set(settings.get("ERROR_FUNCTION", "STRAIN"))

        entry_sigma_threshold.delete(0, tk.END)
        entry_sigma_threshold.insert(0, settings.get("SIGMA_THRESHOLD", "3.0"))

        entry_max_disp.delete(0, tk.END)
        entry_max_disp.insert(0, settings.get("MAX_DISP", "20"))

        entry_n_consider.delete(0, tk.END)
        entry_n_consider.insert(0, settings.get("N_CONSIDER", "15"))

        entry_n_use.delete(0, tk.END)
        entry_n_use.insert(0, settings.get("N_USE", "10"))

        save_trace.set(settings.get("SAVE_TRACE", False))

        entry_trace_path.delete(0, tk.END)
        entry_trace_path.insert(0, settings.get("TRACE_PATH", ""))

        check_linking_accuracy.set(settings.get("CHECK_LINKING_ACCURACY", False))

        entry_linking_data_filename.delete(0, tk.END)
        entry_linking_data_filename.insert(0, settings.get("LINKING_DATA_FILENAME", "linked_data.csv"))

        entry_memory.delete(0, tk.END)
        entry_memory.insert(0, settings.get("MEMORY", "0"))

        messagebox.showinfo("Success", "Settings loaded successfully!")

def run_script():
    # Get the values from the GUI
    CSV_FOLDER = entry_csv_folder.get()
    COLUMN_NAMES = entry_column_names.get().split(',')
    USE_PREDICTORS_FROM_GUI = use_predictors_from_gui.get()
    PREDICTORS_PATH = entry_predictors_path.get() + '/'
    PREDICTORS_FILENAME = entry_predictors_filename.get()
    SAMPLE_RATIO = float(entry_sample_ratio.get())
    SAMPLING_SEARCH_RADIUS_COEF = float(entry_sampling_search_radius_coef.get())
    ERROR_FUNCTION = error_function_combobox.get()
    SIGMA_THRESHOLD = float(entry_sigma_threshold.get())
    MAX_DISP = int(entry_max_disp.get())
    N_CONSIDER = int(entry_n_consider.get())
    N_USE = int(entry_n_use.get())
    SAVE_TRACE = save_trace.get()
    TRACE_PATH = entry_trace_path.get()
    CHECK_LINKING_ACCURACY = check_linking_accuracy.get()
    LINKING_DATA_FILENAME = entry_linking_data_filename.get()
    MEMORY = int(entry_memory.get())

    # Run the script with the provided inputs
    try:
        csv_file_names_list = get_csv_filenames(CSV_FOLDER)

        data = []
        for i in range(len(csv_file_names_list)) :
            tmp_data = pd.read_csv(CSV_FOLDER + '/' + csv_file_names_list[i])
            tmp_data[COLUMN_NAMES[0]] = i+1
            data.append(tmp_data)
        
        combined_data = pd.concat(data, ignore_index=True)

        MY_PREDICTORS = []
        if USE_PREDICTORS_FROM_GUI :
            for i in range(len(csv_file_names_list) - 1) :
                MY_PREDICTORS.append(choose_start_GUI(data[i], data[i + 1], COLUMN_NAMES[1:]))
        else :
            if PREDICTORS_PATH is not None and PREDICTORS_FILENAME is not None :
                print("Loading predictors from ", PREDICTORS_PATH + "/" + PREDICTORS_FILENAME + " ...")
                MY_PREDICTORS = load_predictors_from_xml(PREDICTORS_PATH + PREDICTORS_FILENAME)
            else :
                raise ValueError("You should provide predictors .xml file or use GUI to select particles!")
            
        if PREDICTORS_PATH is not None and PREDICTORS_FILENAME is not None :
            print("Saving predictors to ", PREDICTORS_PATH + "/" + PREDICTORS_FILENAME + " ...")
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

        messagebox.showinfo("Success", "Script executed successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("RAFT Link")
root.geometry("800x500")  # Increased window size

# Create and place the input fields
tk.Label(root, text="CSV Folder:").grid(row=0, column=0, sticky="w")
entry_csv_folder = tk.Entry(root, width=50)
entry_csv_folder.grid(row=0, column=1)
tk.Button(root, text="Browse", command=lambda: entry_csv_folder.insert(0, filedialog.askdirectory())).grid(row=0, column=2)

tk.Label(root, text="Column Names (comma-separated):").grid(row=1, column=0, sticky="w")
entry_column_names = tk.Entry(root, width=50)
entry_column_names.grid(row=1, column=1)
entry_column_names.insert(0, "time,z,y,x")  # Default value

tk.Label(root, text="Use Predictors from GUI:").grid(row=2, column=0, sticky="w")
use_predictors_from_gui = tk.BooleanVar()
tk.Checkbutton(root, variable=use_predictors_from_gui).grid(row=2, column=1, sticky="w")

tk.Label(root, text="Predictors Path:").grid(row=3, column=0, sticky="w")
entry_predictors_path = tk.Entry(root, width=50)
entry_predictors_path.grid(row=3, column=1)
tk.Button(root, text="Browse", command=lambda: entry_predictors_path.insert(0, filedialog.askdirectory())).grid(row=3, column=2)

tk.Label(root, text="Predictors Filename:").grid(row=4, column=0, sticky="w")
entry_predictors_filename = tk.Entry(root, width=50)
entry_predictors_filename.grid(row=4, column=1)

tk.Label(root, text="Sample Ratio:").grid(row=5, column=0, sticky="w")
entry_sample_ratio = tk.Entry(root, width=50)
entry_sample_ratio.grid(row=5, column=1)
entry_sample_ratio.insert(0, "0.03")  # Default value

tk.Label(root, text="Sampling Search Radius Coef:").grid(row=6, column=0, sticky="w")
entry_sampling_search_radius_coef = tk.Entry(root, width=50)
entry_sampling_search_radius_coef.grid(row=6, column=1)
entry_sampling_search_radius_coef.insert(0, "2.5")  # Default value

tk.Label(root, text="Error Function:").grid(row=7, column=0, sticky="w")
error_function_combobox = ttk.Combobox(root, values=["STRAIN", "L2"], width=47)
error_function_combobox.grid(row=7, column=1)
error_function_combobox.set("STRAIN")  # Default value

tk.Label(root, text="Sigma Threshold:").grid(row=8, column=0, sticky="w")
entry_sigma_threshold = tk.Entry(root, width=50)
entry_sigma_threshold.grid(row=8, column=1)
entry_sigma_threshold.insert(0, "3.0")  # Default value

tk.Label(root, text="Max Disp:").grid(row=9, column=0, sticky="w")
entry_max_disp = tk.Entry(root, width=50)
entry_max_disp.grid(row=9, column=1)
entry_max_disp.insert(0, "20")  # Default value

tk.Label(root, text="N Consider:").grid(row=10, column=0, sticky="w")
entry_n_consider = tk.Entry(root, width=50)
entry_n_consider.grid(row=10, column=1)
entry_n_consider.insert(0, "15")  # Default value

tk.Label(root, text="N Use:").grid(row=11, column=0, sticky="w")
entry_n_use = tk.Entry(root, width=50)
entry_n_use.grid(row=11, column=1)
entry_n_use.insert(0, "10")  # Default value

tk.Label(root, text="Save Trace:").grid(row=12, column=0, sticky="w")
save_trace = tk.BooleanVar()
tk.Checkbutton(root, variable=save_trace).grid(row=12, column=1, sticky="w")

tk.Label(root, text="Trace Path:").grid(row=13, column=0, sticky="w")
entry_trace_path = tk.Entry(root, width=50)
entry_trace_path.grid(row=13, column=1)
tk.Button(root, text="Browse", command=lambda: entry_trace_path.insert(0, filedialog.askdirectory())).grid(row=13, column=2)

tk.Label(root, text="Check Linking Accuracy:").grid(row=14, column=0, sticky="w")
check_linking_accuracy = tk.BooleanVar(value=False)  # Default value
tk.Checkbutton(root, variable=check_linking_accuracy).grid(row=14, column=1, sticky="w")

tk.Label(root, text="Linking Data Filename:").grid(row=15, column=0, sticky="w")
entry_linking_data_filename = tk.Entry(root, width=50)
entry_linking_data_filename.grid(row=15, column=1)
entry_linking_data_filename.insert(0, "linked_data.csv")  # Default value

tk.Label(root, text="Memory:").grid(row=16, column=0, sticky="w")
entry_memory = tk.Entry(root, width=50)
entry_memory.grid(row=16, column=1)
entry_memory.insert(0, "0")  # Default value

# Save and Load buttons
tk.Button(root, text="Save Settings", command=save_settings).grid(row=17, column=0, pady=10)
tk.Button(root, text="Load Settings", command=load_settings).grid(row=17, column=1, pady=10)

# Run button
tk.Button(root, text="Run", command=run_script).grid(row=17, column=2, pady=10)

# Start the GUI event loop
root.mainloop()