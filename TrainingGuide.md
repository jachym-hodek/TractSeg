# TractSeg Training

NOTE: If you are confident and have time to spare, there is a bit more concise 13 step guide on training on the official TractSeg github page https://github.com/MIC-DKFZ/TractSeg
NOTE: There were some issues with CUDA, since I use CPU, I don't know if I recommend it, but for training I used my very slightly edited fork of TractSeg: https://github.com/jachym-hodek/TractSeg

Before doing anything, you should activate the TractSeg venv

If the codes at any point don't work, first check if you haven't installed TractSeg separately and the scripts aren't trying to get data from files you aren't actively editing

## Setup

1. Find .tractseg/ directory (it should be in your home directory), inside you should find config.txt file
2. Edit config.txt, change "working_dir" variable to match the path to the directory containing your dataset (the path should be "working_dir"/[subjects_data_folder]/HCP_for_training_COPY/[subject_id]/[nifti_files])

NOTE: the script will expect the individual subject directorie to be in "HCP_for_training_COPY" directory, YOU WILL HAVE TO NAME IT LIKE THIS
NOTE: At the end, in this directory will also be located I. Directory with preprocessed data called "HCP_preproc" containing data in format /[subject_id]/[preprocessed_nifti_files], II. 

 3. Edit "network_dir" to be the same as "woring_dir", but including the path to "HCP_for_training_COPY" (the path should be "network_dir"/HCP_for_training_COPY/[subject_id]/[nifti files])

NOTE: all nifti files containing the same type of data must have the exactly same names, e.g., my peak files of each subject are called "mrtrix_peaks.nii.gz"

NOTE: each subject dir must contain 2 nifti files, I. All 3D bundle masks that you want TractSeg to be trained on (I don't know if including the ones already trained will produce better result, but I would do that) stacked into 4D file

NOTE: tractseg expects one of two resolutions: 1.25mm and 2.5mm, which exactly it is depends on what you set in Config class located in "TractSeg/tractseg/experiments/base.py"


## Adapting preprocessing.py 

1. In "TractSeg/tractseg/data/preprocessing.py" change "DATASET_FOLDER" to the same "[training_subjects]" that is used in "working_dir"

2. Change "dataset" variable to the name of your dataset, the name itself isn't important, but other scripts will need it to determine the name of [subject_id] directories - that is the name of directories containing data of individual subjects - and other stuff, which will be talked about

3. Look for create_preprocessed_files() function, inside there are 3 variables that must be changed: 

- I. Change bb_file to match the name of the peaks file of your individual subject (I changed it to "mrtrix_peaks", because ".nii.gz" is added automatically)
- II. Change filenames_data to match the name of the peaks file of your individual subject (I changed it to "mrtrix_peaks", because ".nii.gz" is added automatically)
- III. Change filenames_seg to match the name of your bundle file (I changed it to "bundle_masks")

## Adapting subjects.py

 1. Look for subjects.py in the same dir as preprocessing.py

 2. Initialise a list containing names of your [subject_id] fies as strings

 3. Find get_all_subjects(dataset="HCP") function, add an elif fork where the code will look for your dataset name and return the name of the list from step 2

## Running preprocessing.py

 NOTE: At this point, I reccommend unistalling  TractSeg and reinstalling the version you just edited to be sure everything runs correctly

 1. Just run the python script from any directory
 2. Output should be located in the "working_dir" you defined at the very beginning



## Adapting my_custom_experiment.py

 located in TractSeg/tractseg/experiments/custom

 1. change FEATURES_FILENAME to match the name of your peaks file (my peaks fike is "mrtrix_peaks.nii.gz" so here I put "mrtrix_peaks")

 2. if you previously changed the name of preprocessing output folder - wchich I don't recommend - change DATASET_FOLDER to match that


# Adapting dataset_specific_utils.py and Config class

dataset_specific_utils.py is located in TractSeg/tractseg/data/dataset_specific_utils.py
class Config is located in TractSeg/tractseg/experiments/base.py

NOTE: This part is a bit tricky, you'll need to edit several functions
1. Add an elif fork in get_bundle_names(CLASSES) lets call whatever you name the class "NEW_BUNDLE_SET" for now, but you can put in whatever you like, the name you decide for here will be used later in the Config class. Make it so for CLASSES == "NEW_BUNDLE_SET", "bundles" equals list of all the bundle names you want to be trained (this will probably mean the 72 already existing ones and the new ones that we want TractSeg to be trained on)

2. Change CLASSES in class Config to match whatever you put in "NEW_BUNDLE_SET"
3. Edit the class Config so the variables like resolution, classes and other match your specific case
4. Go through dataset_specific_utils.py and check there is an option for your Config values, if not, add an the elif option yourself
NOTE: EXPERIMENT_TYPE should be "tract_segmentation", keep that in mind
5. Make sure that get_labels_filename() hands the correct Config.LABELS_FILENAME, that is, the name of your bundles file ("bunle_masks" in my case, since my files are: bundle_masks.nii.gz)

NOTE: get_bundle_names() might need to be edited for new bundles, but I can't know until I actually train with new bundles - I could try to rename some, now that I think of it. The training works so far, so this is just a possible thing to be cautious of

NOTE: You'll need to edit: get_labels_filename(), get_bundle_names(), get_dwi_affine(), get_cv_fold(), I mostly edited them by chan



### FINAL NOTE: Before running everything, I recommend checking the Config class in base.py again, so all the variables are set correctly and make sure that .tractseg folder is in your home - "~" - directory


## RUNNING THE TRAINING
 Now is the point in which you should uninstall and install TractSeg again

run
```sh
ExpRunner --config my_custom_experiment
```
