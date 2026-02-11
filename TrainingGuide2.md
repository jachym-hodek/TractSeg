# TracSeg Tract Segmentation Training Guide
### NOTE: THIS GUIDE SERVES ONLY FOR TRAINING A SINGLE TRACT
## I. Setup
1. Install TractSeg from local source:
```sh
git clone https://github.com/MIC-DKFZ/TractSeg.git
pip install -e TractSeg
```
2.Install BatchGenerators from local source:
```sh
git clone https://github.com/MIC-DKFZ/batchgenerators.git
pip install -e batchgenerators
```

## II. Preparing data
1. Change all masks to same voxel size if not already one of the following options: 1.25mm, 2mm, 2.5mm
```sh
mrconvert $INPUT $OUTPUT -voxel 2.5
```
2. Change all masks to same datatype using mrconvert
```sh
mrconvert $INPUT $OUTPUT -axes 0,1,2,-1 -datatype int32 -force
```
3. If you want to train to segment multible bundles at one, merge all tracts of single subject into one file
```sh
mrcat tract1.nii.gz tract2.nii.gz tract3.nii.gz |training_data_file|.nii.gz -axis 3
```
4. Use TractSeg on raw diffusion data to create peaks.nii.gz
5. Move all subject directories into a single |trainign_data| directory
this should be the data structure:
```
custom_path/|training_data|/|subject_01|/
      '-> peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
      '-> |training_data_file|.nii.gz       (Reference bundle masks; shape: [x,y,z,nr_bundles])
custom_path/|training_data|/|subject_02|/
```

## III. Adapt TractSeg/tractseg/libs/preprocessing.py
NOTE: places where adaptation is needed are marked by todo:adapt
Edit the following variables:

```py
dataset = "|your_dataset_name|"
DATASET_FOLDER = "|training_data|"  # source folder
DATASET_FOLDER_PREPROC = "|training_data_preproc|"  # target folder
bb_file = "peaks"
filenames_data = ["peaks"] 
filenames_seg = ["|training_data_file|"]
```

## IV. Adapt TractSeg/tractseg/data/subjects.py
1. add global variable - a list with names of all subject drectorie
```py
|subject_id_list| = ['subject_01','subject_02',...]
```
2. add elif fork to get_all_subjects(dataset)
```py
elif dataset == "|your_dataset_name|":
      return |subject_id_list|
```

## V. Edit .tractseg/config.txt
NOTE: .tractseg should be located in your home directory
```txt
working_dir = /your/training/data/location
etwork_dir = /your/training/data/location
```

## VI. Run preprocessing.py 
```sh
python .../TractSeg/tractseg/libs/preprocessing.py
```

## VII. Adapt TractSeg/tractseg/experiments/custom/my_custom_experiment.py
```py
class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATASET_FOLDER = "|training_data_preproc|"      # name of folder that contains all the preprocessed subjects (each subject has its own folder with the name of the subjectID)
    FEATURES_FILENAME = "peaks"  # filename of nifti file (*.nii.gz) without file ending; mrtrix CSD peaks; shape: [x,y,z,9]; one file for each subject
```

## VIII. Adapt Config in TractSeg/tractseg/experiments/base.py
```py
CLASSES = "|your_bundles_name|" # collective name for bundles which tractseg will be segmenting, used in the next step
EXP_NAME = "|custom_experiment_name|"
DATASET_FOLDER = "|training_data_preproc|"
DATASET = "|your_dataset_name|"
FEATURES_FILENAME = "peaks"
RESOLUTION = "2mm" # options are 1.25mm, 2mm and 2.5mm, choose voxels size of your training data
```

## IX. Adapt TractSeg/tractseg/data/dataset_specific_utils.py
1. add elif fork for your bundles to get_bundle_names()
```py
elif CLASSES == "|your_bundles_name|":        
    bundles = ["bundle1", "bundle2", ...]  #the else option is a backup specifically for 1 tract
```
2. add if fork to get_labels_filename()
```py
if Config.CLASSES == "cing" and Config.EXPERIMENT_TYPE == "tract_segmentation": # tract_segmentation can be changed for trainging tractography or endings segmentation
        Config.LABELS_FILENAME = "|training_data_file|"
        return Config
```
3. edit get_cv_fold()
NOTE: this function divides the dataset into folds for training, validation and testing, this specific version will work only for >20 subjects, for less subjects you will have to use another method of separation
```py
elif dataset == "|your_dataset_name|":
        subjects = get_all_subjects(dataset)
        cut_point = int(len(subjects) * 0.9)
        return subjects[:cut_point], subjects[cut_point:-2], ["|second_to_last_subject_id|", "|last_subject_it|"]
```
## X. Run the training
### Few notes before running the training:
- I used python 3.11, which forced me to change line 252 in TractSeg/tractseg/libs/plot_utils.py from
```py
plt.grid(b=True, which='major', color='black', linestyle='-')
```
to
```py
plt.grid(visible=True, which='major', color='black', linestyle='-')
```
- if you train with less epochs, you may run into issue, where no weights are yet good enough to be saved and training will end in error after running all epochs

### Run
```sh
ExpRunner --config my_custom_experiment
```

## Results
- results will be located in hcp_exp_nodes/my_custom_experiment
- each run after after the first one will create new directory there called my_cutom_experiment_x|number|
- you can run segmentation with these results by using --exp_name path/to/my_custom_experiment
```sh
TractSeg -i $INPUT -o $OUTPUT --output_type tract_segmentation --exp_name my_custom_experiment
```
