# TracSeg Tract Segmentation Training Guide
### NOTE: THIS GUIDE SERVES ONLY FOR TRAINING A SINGLE TRACT
## Setup
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
## Preparing data
1. Change all masks to the same size and datatype using mrconvert
```sh
mrconvert $INPUT $OUTPUT -axes 0,1,2,-1 -datatype int32 -force
```
this command changes datatype and adds one axis to make the mask into a 4D "collection of tract masks with only one tract"
2. Use TractSeg on raw diffusion data to create peaks.nii.gz
3. Move all subject directories into a single [trainign_data] directory
this should be the data structure:
```
custom_path/[training_data]/subject_01/
      '-> mrtrix_peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
      '-> bundle_masks.nii.gz       (Reference bundle masks; shape: [x,y,z,nr_bundles])
custom_path/[training_data]/subject_02/
```

