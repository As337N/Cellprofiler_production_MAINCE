# 1) General introduction to the workflow.
Version: 2

Data should be stored using the following three paths, all of them defined in the compose.yml file:

	- /workspace/ The path where all the files related to the container are stored.
	- /output/ The path where all output files are stored.
	- /workspace_images/ The path where all images are stored. 

This workflow follows the following steps:

## I. Calculate illumination correction files per plate.
The first step is to adapt and run "/workspace/Pipelines/template_illum.cppipe" to calculate the correction files for the 5 main channels (DNA, RNA, ER, Mito, AGP). 

This pipeline downscales the images to speed up the calculation of the correction function and later it upscales to the original size. The final files are saved as "PLATE_CHANNEL_illum.npy" in the path "/output/Illum_files/"

## II. Run CellPose cell segmentation on corrected images.
First, we apply the illumination correction function on the RNA channel and after that we segment those corrected images with Cellpose-SAM. The pipelines in charge of this steps are "/workspace/Pipelines/template_illum_seg.cppipe" and "/workspace/Code/cellpose_seg.py".

The segmentation masks are stored in "/output/Cellpose_segmentations" as "IMAGENAME.png", where IMAGENAME is the name of the respective RNA tif file. This step should be run in a GPU cluster to reduce the computing time, otherwise it can take more than one day to process one channel per plate.

## III. Run quality control (QC) on plate images and segmentations.
In this step, we run the "/workspace/Pipelines/QC.cppipe" pipeline to create an five images (one per channel) to visualize the output of the microscope per plate and the quality of the segmentations. Additionally, a report with the summary of cell count is generated to evaluate any toxicity or technical noise. Plots with the cellcount per perturbation and plate are also generated at this stage. All output data from this step is stored in "/output/QC"

## IV. Calculate features per plate.
If the data visualized in the QC step looks good (ask a biologist if unsure), then the features can be calculated based on the obtained images. Here we used the "/workspace/Pipelines/template_analysis.cppipe" pipeline. The per cell morphological profiles output from this module are stored in "/output/Profiles/Raw_profiles/".

## V. Feature aggregation, normalization and reduction.
Once that the raw morphological profiles are calculated, the next step is to produce compact and statistically representative profiles. For this step, we use the pipeline in "/workspace/code/profiles_processing.py". This script is based on a PyCytoMiner workflow to 1) Aggregate the cell level profiles using the median into well level, 2) Normalize the well level profiles with the robust Z-score, where the MAD and the median used came from the DMSO wells in the plate, 3) Reduce the number of features from the morphological profiles. The output of each of those three steps are stored in "/output/Profiles/Treated_profiles/"

## VI. Clustering and visualization of profiles.
The last pipeline of this workflow is in charge of running clustering (t-SNE, PCA and UMAP) on the input profiles, with all the normalized features or the reduced version, and visualize those clusters on reduced dimensions. The pipeline is on "/workspace/code/clustering.py" and the output is stored in "/output/Clusters/".

# 2) Technical details about the container.

## - Input data.
The "/workspace_images/" should be organized as follows: 

```
Cohort_A/
├── Plate_Pxx_/
│   └── untreated_data/        # Folder with images (must be named exactly "untreated_data")
├── Plate_Pyy_/
│   └── untreated_data/
├── Plate_Pzz_/
│   └── untreated_data/
└── ...
```

The name of the plates should follow the format "_Pxx_" (ie. a mayus "P" followed by 2 digits "xx") in the filename so a regex can be applied to extract the XX number of the plate (eg. 01, 09, 11, etc.).

## - Output data.
The

```
Cohort_A/
│
├── Vxx/                          # xx = version of the workflow
│   │
│   ├── Plate_Pxx/
│   │   ├── CellProfiler_files/   # All files required to run CellProfiler pipelines
│   │   │   ├── Batch_files/      # Files required to run CellProfiler in Batch mode
│   │   │   ├── Cellpose_seg/     # Cellpose segmentation masks
│   │   │   ├── CSVs/             # Metadata required by the Load_data module (illumination pipeline)
│   │   │   ├── Illum_files/      # Illumination correction functions
│   │   │   ├── Pipelines/        # Feature extraction pipeline for the dataset
│   │   │   └── MP/               # Morphological profiles per plate
│   │   │
│   │   └── QC/
│   │       ├── Images/           # QC images per plate for the five channels
│   │       ├── Reports/          # QC reports with cell count, nucleus and cell dimensions
│   │       └── Collages/         # Plate collages for general QC evaluation
│   │
│   ├── Profiles/
│   │   └── Treated_profiles/     # Output from PyCytoMiner (aggregation, normalization, reduction)
│   │
│   └── Clustering/               # Cluster images from morphological profiles
│
└── ...
```

## - Code.
Conainer is designed as follow

```
MAINCE_container/
│
├── CellProfiler_pipeline/        # Docker container for CellProfiler (excludes CellPose)
│   ├── Code/
│   │   ├── bash_functions.sh     # Auxiliary bash functions used in run_workflow.sh
│   │   ├── correct_illum.sh      # Runs part I of the workflow
│   │   ├── functions.py          # Auxiliary Python functions used in main.py
│   │   ├── main.py               # Produces CSV files required for CellProfiler headless pipelines
│   │   ├── run_workflow.sh       # Main script, runs processes I and III-VI
│   │   ├── III_QC_collage.py     # Step III — quality control check
│   │   ├── V_feat_processing.py  # Step V — morphological profiling postprocessing
│   │   └── VI_Clustering.py      # Step VI — clustering of morphological profiles
│   │
│   ├── Pipelines/                # CellProfiler pipeline templates, adapted by run_workflow.sh before execution
│   │   ├── template_feat_extraction.cppipe  # Feature extraction pipeline (step IV)
│   │   ├── template_illum.cppipe            # Illumination correction function pipeline
│   │   └── template_QC.cppipe               # Auxiliary step III pipeline: generates overlays and calculates QC metrics
│   │
│   ├── Dockerfile                # Container configuration file
│   ├── requirements.txt          # Python dependencies - must be adapted before building
│   └── variables.env             # Environment variables - must be adapted before running
│
├── Cellpose_seg/                 # Docker container for CellPose segmentation
│   ├── Code/
│   │   ├── II_cellpose_seg.py    # Step II — CellPose segmentation
│   │   └── run_cellpose.sh       # Main script for step II
│   │
│   ├── Dockerfile                # Container configuration file
│   ├── requirements.txt          # Python dependencies  - must be adapted before building
│   └── variables.env             # Environment variables - must be adapted before building
│
├── logs/                         # Stores all logs generated during container execution
├── .env                          # Docker user/group ID to run the container with the correct permissions
├── .gitignore                    # Specifies files excluded from Git tracking
├── run_container.sh              # Helper script to run the container
└── compose.yml                   # Docker Compose orchestration file - must be adapted before building
```