# 1) General introduction to the workflow.
Version: 3

Data should be stored using the following three paths, all of them defined in the .env file:

	- /workspace/ The path where all the files related to the container are stored, defined in the .env as WORKSPACE_PATH.
	- /output/ The path where all output files are stored, defined in the .env as OUTPUT_PATH.
	- /workspace_images/ The path where all images are stored, defined in the .env as IMAGES_PATH.

All configuration parameters can be modified in ".env" located at "/workspace/.env".

This workflow follows the following steps:

## I. Calculate illumination correction files per plate.
The first step is to adapt and run "/workspace/Pipelines/template_illum.cppipe" to calculate the correction files for the 5 main channels (DNA, RNA, ER, Mito, AGP) for each plate in the running batch. 

This pipeline downscales the images to speed up the calculation of the correction function and later it upscales to the original size. The final files are saved as "PLATE_CHANNEL_illum.npy" in the path "/output/Illum_files/"

## II. Run CellPose cell segmentation on corrected images. [Optional]
First, we apply the illumination correction function on the RNA channel and after that we segment those corrected images with Cellpose-SAM. The pipelines in charge of this steps are "/workspace/Pipelines/template_illum_seg.cppipe" and the code is in "/workspace/Code/cellpose_seg.py". If cellprofiler segmentation is selected trhough the .env file, this module can be skiped to save computing.

The segmentation masks are stored in "/output/Cellpose_segmentations" as "IMAGENAME.png", where IMAGENAME is the name of the respective RNA tif/tiff file. This step should be run in a GPU cluster to reduce the computing time, otherwise it can take significantly longer to process.

## III. Run quality control (QC) on plate images and segmentations.
In this step, we run the QC CellProfiler pipeline (the template defined in the .env file) to create five images (one per channel) that visualize the output of the microscope per plate and the quality of the segmentations. All output data from this step is stored in "/output/QC"

## IV. Generate the QC report and plate collages.
First, the QC output is aggregated, and after that we generate the plate collages and a report with the summary of cell count to evaluate any toxicity or technical noise. The code for this step is in "/workspace/Code/qc/" and it uses the plate map of the cohort ("/workspace_images/platemap_COHORT.csv") to annotate the perturbations. The collages and the report are stored in "/output/QC_collages"

## V. Calculate features per plate.
If the data visualized in the QC step looks good (ask a biologist if unsure), then the features can be calculated based on the obtained images. Here we used the "/workspace/Pipelines/template_analysis.cppipe" pipeline. The per cell morphological profiles output from this module are stored in "/output/Profiles/Raw_profiles/".

## VI. Feature aggregation, normalization and reduction.
Once that the raw morphological profiles are calculated, the next step is to produce compact and statistically representative profiles. For this step, we use the pipeline in "/workspace/code/profiles_processing.py". This script is based on a PyCytoMiner workflow to 1) Aggregate the cell level profiles using the median into well level, 2) Normalize the well level profiles with the robust Z-score, where the MAD and the median used came from the DMSO wells in the plate, 3) Reduce the number of features from the morphological profiles. The output of each of those three steps are stored in "/output/Profiles/Treated_profiles/"

## VII. Clustering and visualization of profiles.
This step is in charge of running clustering (t-SNE, PCA and UMAP) on the input profiles, with all the normalized features or the reduced version, and visualize those clusters on reduced dimensions. The pipeline is on "/workspace/code/clustering.py" and the output is stored in "/output/Clusters/".

## VIII. Evaluate the reproducibility of the morphological profiles.
Once the profiles are created, we measure how well the replicates of a compound resemble each other compared with the rest of the compounds in the batch. The script "/workspace/Code/VII_Reproducibility.py" searches recursively the profiling output folder ("/output/Profiles/") for the whole-plate normalized profiles, and analyzes each set independently. If several files are found, they are concatenated keeping only the features they share. Other profile sets, such as the negative-control normalized ones, can be analyzed by changing the search patterns with "--norm-glob" and "--reduced-glob". This module needs per-well profiles with at least two replicates per compound, labeled in the "Metadata_Perturbation" column.

Five metrics are computed for every compound:
- **Percent Replicating:** the median pairwise Spearman correlation among the replicates of a compound is compared with the 95th percentile of a null distribution built from random sets of profiles of other compounds (5,000 permutations by default, "--null-reps"). A compound replicates if its observed value is above that threshold.
- **Mean Average Precision (mAP):** for each profile, the other replicates of the same compound should be ranked, by cosine distance, ahead of the profiles of other compounds. The average precision (AP) values are averaged per compound, and a p-value is estimated by permutation (10,000 by default, "--map-null-size") and corrected with Benjamini-Hochberg; a compound is significant if its corrected p-value is below 0.05 ("--map-threshold"). The implementation is equivalent to copairs and can use a GPU through CuPy. The pair design can be changed with "--map-pos-diffby", "--map-neg-sameby" and "--map-neg-diffby"; for example, "--map-pos-diffby Metadata_Plate --map-neg-sameby Metadata_Plate" requires the replicates to come from different plates and compares them against profiles of the same plate.
- **Cosine similarity and Spearman correlation:** mean and standard deviation between the replicates of each compound, computed on standardized features.
- **Cohen's d:** effect size separating the similarity distributions of same-compound pairs and cross-compound pairs, for both cosine and Spearman.

The results are stored in "/output/Reproducibility/combined_red/" (feature-selected profiles) and "/output/Reproducibility/combined_norm/" (normalized profiles). Each folder contains "COHORT_reproducibility_report.csv" with all the metrics per compound, "COHORT_map_per_sample.csv" with the AP of every profile and "COHORT_cohens_d.csv" with the effect size per similarity metric, where COHORT is the name given with "-c". All the random procedures use a fixed seed (42), so repeated runs give the same results. Negative controls (e.g. DMSO) are treated as one more compound, so they also appear in the report.

## IX. Subprofile analysis [Optional]
Work in progress 

## X. Morphological map.
Work in progress 

## XI. Random forest.
Work in progress 

# 2) Technical details about the container.

## - Input data.
The "/workspace_images/" should be organized as follows: 

```
Cohort_A/
├── Plate_Pxx_/
│   └── untreated_data/        # Folder with images (must be named exactly "untreated_data")
|   └── Any other metadata files...
├── Plate_Pyy_/
│   └── untreated_data/
├── Plate_Pzz_/
│   └── untreated_data/
└── ...
```

The name of the plates should contain "_Pxx_" (ie. a mayus "P" followed by 2 digits "xx" and surrounded by underscores"_"), so a regex can be applied to extract the XX number of the plate (eg. 01, 09, 11, etc.).

## - Output data.
The

```
Cohort_A/
│
├── Vxx/                          # xx = version of the workflow
│   │
│   ├── CellProfiler_files/   # All files required to run CellProfiler pipelines
│   │   ├── Batch_files/      # Files required to run CellProfiler in Batch mode
│   │   ├── Cellpose_seg/     # Cellpose segmentation masks
│   │   ├── CSVs/             # Metadata required by the Load_data module (illumination pipeline)
│   │   ├── Illum_files/      # Illumination correction functions
│   │   ├── Pipelines/        # Feature extraction pipeline for the dataset
│   │   └── MP/               # Morphological profiles per mini-batch*
│   │
│   ├── QC/
│   |   ├── Images/           # QC images per mini-batch* for the five channels
│   |   ├── Reports/          # QC reports with cell count, nucleus and cell dimensions [Obsolete]
│   |   └── Collages/         # Plate collages for general QC evaluation
│   │
│   ├── Profiles/
│   │   └── Treated_profiles/     # Output from PyCytoMiner (aggregation, normalization, reduction)
│   │
│   └── Clustering/               # Cluster images from morphological profiles
│
└── ...
```

* A mini-batch is defined here as the split of the images made by CellProfiler to parallelize the pipeline.

## - Code.
Conainer is designed as follow

```
MAINCE_container/
│
├── CellProfiler_pipeline/            # Docker container for CellProfiler (excludes CellPose)
│   ├── Code/
│   │   ├── bash_functions.sh         # Auxiliary bash functions used in run_workflow.sh
│   │   ├── correct_illum.sh          # Runs part I of the workflow
│   │   ├── csv_generator.py          # Auxiliary Python functions used in main.py
│   │   ├── main.py                   # Produces CSV files required for CellProfiler headless pipelines
│   │   ├── run_workflow.sh           # Main script, runs processes I and III-VI
│   │   ├── qc                        # Folder with all code related to the image QC workflow
│   │   |	├── assets                # Folder for the QC interface of the report
|   |   |   |   ├── report.css        # Report style configurations
|   |   |   |   ├── report.html       # Report frame
|   |   |   |   ├── report.js         # Logic for the interactive report
|   |   |   |   └── utils.js          # UTIF.js library to process 16 bits tiff files
|   |   |   |
│   │   |	├── io                    # Folder with code in charge of loading the QC morphological profiles from CellProfiler
|   |   |   |   ├── __init__.py       
|   |   |   |   ├── platemap.py       # Load metadata information related to the platemap of each plate
|   |   |   |   └── readers.py        # Code in charge of loading all relevant features from CellProfiler output files
|   |   |   |
│   │   |	├── output                # Folder with code for the QC output files
|   |   |   |   ├── html.py           # Generation of HTML QC report
|   |   |   |   └── pdf.py            # Generation of PDF summary report
|   |   |   |
│   │   |	├── payload               # Folder with code for the conection between Python objects and java script code
|   |   |   |   └── builder.py        # Code that generates the nested dict used by report.js to build the QC graphical report
|   |   |   |
│   │   |	├── render                # Folder with all code related to rendering images for the QC report
|   |   |   |   ├── __init__.py       
|   |   |   |   ├── collage.py        # Make the plate collage to display at the QC report
|   |   |   |   ├── montage.py        # I/O layer for the tiff files and create graphical elements for the report
|   |   |   |   └── overview.py       # Make the miniatures for the QC report and compress images
|   |   |   |
│   │   |	└── stats                 # Folder with code realated to all statistical measurements
|   |   |       ├── __init__.py       
|   |   |       ├── aggregate.py      # Aggregation of well/site/plate morphological profiles, keep track of worst site at syte level.
|   |   |       ├── colors.py         # Assignation of colors accordingly to pass or failed thresholds
|   |   |       ├── engine.py         # Threshold engine to classify CellProfiler QC features outliers
|   |   |       ├── expressions.py    # Polars expressions to compute used statistics from dataframes with QC features
|   |   |       ├── pca.py            # Code to compute PCA to analyze QC features
|   |   |       ├── scores.py         # Batch level statistics such as Z-scores MAD, colorscales and failed wells counting
|   |   |       └── site_flags.py     # Auxiliary code to prepare a parquet file with all sytes that failed any QC criteria
│   │   |
│   │   ├── VI_feat_processing.py     # Step VI — morphological profiling postprocessing
│   │   └── VII_Clustering.py         # Step VII — clustering of morphological profiles
│   │   └── VIII_Reproducibility.py   # Step VIII — reproducibility evaluation of profiles
│   │   └── IX_Subprofiles.py         # Step IX — generation of suprofiles per perturbation 
│   │   └── X_Morphological_map.py    # Step X — graph analysis of morphological profiles network
│   │   └── XI_Random_forest.py       # Step XI — random forest to detect main features that produce the morphological profile signature
│   │
│   ├── Pipelines/                           # CellProfiler pipeline templates, adapted by run_workflow.sh before execution
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
├── .env                          # Parameters required to configure the running of the full workflow
├── .gitignore                    # Specifies files excluded from Git tracking
├── run_container.sh              # Helper script to run the container
└── compose.yml                   # Docker Compose orchestration file - must be adapted before building
```
