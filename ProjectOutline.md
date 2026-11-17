# Where has my fish gone: Acoustic classification of Fish from Moored Echosounder Data

**Team:** Olivia Whitelaw, Amanda Tu, Ali Stein-Gagnon, Carson Underhay

## 1. Project Overview

Acoustic surveys are widely used to monitor fish stocks and zooplankton (Rousseau et al., 2022). Primitive research into acoustic fish detection dates back to 1929, when sound waves were used to detect goldfish in a pond (Johannesson & Mitson, 1983). Modern echosounders transmit sound waves and record returning echoes to study the distribution and abundance of aquatic organisms. For example, Rousseau et al. (2018) used fixed, autonomous Acoustic Zooplankton Fish Profilers (AZFPs) to monitor juvenile salmon migration in the Discovery Islands and Johnstone Strait, British Columbia.

Acoustic data analysis remains time-consuming and requires expert knowledge to classify biological targets from noise and other acoustic signals (Slonimer, 2022). This limits the efficiency of processing large datasets. Our project aims to develop models to classify fish targets in labelled AZFP data provided by Fisheries and Oceans Canada (DFO). This involves preprocessing and standardizing the data, developing models informed by existing research, and evaluating their performance. 

## 2. Problem Statement

This project builds and compares two classifiers that label echogram regions from a single DFO moored echosounder site as herring or salmon: a random forest (RF) trained on per-region acoustic, shape, and context features, and a convolutional neural network (CNN) trained on multi-frequency echogram images. Both models are trained and evaluated on the same data split, so their performance can be directly compared. As stretch goals, we will test how well both models generalize to a second, unseen site without retraining.

## 4. Dataset Description

### Data Collection

The data were provided by Fisheries and Oceans Canada (DFO) and were collected by an autonomous Acoustic Zooplankton Fish Profiler (AZFP; ASL Environmental Sciences) moored on the seafloor in the Malaspina Strait, British Columbia. The instrument was deployed at a depth of 110 m looking upward and recorded continuously at 67 and 125 kHz from 1 March 2025 to 1 March 2026. Acquisition settings are listed in Table 1. They were taken from the deployment's `.XML` and `.DPL` files and remained constant over the year. The raw data consist of 8,745 hourly `.01A` files, totalling about 31 GB.

**Table 1.** Acquisition settings of the Malaspina Strait AZFP deployment (March 2025–March 2026). Pulse durations are given for 67 kHz / 125 kHz.

| Parameter | Value |
|---|---|
| Frequencies | 67, 125 kHz |
| Ping interval | 12 s |
| Pulse duration | 500 µs / 300 µs |
| Digitization rate | 40 kHz |
| Range bin size | 0.109 m |
| Range bins | 1,192 |
| Maximum range | 129.7 m |
| Instrument depth | 110 m |

### Labelled Regions

DFO analysts annotated the data in Echoview, producing weekly region files (`.evr`) and surface and bottom lines (`.evl`). These contain 427 labelled regions (Table 2). Hake and krill/zooplankton have too few examples to train on, so we use the 353 herring and salmon regions.

**Table 2.** DFO-labelled analysis regions in the Malaspina Strait deployment, by class.

| Class | Regions |
|---|---:|
| Herring | 217 |
| Salmon | 136 |
| Unclassified | 70 |
| Krill/zooplankton | 3 |
| Hake | 1 |
| **Total** | **427** |

## 5. Proposed Methodology

- **Label verification:** count DFO labelled regions per species
- **Acoustic processing:** convert raw AZFP data to Sv; remove bad pings, the surface boundary and background noise; separate day and night
- **Region extraction:** match DFO's labelled regions to cleaned data; compute the dB different
- **Classification models:** random forest on per-region features vs a CNN on region images, using the same train/ test split
- **Evaluation:** per class precision, recall and F1; RF vs CNN comparison

## 6. Tools and Technologies

- **Python and Echopype:** Echopype is an open-source Python library for interoperable and scalable echosounder data processing (Lee et al., 2024). It supports conversion of raw AZFP echosounder data files (.01A) into standardized NetCDF datasets which can then be analyzed and processed (Echopype Developers, 2024). It allows for integration with Python machine learning libraries which makes it suitable for this project.
- **Echofilter:** open-source deep-learning tool that detects the surface in upward looking echosounder data 
- **Xarray, Dask and Zarr:** Echopype is built on xarray; Zarr stores data in compressed chunks and Dask processes those chunks in parallel (Lee et al., 2024)
- **NumPy, Scipy and scikit-image:** used for array processing, filtering and morphological operations
- **Pvlib or astral:** calculate solar elevation from ping timestamps and mooring coordinates
- **Pytorch and torchvision:** used to build and train CNN
- **Pytorch-grad-cam:** Grad-CAM heatmaps for the CNN
- **GPU access**

## 7. Evaluation Metrics

- Balanced Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- Learning curve

## 8. Timeline

| Weeks | Tasks |
|---|---|
| 1-2 | Preparation of dataset |
| 3-4 | Model training and tuning |
| 5 | Evaluation |
| 6 | Wrap up, final report and presentation |

## 9. Related Work

- **Rousseau et al. (2022):** DFO study using a seafloor-mounted, upward-looking, multi-frequency AZFP in Okisollo Channel (Discovery Islands), the same setup as our data. A random forest on per-school acoustic, shape and depth features separated juvenile salmon from herring with 98% accuracy, using daytime data only, background noise removal and a multi-frequency fish mask. Our RF baseline, feature set and fish mask follow this paper. 
- **Marques et al. (2021):** Uses Mask R-CNN to outline herring and juvenile salmon schools in 67 kHz echograms from DFO's seafloor AZFP in Okisollo Channel, with 153 herring and 252 salmon examples. Training from scratch performed poorly, so the model started from pretrained weights (transfer learning), which supports our use of a pretrained ResNet-18. 
- **Slonimer (2022); Slonimer et al. (2023):** Trains a U-Net to label every echogram pixel as herring, salmon, bubbles, surface or background, using AZFP data at 67, 125, 200 and 455 kHz stacked as input channels plus depth and solar elevation channels. Reached F1 scores of 93.0% (herring), 87.3% (salmon) and 86.5% (bubbles), and the added depth and solar elevation channels improved accuracy. Source for our fixed dB scaling, context channels and surface detection approach. 
- **Pala et al. (2024):** Their model first learns from unlabelled echogram patches, then is fine-tuned on the few labelled ones. Open-source code exists. Relevant because most of our year-long recordings are unlabelled. 


## 10. Individual Objectives

### Olivia Whitelaw

**Preprocessing**

**Objective 1: Data inventory and label audit**
- PI1 (basic, done): Parse all 53 weekly `.evr` files into a region table and count regions per class and week
- PI2 (basic, done): Extract settings and calibration from every month's `.XML`/`.DPL` files and flag inconsistencies
- PI3 (basic, done): Sort the `.01A` files into weekly folders matching the `.evr` weeks
- PI4 (expected, in progress): Complete the Sv calibration table by cross-checking with the `.ECS` file
- PI5 (expected, not started): Confirm that region times and depths align with the Sv echograms, including the clock time zone

All 53 weekly Echoview region files were parsed into a single table of 1,054 regions, recording each region's time span, depth range, class and type (PI1). Each file was checked against its expected weekly time window, and all 53 were consistent. Instrument settings and calibration values were extracted from every month's `.XML` and `.DPL` files (PI2). They were constant over the deployment, apart from a disagreement in the number of averaged burst pings between the two file types. The raw files were sorted into weekly folders so that each week of Sv data matches its region file (PI3).

The Sv calibration table is partly complete (PI4): the transducer coefficients have been cross-checked against the Echoview calibration file, but sound speed, absorption and salinity still need to be derived from the RBR sensor data. The main open issue is that the instrument clock does not record its time zone, so region alignment with the Sv data (PI5) cannot yet be confirmed.

**Objective 2: Boundary removal**
- PI1 (basic, in progress): Flag and remove bad pings, gaps and Echoview "bad data" regions
- PI2 (basic, not started): Mask the surface and near-field using the `.evl` lines where they exist
- PI3 (expected, not started): Detect the surface automatically for periods without `.evl` lines, first testing Echofilter (Lowe et al., 2022) and then a detector based on the high-frequency Sv gradient (Slonimer, 2022)
- PI4 (expected, not started): Validate automatic surface detection against the existing `.evl` lines
- PI5 (advanced, not started): Mask the seafloor or obstructions if they appear in the echograms

The 626 bad-data regions identified during the label audit (Objective 1) give the time and depth extents to be removed for PI1. The next step is to convert them into masks on the Sv grid, together with pings that contain missing or corrupted samples. Surface masking with the `.evl` lines (PI2) depends on the region alignment check (Objective 1, PI5), since the lines share the same time and depth references as the regions. Echofilter was trained on Simrad EK80 data rather than AZFP recordings, so it may need adjusting before it performs well on this dataset. The gradient-based detector is a fallback if it does not.

**ML: CNN**

**Objective 3: Build the CNN image dataset**
- PI1 (basic, in progress): Extract the cleaned Sv inside each labelled region as a fixed-size image
- PI2 (basic, not started): Stack Sv at 67 kHz, Sv at 125 kHz and their difference as input channels, using one fixed dB range for all images
- PI3 (expected, not started): Sample "no fish" background images from empty water
- PI4 (expected, not started): Add data augmentation (crop shifts, time flips and small level offsets) to compensate for the small number of labelled regions
- PI5 (advanced, not started): Add context channels for depth below the surface and solar elevation, which improved accuracy in previous work (Slonimer et al., 2023)

The time, depth and polygon outline of each of the herring and salmon regions have been exported, together with their assignment to training, validation and test sets, so the regions can be cut out of the Sv data once it has been cleaned (Objectives 1 and 2). Images will be scaled with a single fixed dB range rather than per image, so that differences in backscatter strength between species are preserved (Slonimer, 2022). With only 136 salmon regions, augmentation (PI4) is likely to matter more than in typical image tasks.

### Amanda Tu

**Preprocessing**

**Objective 1: Noise removal**
- PI1 (basic): Apply background noise removal to the Sv data
- PI2 (basic): Produce before/ after echograms
- PI3 (expected): Add transient and impulse noise filtering
- PI4 (expected): Measure how many labelled regions remain after each cleaning step

**Objective 2: Train the CNN**
- PI1 (basic): Fine-tune a pretrained CNN (ResNet-18) on the shared split
- PI2 (basic): Apply regularization and early stopping to prevent overfitting
- PI3 (expected): Use a hierarchical output: the model decides in stages (fish or not, then herring or salmon, then adult or juvenile salmon, if enough labels), so "Unclassified" regions still train the first decision
- PI4 (expected): Report per-class precision, recall, F1 and confusion matrices on the same test set as the RF
- PI5 (advanced): Use hierarchical output to stop at broader label when the model is unsure, and measure how accuracy changes when it's allowed to answer less specifically

### Ali Stein-Gagnon

**Preprocessing**

**Objective 1: Day/ night separation and fish masking**
- PI1 (basic): Calculate solar elevation from each ping's timestamp and mooring coordinates
- PI2 (basic): Separate day and night data
- PI3 (expected): Compute the dB difference between the two frequencies
- PI4 (expected): Apply multi-frequency mask to separate fish from background (Rousseau et al., 2022)
- PI5 (advanced): Use the fish mask to remove background pixels inside each labelled region before the images are made

**ML: Random Forest**

**Objective 2: Feature Extraction**
- PI1 (basic): Per-region features: mean Sv at 67 and 125 kHz, Sv (67-125 kHz), mean depth, height, length, area
- PI2 (basic): Add context features: time of day, day of year, solar elevation, depth below surface
- PI3 (expected): Add shape and texture features: Sv variability, skewness, kurtosis
- PI4 (expected): Correlation/VIF screening to remove redundant features (following Rousseau et al., 2022)
- PI5 (advanced): Compare with Echoview's "analysis by region" school descriptor export if DFO has one

**Objective 3: Explore CNN variants**
- PI1 (basic): Backbone vs full fine-tuning
- PI2 (basic): Compare using 67 kHz only vs both frequencies plus their difference
- PI3 (expected): Plot CNN accuracy against the number of training regions for comparison with the RF
- PI4 (advanced): Use Grad-CAM heatmaps to check that the CNN focuses on the school rather than the surface or background

### Carson Underhay

**Preprocessing**

**Objective 1: Shared train/ validation/ test splits**
- PI1 (basic): Split the data by deployment week so no week appears in more than one set
- PI2 (basic): Save one split file used by both the RF and the CNN
- PI3 (expected): Report the number of regions per species in each split
- PI4 (expected): Flag species with too few regions for the CNN
- PI5 (advanced): Set up grouped cross-validation by week for more reliable scores on small classes

**ML: Random Forest**

**Objective 2: RF training and tuning**
- PI1 (basic): Baseline RF on the shared split
- PI2 (basic): Class weighting or balanced sampling for imbalanced classes
- PI3 (expected): Hyperparameter tuning with grouped cross-validation (never random folds)
- PI4 (expected): Per-class precision, recall, F1 and confusion matrices
- PI5 (advanced): Reproduce Rousseau's random-split setup next to the week-held-out split to measure how much a random split inflates scores

**Objective 3: Interpretation**
- PI1 (basic): Rank feature importance
- PI2 (basic): Check whether depth or date dominate predictions, which would suggest that the model relies on position and season rather than acoustic cues
- PI3 (expected): Identify which species are most often confused
- PI4 (expected): Show example echograms of correctly and incorrectly classified regions
- PI5 (advanced): Plot RF accuracy against the number of training regions for comparison with the CNN

## 11. References

**Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

> Introduces the random forest, a classifier that combines many decision trees trained on random subsets of the data and features. It is the baseline model we compare against the CNN.

**Echopype Developers. (2024).** *Converting raw files*. Echopype documentation. https://echopype.readthedocs.io/en/stable/convert.html#supported-raw-file-types

> Technical documentation listing the raw file types Echopype supports, including AZFP (.01A) files, which require the deployment's .XML file for conversion via the `sonar_model="AZFP"` setting.

**He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90

> Introduces ResNet, a CNN architecture. ResNet-18, the smallest version, is the pretrained backbone we fine-tune for species classification.

**Johannesson, K. A., & Mitson, R. B. (1983).** *Fisheries acoustics: A practical manual for aquatic biomass estimation* (FAO Fisheries Technical Paper No. 240). Food and Agriculture Organization of the United Nations. https://www.fao.org/4/x5818e/x5818e00.htm

> Foundational FAO manual on using echosounders to estimate fish biomass, including the early history of acoustic fish detection cited in our overview.

**Lee, W.-J., Setiawan, L., Tuguinay, C., Mayorga, E., & Staneva, V. (2024).** Interoperable and scalable echosounder data processing with Echopype. *ICES Journal of Marine Science*, 81(10), 1941–1951. https://doi.org/10.1093/icesjms/fsae133

> Paper from the Echopype developers describing the open-source Python package that converts raw echosounder data (including AZFP) into a standard netCDF/Zarr format and computes Sv. Built on xarray with Dask, so it scales from a laptop to the cloud for large datasets like ours.

**Lowe, S. C., McGarry, L. P., Douglas, J., Newport, J., Oore, S., Whidden, C., & Hasselman, D. J. (2022).** Echofilter: A deep learning segmentation model improves the automation, standardization, and timeliness for post-processing echosounder data in tidal energy streams. *Frontiers in Marine Science*, 9. https://doi.org/10.3389/fmars.2022.867857

> Echofilter is an open-source deep-learning tool that draws boundary lines you'd otherwise draw by hand in Echoview for upward-looking moorings. It was trained on Simrad EK80 data, not AZFP, so signal and noise will differ and it might work less well without retraining.

**Marques, T. P., Cote, M., Rezvanifar, A., Branzan Albu, A., Ersahin, K., Mudge, T., & Gauthier, S. (2021).** Instance segmentation-based identification of pelagic species in acoustic backscatter data. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. https://openaccess.thecvf.com/content/CVPR2021W/PBVS/papers/Marques_Instance_Segmentation-Based_Identification_of_Pelagic_Species_in_Acoustic_Backscatter_Data_CVPRW_2021_paper.pdf

**OSOceanAcoustics. (n.d.).** *Echoregions* [Computer software]. GitHub. https://github.com/OSOceanAcoustics/echoregions

> Open-source Python library that reads Echoview region (`.evr`) and line (`.evl`) files and converts them into masks aligned with Echopype Sv data. Links DFO's Echoview labels to our processed data.

**Pala, A., Oleynik, A., Malde, K., & Handegard, N. O. (2024).** Self-supervised feature learning for acoustic data analysis. *Ecological Informatics*, 84, 102878. https://doi.org/10.1016/j.ecoinf.2024.102878

**Rezvanifar, A., Marques, T. P., Cote, M., Branzan Albu, A., Slonimer, A., Tolhurst, T., Ersahin, K., Mudge, T., & Gauthier, S. (2019).** A deep learning-based framework for the detection of schools of herring in echograms. *NeurIPS 2019 Workshop on Tackling Climate Change with Machine Learning*. arXiv:1910.08215. https://arxiv.org/abs/1910.08215

> A two-stage system on AZFP data from Discovery Passage, BC: a rule-based detector (median filtering, adaptive thresholding, morphological opening/closing, dropping small or flat blobs) proposes regions, then a CNN classifies each as herring or background. DenseNet201 performed best (F1 ~ 0.82).

**Rousseau, S., Gauthier, S., Johnson, S., Neville, C., & Trudel, M. (2018).** *Juvenile salmon acoustic monitoring in the Discovery Islands, British Columbia* (Canadian Technical Report of Fisheries and Aquatic Sciences 3277). Fisheries and Oceans Canada. https://publications.gc.ca/collections/collection_2018/mpo-dfo/Fs97-6-3277-eng.pdf

**Rousseau, S., Gauthier, S., Neville, C., Johnson, S., & Trudel, M. (2022).** Acoustic classification of juvenile Pacific salmon (*Oncorhynchus* spp) and Pacific herring (*Clupea pallasii*) schools using random forests. *Frontiers in Marine Science*, 9, 857645. https://doi.org/10.3389/fmars.2022.857645

**Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).** Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74

> Introduces Grad-CAM, which produces a heatmap showing which parts of an image a CNN relied on for its prediction. We use it to check that our CNN focuses on the fish school rather than the surface or background.

**Slonimer, A. (2022).** *Automatic classification of fish and bubbles at pixel-level precision in multi-frequency acoustic echograms using U-Net convolutional neural networks* [Master's thesis, University of Victoria]. http://hdl.handle.net/1828/13850

**Slonimer, A. L., Dosso, S. E., Branzan Albu, A., Cote, M., Marques, T. P., Rezvanifar, A., & Ersahin, K. (2023).** Classification of herring, salmon, and bubbles in multifrequency echograms using U-Net neural networks. *IEEE Journal of Oceanic Engineering*, 48(4), 1236–1254. https://doi.org/10.1109/JOE.2023.3272393

