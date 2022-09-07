The main training and testing files:

1) Anomalytrain.py : This file contains the code for GAN model which includes both Generator and Discriminator architecture.
2) Anomalytest.py  : This file contains the code for testing the GAN model on different test datasets.


Script Files:
1) train_submit.sbatch : Script submitted to kudu machine to make use of the resources during training of the model.
                        Use sbatch train_submit.sbatch command to run the script.
2) test_submit.sbatch : Script submitted to kudu machine to make use of the resources during testing of the model.
                        However, the testing can also be done by directly running the Anomalytest.py file.


joboutput_*.out Files: This files contain the output that is generated when train_submit.sbatch file is executed.


Dataset Folders:

# The dataset Folders are removed from the Project Directory as the size of these folders were very large (Total of 22 GB)
Links to download these datasets are mentioned at the bottom.
1) UCSD_Anomaly_Dataset : contains UCSD Ped1 and Ped2 Dataset
2) Avenue Dataset : contains CUHK Avenue Dataset
3) ShanghaiTech : contains ShanghaiTech Dataset

Model_Save Folder : Contains saved gan model for all datasets at 0,1000,2000,3000,4000,5000 epochs

Generated_Image_* Folder: Contains the reconstruced sample frames for each dataset at 0,1000,2000,3000,4000,5000 epochs

Generated_Anomaly_Score_* Folder: This folder contains-
                        1) Regularity Score for each test video stored in csv files
                        2) Ground Truth files for the test videos (stored in file names frames_*)
                        3) ROC folder: contains roc graph of tested video frames
                        4) Graphs Folder: contains the plot of Regularity Score vs Frame Number for each test videos
                        5) contains scripts to plot graphs and evaluate results (graph_script.py and evaluation_*.py). 
                            evaluation_*.py file is for ploting the ROC curves and calculating the auc and eer.
                            graph_script.py file contains code to plot the Regularity Score for each test videos. These plots are stored in Graph folder.


CUHK Avenue Dataset Download:  http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
UCSD Dataset Download: 	 http://www.svcl.ucsd.edu/projects/anomaly/dataset.html
ShanghaiTech Dataset Download: https://svip-lab.github.io/dataset/campus_dataset.html