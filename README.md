# Micro-Expression-with-Deep-Learning
Experimentation of deep learning on the subjects of micro-expression spotting and recognition. 

# Platforms and dependencies
Ubuntu 16.04
Python 3.6
Keras 2.0.6
Opencv 3.1.0
pandas 0.19.2
CuDNN 5110. (Optional but recommended for deep learning)


# Download files from this url
https://www.dropbox.com/sh/vccpos21320la6j/AABqP7tpLMbYnpURt7C-PsyEa?dl=0

Since LSTM is used, all the numbers of files have to be the same length. Currently the code does not work on CASMEII Raw. SMIC not tested.

Shape predictor for Facial Landmarks extraction: dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
vgg-16 model pretrained on LFW dataset: https://drive.google.com/file/d/137qZ4dcqOz0sCBAyvihwEOZOqcOyPYbh/view?usp=sharing
CASME2_Optical: https://drive.google.com/open?id=1fq_eHCLiUT9hP0npq6vkMYiO2Ka-39Mf
CASME2_STRAIN: https://drive.google.com/open?id=1-l_CtP9awfMV6pXSrBIPRiIujQLjCv9H

# Running from scratch
main.py is a main control script to run the codes and there are several parameters to tune the training. The guide is as follows:


**List of parameters**:
--train: determines the training script to run. eg: train.py, train_samm_cross.py
--batch_size: the number of data to be run per batch
--spatial_epochs: the number of epochs to run for spatial module(vgg module)
--temporal_epochs: the number of epochs to run for the LSTM/Recurrent module.
--train_id: the name of the training.
--dB: the database/databases to be used. 
--spatial_size: the image resolution
--flag: the type of training to be run. can choose whether to perform Spatial Enrichment, Temporal Enrichment or train single module only
--objective_flag: choose either objective labels or emotion labels.
--tensorboard: choose to use tensorboard. Deprecated.

**Type of flags**:
st - spatial temporal. it's used to train single DB with both vgg-lstm.
st4se - spatial temporal four channel spatial enrichment. optical flow + optical strain.
st4te - spatial temporal four channel temporal enrichment. train both optical flow and optical strain with pre-trained weights and separately.
st5se - flow + strain + grayscale raw image. 
st5te - flow, strain and grayscale train separately with vgg pre-trained weights.

flags with cde behind indicates that to use composite database evaluation as proposed in MEGC 2018. 


**Deprecated/not supported flags**:
s  - spatial only. training vgg only. Remember to use --train './train_spatial_only.py'
t  - temporal only. training the lstm only.
nofine - without finetuning. use the pre-trained weights directly 

**Type of scripts**:
main.py - control scripts.
train.py - training scripts for single db and cde.
models.py - deep models
utilities.py - various functions for preprocessing and data loading.
list_databases.py - scripts to load databases and restructure data.
train_samm_cross.py - hde training
test_samm_cross.py - hde testing
evaluation_matrix.py - for evaluation purposes.
labelling.py - load labels for designated db
lbptop.py - where we created baselines using lbptop
samm_utilities - preprocessing functions for samm only.


**Examples for training temporal only: (the spatial size used in paper is 50)**
Not supported yet. but the code is in train_temporal_only.py

note: for two layers lstm, you can go to models.py and add another line in temporal_module. 
model.add(LSTM(3000, return_sequences=False))


**Example for training spatial only**:
Not supported yet. but the code is in train_spatial_only.py

**Example for single db**:
python main.py --dB 'CASME2_Optical' --batch_size=1 --spatial_epochs=100 --temporal_epochs=100 --train_id='default_test' --spatial_size=224 --flag='st'

**Example for training CDE**:
python main.py --dB 'CASME2_Optical' 'CASME2_Strain_TIM10' --batch_size=1 --spatial_epochs=100 --temporal_epochs=100 --train_id='default_test' --spatial_size=224 --flag='st4se'

**Example for training HDE**:
python main.py --train './train_samm_cross' --dB 'CASME2_Optical' --batch_size=1 --spatial_epochs=100 --temporal_epochs=100 --train_id='default_test' --spatial_size=224 --flag='st4se'

python main.py --dB './test_samm_cross' 'SAMM_Optical' --batch_size=1 --spatial_epochs=100 --temporal_epochs=100 --train_id='default_test' --spatial_size=224 --flag='st4se'

**file structure as follow**:
* asterisk indicates that the folder needs to be created manually

/db* (root):

  /db*
  
    /subjects
    
      /videos
      
        /video_frames.png
        
  /Classification*
  
    /Result*
    
      /db*
      
  /CASME2_label_Ver_2.xls , CASME2-ObjectiveClasses.xlsx, SAMM_Micro_FACS_Codes_v2.xlsx
  
for eg:
/CASME2_Optical:

  /CASME2_Optical
  
    /sub01...sub26
    
      /EP...
      
        /1...9.png
  
  /Classification
  
    /Result
    
      /CASME2_Optical
      
  /CASME2_label_Ver_2.xls , CASME2-ObjectiveClasses.xlsx, SAMM_Micro_FACS_Codes_v2.xlsx

# Results
**Single DB**

F1          : 0.4999726178

Accuracy/WAR: 0.5243902439

UAR         : 0.4395928516

**CDE**

F1          : 0.4107312702

Accuracy/WAR: 0.57

UAR         : 0.39

**HDE**

F1          : 0.3411487289

Accuracy/WAR: 0.4345389507

UAR         : 0.3521973582


**If you find this work useful, here's the paper and citation**

https://arxiv.org/abs/1805.08417

@article{khor2018enriched,
  title={Enriched Long-term Recurrent Convolutional Network for Facial Micro-Expression Recognition},
  author={Khor, Huai-Qian and See, John and Phan, Raphael CW and Lin, Weiyao},
  journal={arXiv preprint arXiv:1805.08417},
  year={2018}
}
