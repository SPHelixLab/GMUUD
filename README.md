# G-MUUD
## About The Project
G-MUUD allows users who provide training data for the model to make unlearn requests and remove the influence of his or her data on the model.

## Presented Unlearning Methods
G-MUUD is a machine unlearning approach that leverages guidance models trained on unlearned samples to induce unlearning within the framework of multi-teacher knowledge distillation. G-MUUD treats the original model as the teacher model and use the guidance model to induce the student model unlearning on the unlearned data and remaining data separately. Finally, G-MUUD treats the student model as the unlearned model.

The main function is contained in GMUUD_main.py. 

## Getting Started
### Prerequisites
**GMUUD_unlearn** requires the following packages: 
- Python 3.9.13
- Pytorch 1.13.1
- Sklearn 1.0.2
- Numpy 1.21.5
- Scipy 1.7.3

### File Structure 
```
GMUUD_unlearn
├── datasets
│   ├── Cifar-10
│   ├── Cifar-100
│   └── MNIST
├── data_preprocess.py
├── GMUUD_unlearn_base.py
├── GMUUD_unlearn_main.py
├── Resnet_deep.py
├── Resnet_shallow.py
└── model_initiation.py
```
There are several parts of the code:
- datasets folder: This folder contains the training and testing data for the target model.  In order to reduce the memory space, we just list the  links to theset dataset here.
   -- Cifar-10: https://www.cs.toronto.edu/~kriz/cifar.html
   -- Cifar-100: https://www.cs.toronto.edu/~kriz/cifar.html
   -- MNIST: http://yann.lecun.com/exdb/mnist/
- data_preprocessing.py: This file contains the preprocessing of the raw data in datasets folder.
- GMUUD_unlearn_base.py: This file contains the base function of GMUUD_unlearn, which corresponds to **Section III** in our paper.
- ***GMUUD_unlearn_main.py: The main function of GMUUD_unlearn.***
- model_initiation.py: This file contains the structure of the global model corresponding to each dataset that we used in our experiment.  

## Parameter Setting of GMUUD_unlearn
The settings of GMUUD_unlearn are determined in the parameter **GMUUD_params** in **GMUUD_unlearn_main.py**. 
-- GMUUD_params.data_name: select the dataset 
-- GMUUD_params.num_classes: the number of classes in the dataset
-- GMUUD_params.seed_num: random seed
-- GMUUD_params.use_data_prune: If this parameter is set to True, the dataset is pruned using the method in the paper; if this parameter is set to False, the full dataset is used.
-- GMUUD_params.num_perclass: the number of samples retained in each class in the pruned dataset
-- GMUUD_params.batch_size: the batch size of the model training
-- GMUUD_params.Teacher1_model_train:Used to control whether to retrain the target model, if this parameter is set to True, the target model is retrained; if this parameter is set to False, GMUUD_params.teacher1_path indicates the path where the target model is stored
-- GMUUD_params.KD_epochs: the number of training epoch in knowledge distillation 
-- GMUUD_params.T: distillation temperature
-- GMUUD_params.alpha: hyperparameters for distillation losses
-- GMUUD_params.beta: hyperparameters for distillation losses
-- GMUUD_params.gama: hyperparameters for unlearning loss
-- GMUUD_params.finetune_epochs: controlling whether fine-tuning is performed on the remaining data after unlraning, the value of this parameter indicates the number of epoch of fine-tuning
-- GMUUD_params.cuda_state: check whether gpu is available (torch.cuda.is_available())
-- GMUUD_params.use_gpu: controlling whether to use gpu 


## Execute GMUUD_unlearn
*** Run GMUUD_unlearn_main.py.  ***




