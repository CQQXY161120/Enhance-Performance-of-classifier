# Enhance-Performance-of-classifier
tkde'23:Efffcient Classiffcation by Removing Bayesian Confusing Samples


## Introduction:

This is the code repository to reproduce the experiments in the paper **[Efffcient Classiffcation by Removing Bayesian Confusing Samples]
(https://ieeexplore.ieee.org/document/10214081)**. This repository is based on numpy, scipy and [Sklearn](https://scikit-learn.org/stable/).

## Prerequisite

We implement our methods by Python 3.8. The environment is as bellow:

- Anaconda 3  
- Linux operating system or Windows operating system  
- Sklearn, numpy

## Run demo

### Synthetic Data Set and the classification decision boundary of three classifiers
<p align="center">
  <img src="https://github.com/CQQXY161120/Enhance-Performance-of-classifier/blob/main/Generated_dataset.png" width='30%' height='30%'/>
  <img src="https://github.com/CQQXY161120/Enhance-Performance-of-classifier/blob/main/DT_before.png" width='30%' height='30%'/>
  <img src="https://github.com/CQQXY161120/Enhance-Performance-of-classifier/blob/main/KNN_before.png" width='30%' height='30%'/>
  <img src="https://github.com/CQQXY161120/Enhance-Performance-of-classifier/blob/main/SGLB_before.png" width='30%' height='30%'/>
</p>



### The Data Set processed by RLP and the classification decision boundary of three classifiers on the processed datasets
<p align="center">
  <img src="https://github.com/CQQXY161120/Instance-Selection/blob/main/Experimental%20Results/circles_reduced.png" width='30%' height='30%'/><img src="https://github.com/CQQXY161120/Instance-Selection/blob/main/Experimental%20Results/moons_reduced.png" width='30%' height='30%'/><img src="https://github.com/CQQXY161120/Instance-Selection/blob/main/Experimental%20Results/Gaussian_reduced.png" width='30%' height='30%'/>
</p>

## Technical Details and Citations:  
You can find more details in the paper:  
[Efffcient Classiffcation by Removing Bayesian Confusing Samples](https://ieeexplore.ieee.org/document/10214081)

If you're using this repo in your research or applications, please cite this BibTeX:

@ARTICLE{10214081,
  author={Cao, Fuyuan and Chen, Qingqiang and Xing, Ying and Liang, Jiye},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Efficient Classification by Removing Bayesian Confusing Samples}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TKDE.2023.3303425}}
