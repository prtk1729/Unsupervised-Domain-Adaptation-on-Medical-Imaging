# Unsupervised-Domain-Adaptation-on-Medical-Imaging
Unsupervised Domain Adaptation on data from different medical centers and devices


[![maintained by dataroots](https://img.shields.io/badge/maintained%20by-dataroots-%2300b189)](https://dataroots.io)
[![PythonVersion](https://img.shields.io/pypi/pyversions/gino_admin)](https://img.shields.io/pypi/pyversions/gino_admin)
[![Codecov](https://codecov.io/github/datarootsio/ml-skeleton-py/badge.svg?branch=master&service=github)](https://github.com/datarootsio/ml-skeleton-py/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![](https://scontent.fbru1-1.fna.fbcdn.net/v/t1.0-9/94305647_112517570431823_3318660558911176704_o.png?_nc_cat=111&_nc_sid=e3f864&_nc_ohc=-spbrtnzSpQAX_qi7iI&_nc_ht=scontent.fbru1-1.fna&oh=483d147a29972c72dfb588b91d57ac3c&oe=5F99368A "Logo")


### `International Multi-Centre Validation of Unsupervised Domain Adaptation for Precise Discrimination between Normal and Abnormal Retinal Development`


### Objective

`To develop an AI system using Unsupervised Domain Adaptation (UDA) to grade arrested retinal development from OCT scans across various devices and protocols.`

### Explorative results

| Source - Target   | Sensitivity | Specificity | Precision | FPR    | FDR    | FNR    | F1 score |
|-------------------|-------------|-------------|-----------|--------|--------|--------|----------|
| **TM-OCT1 - HH-OCT1** | **0.9838**   | 0.8911      | 0.9651    | 0.1089 | 0.0349 | 0.0162 | **0.9744** |
| TM-OCT1 - TM-OCT2 | 0.9421      | 0.8867      | 0.9547    | 0.1133 | 0.0453 | 0.0579 | 0.9483   |
| HH-OCT1 - TM-OCT1 | 0.8868      | 0.8616      | 0.9158    | 0.1384 | 0.0842 | 0.1132 | 0.9011   |
| HH-OCT1 - TM-OCT2 | 0.9316      | 0.7800      | 0.9147    | 0.220  | 0.0853 | 0.0684 | 0.9231   |
| TM-OCT2 - TM-OCT1 | 0.9658      | 0.8929      | 0.9386    | 0.1071 | 0.0614 | 0.0342 | 0.952    |
| TM-OCT2 - HH-OCT1 | 0.94443     | **0.9802**   | **0.9931**  | **0.0198** | **0.0069** | 0.0557 | 0.9681   |


### How to Run the Project?

#### Dependency
 
```bash
torch>=1.7.0
torchvision>=0.8.0
termcolor>=1.1.0
yacs>=0.1.8
```
 
#### Train
 
 
- [x] Run UDA with DANN from source and target:
    ```bash
    # source and target domains can be defined by "--source" and "--target"
    python main.py configs/UDA_OCT_DANN.yaml --data_root /data/neuroretinal/UDA/6class --source [fov/biop/ukb] --target [fov/biop/ukb]   --output_root exps
    ```
 
- [x] DANN with TM-OCT1 as source and HH-OCT1 as target:
    ```bash
    # example to train from TM-OCT1(fov) - HH-OCT1(biop)
    python main.py configs/UDA_OCT_DANN.yaml --data_root /data/neuroretinal/UDA/6class --source fov --target biop --output_root exps
    ```




### Affiliations

* The University of Leicester Ulverscroft Eye Unit, School of Psychology and Vision Sciences, University of Leicester, RKCSB, PO Box 65, Leicester, LE2 7LX, UK
* International Institute of Information Technology, Gachibowli, Hyderabad - 500 032, India 
  
> Authors
  * `Prateek Pani` (`prateek.pani@research.iiit.ac.in`)
  * `Nikhil Reddy` (`nikhilreddybilla128@gmail.com`)
  * `Dr. Girish Varma` (`girish.varma@iiit.ac.in`)
  * `Dr. Zhanhan Tu` (`zhanhan.tu@leicester.ac.uk`)
  * `Dr. Mervyn Thomas` (`mt350@leicester.ac.uk`)

