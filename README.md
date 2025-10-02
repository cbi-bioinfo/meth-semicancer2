# meth-semicancer2
A cancer subtyping framework that integrates semi-supervised learning with domain adaptation and contrastive learning to address the challenges of batch effects and high-dimensional representation learning in DNA methylation data

## Requirements
* Python (>= 3.6)
* Pytorch (>= v1.6.0)
* Other python packages : numpy (>=1.19.1), pandas (>=1.1.1), os, sys, random

## Usage
Clone the repository or download source code files.

## Inputs
[Note!] All the example datasets can be found in './example_data/' directory.

### 1. Source dataset
* Cancer methylation profiles to be used as source domain (**Source_X**)
  - Row : Sample, Column : Feature (CpG or CpG cluster)
  - The first column should have "sample id", and the first row should contain the feature names
  - Example : ./example_data/example_source_X.csv
* Integer-converted subtype label for the source dataset (**Source_Y**)
  - The column name should be "subtype", and the rows should be sorted in the same way as the ones in **Source_X**.
  - The subtype label should start from 0
  - Example : ./example_data/example_source_y.csv

### 2. Target dataset
* Unlabeled cancer methylation profiles to be used as target domain and to be used for prediction (**Target_X**)
   - Row : Sample, Column : Feature (CpG or CpG cluster)
   - The first column should have "sample_id" and the last two coulmns shoud be "batch" and "domain_idx" which contain the batch name (string) and integer number (index) discriminating each dataset. Samples in the same dataset should have same number.
   - The first row should contain the feature names.
   - Example : ./example_data/example_target_X.csv
 
## How to run (Example)
1. Clone the respository, move to the cloned directory, and edit the **run_methSemiCancer2.sh** to make sure each variable indicate the corresponding files.
2. Run the below command :
```
chmod +x run_methSemiCancer2.sh
./run_methSemiCancer2.sh
```
If you clone the directory and run the above command directly, you will get the result for the example dataset.

3. All the results will be saved in the newly created **results** directory.
   * target_prediction.csv : predicted subtype label for each sample in target dataset

## Contact
If you have any questions or problems, please contact to **joungmin AT vt.edu**.
