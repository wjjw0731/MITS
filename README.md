# HFTC
# 1.Description
Classifying fungal species via ITS sequences effectively is vital for ecology and society. Existing models fall short. We introduce HFTC, a tree - structured multi - level model, slashing feature dimensions and boosting hierarchical accuracy in full classification. 
# 2. HFTC framefowk
![MITS framefowk](/framework.png)
# 3.Availability
## 3.1.Datasets and source code are available at:
https://github.com/wjjw0731/HFTC/tree/master

As the limitation of file size of github, the trained models are deposited at Zenodo. Please download it form https://zenodo.org/uploads/14826761 and unzip the file(model.rar.gz) to folder "./model" in HFTC.
## 3.2 Local running
### 3.2.1 Environment
The following table describes the required configuration for HFTC：

python==3.9.20

pandas==2.2.2

scikit-learn==1.5.1

gensim==4.3.3
### 3.2.2 Running
Perform both classification and calculation of metrics:

python HierarchicalClassification.py --classify your_json --metrics your_classification_result_csv --output your_metrics_csv
