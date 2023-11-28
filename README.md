# 02807_final_project
Final project code base for [02807 Computational Tools for Data Science](http://courses.compute.dtu.dk/02807/2023/).

Contributors:
- [s204125 Andreas Lyhne](https://github.com/AndreasLF)
- [s204138 Michael Harborg](https://github.com/Maplewarrior)
- [s204139 August Tollerup](https://github.com/4ug-aug)
- [s200925 Andreas Bigom](https://github.com/AndreasBigom)

## Abstract
This project compares different methodologies for information retrieval from a document corpus using text queries.
Legacy approaches, which rely on the lexical overlap between queries and documents, such as TF-IDF and the BM25 algorithm, are compared with methods focusing on semantic resemblance. These methods include embedding-based techniques like Dense Passage Retrieval (DPR).
Attempts were made to improve the performance of DPR by performing reranking using an attention-based cross-encoder.
Moreover, trade-offs between the running time and performance of embedding-based methods were considered through clustering approaches, namely K-Means and CURE. 
All methods were implemented from scratch in a Python framework and emphasis was placed on optimizing the running time at inference for all methods.


The methods were compared using two benchmarks: The Financial Question Answering (FiQA) 2018 dataset and a phishing dataset sourced from Kaggle.
The FiQA 2018 dataset includes questions paired with various relevant documents, while the phishing dataset consists of emails tagged with a binary label, as either phishing or non-phishing.
The primary findings of this report were that embedding-based methods were superior on the FiQA 2018 dataset, whereas no clear distinction in classification accuracy could be made for the phishing dataset. 
This indicates, that the performances of the compared methods are highly dependent on the domain of the problem. Lastly, it was evident that a favorable running time when employing KMeans and CURE came at a cost of considerably degraded performance in both cases.

## Reproducing results
To execute the experiments run in this project two notebooks are presented: **run_experiments.ipynb** and **phishing_notebook**.
Firstly, clone the repository:
```bash
git clone https://github.com/Maplewarrior/02807_final_project.git
```
Then, in the root directory of the project, create a virtual env of your choice and/or install `requirements.txt`:
```bash
python -m pip install -r requirements.txt
```
[!IMPORTANT]
On macos the requirement of `pywin32` does not exist and should be commented out directly in the `requirements.txt` file.

