# 02807_final_project
Final project code base for [02807 Computational Tools for Data Science](http://courses.compute.dtu.dk/02807/2023/).

Contributors:
- [s204125 Andreas Lyhne](https://github.com/AndreasLF)
- [s204138 Michael Harborg](https://github.com/Maplewarrior)
- [s204139 August Tollerup](https://github.com/4ug-aug)
- [s200925 Andreas Bigom](https://github.com/AndreasBigom)

## Content
<!-- TOC start  -->
- [Abstract](#abstract)
- [Reproducing results](#reproducing-results)
   * [(Optional)  Using prebuilt models and indices](#optional-using-prebuilt-models-and-indices)
   * [Running experiment on FiQA 2018 benchmark](#running-experiment-on-fiqa-2018-benchmark)
      + [(Optional) Getting the data](#optional-getting-the-data)
      + [Running the experiment](#running-the-experiment)
   * [Testing models on phishing data](#testing-models-on-phishing-data)
      + [Getting the data](#getting-the-data)
      + [Running the experiments](#running-the-experiments)
<!-- TOC end -->


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
> [!NOTE]
> All results in this project are produced using an RTX 4070 on [Python 3.9.13](https://www.python.org/downloads/release/python-3913/) and it is recommended to run this with a GPU and CUDA available.

To execute the experiments run in this project two notebooks are presented: `run_experiments.ipynb` and ``phishing_notebook.ipynb``.
Firstly, clone the repository:
```bash
git clone https://github.com/Maplewarrior/02807_final_project.git
```
Then, in the root directory of the project, create a [virtual env](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) of your choice and install `requirements.txt`:
```bash
python -m pip install -r requirements.txt
```
> [!NOTE]
> On MacOS the requirement of `pywin32` does not exist and should be commented out directly in the `requirements.txt` file.

### (Optional) Using prebuilt models and indices
As building the models and indices can be very time consuming especially on a CPU, we have precomputed these and uploaded them to [Google Drive](https://drive.google.com/drive/folders/13jPojmVvgFjntUt7AwJRSBcBQxO_-FxV).
Models should be placed in models/pickled_models and indices should be placed in indexes with subfolders as shown below:
```
├── configs
│   ├── ...
├── data
│   ├── ...
├── indexes
│   ├── fiqa
│   │   ├── embedding_index.pickle  <------ Place prebuilt indexes here
│   ├── phishing
│   │   ├── embedding_index.pickle  <------ Place prebuilt indexes here
├── models
│   ├── builders
│   │   ├── ...
│   ├── pickled_models
│   │   ├── fiqa
│   │   │   ├── BM25.pickle        <------ Place prebuilt model here
│   │   │   ├── TF-IDF.pickle      <------ Place prebuilt model here
│   │   │   ├── ...                <------ Place prebuilt model here
│   │   ├── phishing
│   │   │   ├── BM25.pickle        <------ Place prebuilt model here
│   │   │   ├── TF-IDF.pickle      <------ Place prebuilt model here
│   │   │   ├── ...                <------ Place prebuilt model here
│   ├── ...
├── old_code
│   ├── ...
├── utils
│   ├── ...
└── ...
```

> [!IMPORTANT]
> When running the notebooks you should set `load_saved_models = True` in the experiment configuration code block.

Now you should be ready to use the prebuilt models and indices.

> [!NOTE]
> Even with our prebuilt models and indices the inference might still take a while to run. We highly recommend running it on a GPU!

### Running experiment on FiQA 2018 benchmark

#### (Optional) Getting the data
We have made a script to download the dataset automatically from the [BEIR Project](https://github.com/beir-cellar/beir)) which means you only need to run the notebook. If you want to manually download it, extract the content to `data/datasets/fiqa`.

Whether downloading it manually or relying on the script integrated into the notebook, the folder structure should look as follows:

```
├── configs
│   ├── ...
├── data
│   ├── datasets
|   |   ├── fiqa
|   |   |   ├── qrels
|   |   |   |   ├── dev.tsv
|   |   |   |   ├── test.tsv
|   |   |   |   ├── train.tsv
|   |   |   ├── corpus.jsonl
|   |   |   ├── queries.jsonl
└── ...
```

#### Running the experiments
Just run `run_experiments.ipynb`. Easy! 
You can change the experiment configurations in the _Define Experiment Configuration_ code block. We ran the experiments with the following settings:

```Python
config = configparser.ConfigParser()
config.read('configs/config.ini')
datasets = list(config['DATASETS'])
data_loader = DataLoader(config)

load_saved_models = False

embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1" #'ProsusAI/finbert' #"bert-base-uncased"
embedding_index_folder_path = "indexes"
top_k = 100
batch_size = 50
subset_factors = [1, 2, 4]

model_descriptions = {
        "TF-IDF": {},
        "BM25": {},
        "DPR": {},
        "Crossencoder": {"n" : top_k*2},
        "KMeans": {"k":3},
        "CURE": {"n": 25, # Represenative points
                "shrinkage_fraction" : 0.1, # Fraction of points to be removed
                "threshold": 0.35, # Threshold for merging clusters
                "initial_clusters": 50, # Initial number of clusters
                "subsample_fraction": 0.5, # Fraction of points to be used for clustering
                "similarity_measure": "cosine"}}
```

### Testing models on phishing data

#### Getting the data
First, download the dataset from [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails) and place it in the `data/dataset` folder.
It should look as follows:
```
├── configs
│   ├── ...
├── data
│   ├── datasets
|   |   ├── fiqa
|   |   |   ├──  ...
|   |   ├── Phishing_Email.csv
└── ...
```
> [!NOTE]
> You will need a Kaggle account to download the data.
> We have not provided it in the repo as it is not ours and we do not have permission to distribute it.

#### Running the experiments
To run the experiments, just run the notebook `phishing_notebook.ipynb`.


Again, experiment configurations can be changed in the _Define Experiment Configuration_ code block. We ran the experiments with the following settings:

```Python
config = configparser.ConfigParser()
config.read('configs/config.ini')
data_loader = DataLoader(config)

top_k = 25
test_split = 0.2
batch_size=25

model_descriptions = {
        "TF-IDF": {},
        "BM25": {},
        "DPR": {},
        "Crossencoder": {"n":2*top_k},
        "KMeans": {"k":3},
        "CURE": {"n": 25,
                "shrinkage_fraction" : 0.1,
                "threshold": 0.25,
                "initial_clusters": 50,
                "subsample_fraction": 0.5,
                "similarity_measure": "cosine"}}

load_saved_models = False

embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"#"bert-base-uncased"
embedding_index_folder_path = "indexes"
phishing_dataset_path = "data/datasets/phishing_dataset.pickle"
datasets_path = "data/datasets/Phishing_Email.csv"
```


