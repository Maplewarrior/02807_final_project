import pathlib, os, requests, zipfile, io, json
import pandas as pd

def DownloadData():
    dataset = "nfcorpus"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets", dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(out_dir)
        print("\nDownloaded and unzipped dataset to\n {}".format(out_dir))
    else:
        print("Dataset already exists in {}".format(out_dir))
        
def LoadData():
    return pd.read_json(path_or_buf='datasets/nfcorpus/nfcorpus/corpus.jsonl', lines=True)

def LoadQueries():
    return pd.read_json(path_or_buf='datasets/nfcorpus/nfcorpus/queries.jsonl', lines=True)

def LoadRelevants():
    return pd.read_csv('datasets/nfcorpus/nfcorpus/qrels/train.tsv',sep='\t')