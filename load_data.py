import pathlib, os, requests, zipfile, io

dataset = "quora"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets", dataset)
# download and unzip the dataset
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(out_dir)
    print("Downloaded and unzipped dataset to {}".format(out_dir))
else:
    print("Dataset already exists in {}".format(out_dir))