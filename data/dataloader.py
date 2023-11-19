import json
import os
import pathlib, os, requests, zipfile, io
import csv

class DataLoader():
    def __init__(self, config_parser):
        self.dataset_urls = dict(config_parser['DATASETS'])
        self.dataset_path = config_parser['PATHS']['datasets']
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        
    def load_jsonl(self, path):
        """Load a jsonl file.

        Args:
            path (str): The path to the jsonl file.
        
        Returns:
            list[dict]: The data in the jsonl file.
        """
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]
        
    def load_data(self, path):
        """Load the corpus and queries from jsonl files.
        
        Args:
            path (str): The path to the dataset folder.
        
        Returns:
            corpus (list[dict]): The corpus of the dataset.
            queries (list[dict]): The queries of the dataset.
        """
        corpus_path = os.path.join(path, 'corpus.jsonl')
        queries_path = os.path.join(path, 'queries.jsonl')
        print(corpus_path)
        self.corpus = self.load_jsonl(corpus_path)
        self.queries = self.load_jsonl(queries_path)
        return self.corpus, self.queries
    
    def convert_to_dict(self, list_of_dicts):
        """
        Convert a list of dictionaries to a dictionary keyed by '_id'.

        Args:
            list_of_dicts (list): List of dictionaries, each containing an '_id' key.

        Returns:
            dict: Dictionary where each key is an '_id' and its value is the corresponding dictionary.
        """
        return {item['_id']: item for item in list_of_dicts}


    def get_corpus(self):
        """Return the loaded corpus.
        
        Returns:
            corpus (list[dict]): The corpus of the dataset.
        """
        return self.convert_to_dict(self.corpus)

    def get_queries(self):
        """Return the loaded queries.
        
        Returns:
            queries (list[dict]): The queries of the dataset.
        """
        return self.convert_to_dict(self.queries)

    def check_dataset(self, dataset_name):
        """Check if the dataset with the folder name is already downloaded.
        
        Args:
            dataset_name (str): The folder name of the dataset.

        Returns:
            bool: True if the dataset is already downloaded, False otherwise.
        """
        # check if dataset_name is in config
        if dataset_name not in self.dataset_urls:
            raise ValueError("Dataset not defined in config")

        return os.path.exists(os.path.join(self.dataset_path, dataset_name))
    
    def download_dataset(self, dataset_name):
        """Download the dataset.

        Args:
            dataset_name (str): The folder name of the dataset.
        """
        # check if dataset_name is in config
        if dataset_name not in self.dataset_urls:
            raise ValueError("Dataset not defined in config")

        out_dir = self.dataset_path
        # out_dir = os.path.join(out_dir, dataset_name)

        # check if dataset already exists
        if not self.check_dataset(dataset_name):
            # download dataset
            url = self.dataset_urls[dataset_name]
            print("Downloading dataset from {}".format(url))
            # Send a GET request to get the file size
            file_size = int(requests.head(url).headers.get('content-length', 0))
            
            with requests.get(url, stream=True) as r:
                with open(os.path.join(out_dir, os.path.basename(url)), 'wb') as file:
                    chunk_size = 8192  # Adjust the chunk size as needed
                    downloaded_size = 0

                    for chunk in r.iter_content(chunk_size=chunk_size):
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Calculate and print the download progress
                        progress = (downloaded_size / file_size) * 100
                        print(f"\rDownload progress: {downloaded_size}/{file_size} bytes ({progress:.2f}%)", end="")

            print()  # Print a newline after the progress indicator
            print("Extracting dataset to {}".format(out_dir))
            z = zipfile.ZipFile(os.path.join(out_dir, os.path.basename(url)))
            z.extractall(out_dir)
            print("\nDownloaded and unzipped dataset to\n {}".format(out_dir))

        else:
            print("Dataset already exists in {}".format(out_dir))

    def get_dataset(self, dataset_name):
        """Get the corpus and queries from the dataset. Download the dataset if it is not already downloaded.
        
        Args:
            dataset_name (str): The folder name of the dataset.
        
        Returns:
            corpus (list[dict]): The corpus of the dataset.
            queries (list[dict]): The queries of the dataset.
        """
        check = self.check_dataset(dataset_name)
        if not check:
            print("\nDataset not found. Downloading dataset...")
            self.download_dataset(dataset_name)
        
        # Returning the corpus and queries
        print("\nLoading dataset from {}".format(os.path.join(self.dataset_path, dataset_name)))
        return self.load_data(os.path.join(self.dataset_path, dataset_name))


    def get_relevants(self, dataset_name):
        """Get the ground truth for the queries.
        
        Returns:
            ground_truth (dict): The ground truth for the queries.
        """
        ground_truth = {}

        # check if dataset_name is in config
        if not self.check_dataset(dataset_name):
            print("\nDataset not found. Calling get_dataset()...")
            self.get_dataset(dataset_name)
            
        tsv_file_path = os.path.join(self.dataset_path, f'{dataset_name}/qrels/train.tsv')

        with open(tsv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader, None)  # Skip the header

            for row in reader:
                query_id, corpus_id, score = row
                if query_id not in ground_truth:
                    ground_truth[query_id] = []
                ground_truth[query_id].append((corpus_id, int(score)))

        return ground_truth
        
        
if __name__ == '__main__':
    D = DataLoader()
    D.download_dataset()