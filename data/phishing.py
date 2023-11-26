"""
OogaBooga script to load the phishing dataset
"""
# We'll make everything as pydantic as possible
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import re
import uuid
import numpy as np

np.random.seed(42) # set seed for reproducibility when shuffling data

# We get the phishing data on the form:
# ,Email Text,Email Type

def clean_text(text) -> str:
    """ Clean the text of the email """
    # remove everything that is not a letter or a space
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z ]", "", text)
    # remove multiple spaces
    text = re.sub(r" +", " ", text)
    # remove leading and trailing spaces
    text = text.strip()
    # make everything lowercase
    text = text.lower()
    return text

class PhishingEmail(BaseModel):
    """ Check if email_text is str else convert to str """
    Id: Optional[int] = None
    _id = Id
    text: str
    label: str
    title: Optional[str] = ""
    
    def GetTitle(self):
        return self.title

    def GetId(self):
        return self.Id
    
    def GetText(self):
        return self.text
    
    def GetLabel(self):
        return self.label
    
    def __getitem__(self, key):
        return getattr(self, key)

class PhishingDataset:
    def __init__(self, documents: list[dict]) -> None:
        self.documents, self.ids_to_labels = self.__BuildDocuments(documents)
        
    def __BuildDocuments(self, documents):
        ids = [uuid.uuid4().int for _ in documents]
        ids_to_labels = {ids[i]: document["Email Type"] for i,document in enumerate(documents)}
        return [PhishingEmail(text=clean_text(document["Email Text"]),
                        label = document["Email Type"],
                        Id = ids[i]) for i,document in enumerate(documents)], ids_to_labels
        
    def Shuffle(self):
        np.random.shuffle(self.documents)
         
    def GetDocuments(self):
        return self.documents
    
    def GetLabelFromId(self, id: str):
        return self.ids_to_labels[id]
    
    def GetDocumentDicts(self):
        return [
            {
                "title": document.GetTitle(),
                "text": document.GetText(),
                "_id": document.GetId()
            } for document in self.documents
        ]
    
    def __getitem__(self, key) -> PhishingEmail:
        return self.documents[key]
    
    def __len__(self):
        return len(self.documents)
    
    def getRelatedDocuments(self, document):

        """ We define relevant documents as documents with the same label,
        except for the document itself """
        return [doc for doc in self.documents if doc.label == document.label and doc.text != document.text]

def LoadPhishingDataset(path: str = "data/datasets/Phishing_Email.csv") -> PhishingDataset:
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"])
    phishing_dataset = PhishingDataset(df.to_dict("records"))
    return phishing_dataset

if __name__ == "__main__":
    df = pd.read_csv("datasets/Phishing_Email.csv").drop(columns=["Unnamed: 0"])
    # check balance in labels
    print(df["Email Type"].value_counts())
