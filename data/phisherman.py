"""
OogaBooga script to load the phishing dataset
"""
# We'll make everything as pydantic as possible
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import re

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

    def GetId(self):
        return self.Id
    
    def GetText(self):
        return self.text
    
    def __getitem__(self, key):
        return getattr(self, key)

class PhishingDataset:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = self.__BuildDocuments(documents)
        
    def __BuildDocuments(self, documents):
        docs = []
        for _, document in enumerate(documents):
            docs.append(PhishingEmail(text=clean_text(document["Email Text"]),
                        label = document["Email Type"],
                        Id = _))

        return docs
         
    def GetDocuments(self):
        return self.documents
    
    def __getitem__(self, key):
        return self.documents[key]
    
    def __len__(self):
        return len(self.documents)
    
    def getRelatedDocuments(self, document):

        """ We define relevant documents as documents with the same label,
        except for the document itself """
        return [doc for doc in self.documents if doc.label == document.label and doc.text != document.text]

def LoadPhishingDataset(path: str = "datasets/Phishing_Email.csv") -> PhishingDataset:
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"])
    phishing_dataset = PhishingDataset(df.to_dict("records"))
    return phishing_dataset

if __name__ == "__main__":
    LoadPhishingDataset()
