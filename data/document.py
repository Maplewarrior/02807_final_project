
class Document:
    def __init__(self, title: str, text: str, _id: str) -> None:
        self.title = title
        self.text = text
        self._id = _id
        
    def GetId(self):
        return self._id
    
    def GetText(self):
        return self.title + " " + self.text