
class Document:
    def __init__(self, text: str, id: str) -> None:
        self.text = text
        self._id = id
        
    def GetId(self):
        return self._id
    
    def GetText(self):
        return self.text