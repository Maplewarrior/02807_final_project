
class Document:
    def __init__(self, text: str, id: str) -> None:
        self.text = text
        self.id = id
        
    def GetId(self):
        return self.id
    
    def GetText(self):
        return self.text