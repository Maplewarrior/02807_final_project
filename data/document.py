
class Document:
    def __init__(self, title: str, text: str, id: str) -> None:
        self.title = title
        self.text = text
        self.id = id
        
    def GetId(self):
        return self.id
    
    def GetText(self):
        return self.title + " " + self.text