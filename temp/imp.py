from cls import ori

class changed(ori):
    def load(self):
        print(2)

aha = ori()
aha.load()

ohou = changed()
ohou.load()

