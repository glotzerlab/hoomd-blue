
class Groups(dict):
    def __setitem__(self, key, value):
        if key in self:
            return self[key]
        else:
            return super().__setitem__(key, value)
