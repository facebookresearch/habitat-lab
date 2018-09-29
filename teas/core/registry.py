class Spec:
    def __init__(self, id, entry_point, **kwargs):
        self.id = id
        self._entry_point = entry_point
    
    def make(self, **kwargs):
        return self._entry_point(**kwargs)
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.id)


class Registry:
    def __init__(self):
        self.specs = {}
    
    def make(self, id, **kwargs):
        spec = self.get_spec(id)
        return spec.make(**kwargs)
    
    def all(self):
        return self.specs.values()
    
    def get_spec(self, id):
        spec = self.specs.get(id, None)
        if spec is None:
            raise KeyError("No registered specification with id: {}".format(id))
        return spec
    
    def register(self, id, **kwargs):
        raise NotImplementedError
