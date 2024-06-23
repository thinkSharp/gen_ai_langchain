
class factory:
    def __init__(self):
        self._creators = {}

    def register_creator(self, key, creator):
        self._creators[key] = creator

    def get_instance(self, key, *args, **kwargs):
        creator = self._creators.get(key)
        if not creator:
            raise ValueError(f"Creator not found for key: {key}")
        return creator(*args, **kwargs)
