class AttrDict(dict):
    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if self.__dict__[AttrDict.IMMUTABLE]:
            raise AttributeError('Attempted to set "{}" to "{}", but AttrDict is immutable'.format(key, value))

        if isinstance(value, dict):
            value = AttrDict(value)

        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def set_immutable(self):
        self.__dict__[AttrDict.IMMUTABLE] = True

        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                self.set_immutable()

        for v in self.values():
            if isinstance(v, AttrDict):
                self.set_immutable()

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]
