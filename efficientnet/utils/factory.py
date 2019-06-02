def get_factory(obj):

    class Factory(object):

        @staticmethod
        def create(*args, **kwargs):
            name = kwargs.pop('name')
            return getattr(obj, name)(*args, **kwargs)

    return Factory
