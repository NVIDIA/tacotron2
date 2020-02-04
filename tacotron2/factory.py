from pydoc import locate


class Factory:
    @staticmethod
    def get_object(class_path, *args, **kwargs):
        cls = locate(class_path)
        obj = cls(*args, **kwargs)
        return obj

    @staticmethod
    def get_class(class_path):
        cls = locate(class_path)
        if cls is None:
            raise ValueError('Could not find class: {}'.format(class_path))
        return cls
