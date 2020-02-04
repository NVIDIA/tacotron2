from pathlib import Path

import yaml


class HParams(dict):
    """
    Example:
    m = HParams({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(HParams, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(HParams, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(HParams, self).__delitem__(key)
        del self.__dict__[key]

    @classmethod
    def from_yaml(cls, file_path: Path):
        file_path = Path(file_path)
        with file_path.open() as file:
            dict_ = yaml.load(file, Loader=yaml.FullLoader)

        return cls(dict_)