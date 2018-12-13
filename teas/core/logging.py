import logging


class TeasLogger(logging.Logger):
    def __init__(self, name, level, filename=None, filemode='a', stream=None,
                 format=None, dateformat=None,
                 style='%'):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)
        else:
            handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(format, dateformat, style)
        handler.setFormatter(formatter)
        super().addHandler(handler)


logger = TeasLogger(name='teas', level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
