class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class EmbodiedTask:
    _env = None
    _dataset = None

    # TODO(akadian): Add agent attribute to be defined by subclasses
    def episodes(self, *args):
        r"""Returns dataloader for episodes for the EmbodiedTask.
        """
        raise NotImplementedError

    # TODO(akadian): Add distributed episodes loader.
    def seed(self, seed):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
