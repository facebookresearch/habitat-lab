import json

try:
    from _json import \
        encode_basestring_ascii
except ImportError:
    encode_basestring_ascii = None
try:
    from _json import encode_basestring
except ImportError:
    encode_basestring = None


class DatasetFloatJSONEncoder(json.JSONEncoder):
    """
        JSON Encoder that set float precision for space saving purpose.
    """
    # Version of JSON library that encoder is compatible with.
    __version__ = '2.0.9'

    def default(self, object):
        return object.__dict__

    # Overriding method to inject own `_repr` function for floats with needed
    # precision.
    def iterencode(self, o, _one_shot=False):

        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(o, allow_nan=self.allow_nan,
                     _repr=lambda x: format(x, '.5f'), _inf=float('inf'),
                     _neginf=-float('inf')):
            if o != o:
                text = 'NaN'
            elif o == _inf:
                text = 'Infinity'
            elif o == _neginf:
                text = '-Infinity'
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))

            return text

        _iterencode = json.encoder._make_iterencode(
            markers, self.default, _encoder, self.indent, floatstr,
            self.key_separator, self.item_separator, self.sort_keys,
            self.skipkeys, _one_shot)
        return _iterencode(o, 0)
