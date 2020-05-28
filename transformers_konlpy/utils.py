import unicodedata


def unicode_normalize(s):
    return unicodedata.normalize('NFD', s)
