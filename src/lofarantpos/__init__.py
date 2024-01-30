try:
    import importlib.metadata

    __version__ = importlib.metadata.version("lofarantpos")
except ModuleNotFoundError:
    __version__ = "Unknown (python 3.7 does not have importlib.metadata)"
