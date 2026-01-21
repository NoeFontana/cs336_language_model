import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"
