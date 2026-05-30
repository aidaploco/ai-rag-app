from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rag-sdk")
except PackageNotFoundError:
    __version__ = "unknown"
