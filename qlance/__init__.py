try:
    from ._version import version as __version__
except ModuleNotFoundError:
    try:
        import setuptools_scm
        __version__ = setuptools_scm.get_version(fallback_version='?.?.?')
    except (ModuleNotFoundError, TypeError, LookupError):
        __version__ = '?.?.?'
