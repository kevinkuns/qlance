"""
Requires pytest to import
"""
import os
from os import path
import os
from shutil import rmtree
import contextlib

import pytest

_options_added = False

def pytest_addoption(parser):
    global _options_added
    #this check fixes issues if this gets run multiple times from sub conftest.py's
    if _options_added:
        return
    else:
        _options_added = True

    parser.addoption(
        "--plot",
        action="store_true",
        dest = 'plot',
        help = "Have tests update plots (it is slow)",
    )

    parser.addoption(
        "--do-stresstest",
        action = "store_true",
        help   = "Run slow repeated stress tests"
    )

    parser.addoption(
        "--no-preclear",
        action="store_true",
        default=False,
        dest='no_preclear',
        help="Do not preclear tpaths",
    )


@pytest.fixture
def plot(request):
    return request.config.getvalue('--plot')
    return request.config.option.plot


@pytest.fixture
def tpath_preclear(request):
    """
    Fixture that indicates that the test path should be cleared automatically
    before running each test. This cleans up the test data.
    """
    tpath_raw = tpath_raw_make(request)
    no_preclear = request.config.getvalue('--no-preclear')
    if not no_preclear:
        rmtree(tpath_raw, ignore_errors = True)


@pytest.fixture
def tpath(request):
    """
    Fixture that takes the value of the special test-specific folder for test
    run data and plots. Usually the <folder of the test>/tresults/test_name/
    """
    tpath_raw = tpath_raw_make(request)

    os.makedirs(tpath_raw, exist_ok = True)
    os.utime(tpath_raw, None)
    return tpath_raw


@pytest.fixture
def tpath_join(request):
    """
    Fixture that joins subpaths to the value of the special test-specific folder for test
    run data and plots. Usually the <folder of the test>/tresults/test_name/.

    This function should be use like test_thing.save(tpath_join('output_file.png'))
    """
    tpath_raw = tpath_raw_make(request)
    first_call = True

    def tpath_joiner(*subpath):
        nonlocal first_call
        if first_call:
            os.makedirs(tpath_raw, exist_ok = True)
            os.utime(tpath_raw, None)
            first_call = False
        return path.join(tpath_raw, *subpath)

    return tpath_joiner


@pytest.fixture
def fpath(request):
    """
    py.test fixture that returns the folder path of the test being run. Useful
    for accessing data files.
    """
    return fpath_raw_make(request)


@pytest.fixture
def fpath_join(request):
    """
    py.test fixture that runs :code:`os.path.join(path, *arguments)` to merge subpaths
    with the folder path of the current test being run. Useful for referring to
    data files.
    """
    def join_func(*path):
        return os.path.join(fpath_raw_make(request), *path)
    return join_func

@pytest.fixture
def closefigs():
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


@pytest.fixture
def test_trigger():
    """
    This fixture provides a contextmanager that causes a function to call
    if an AssertionError is raised. It will also call if any of its argument,
    or keyword arguments is true. This allows you to conveniently force
    calling using other flags or fixtures.

    The primary usage of this is to plot outputs only on test failures, while also
    allowing plotting to happen using the plot fixture and pytest cmdline argument
    """
    run_store = []

    @contextlib.contextmanager
    def fail(call, **kwargs):
        run_store.append(call)

        def call(did_fail):
            do_call = did_fail
            for k, v in kwargs.items():
                if v:
                    do_call = True
                    break

            if do_call:
                for call in run_store:
                    call(fail = did_fail, **kwargs)
                run_store.clear()
        try:
            yield
        except AssertionError:
            call(True)
            raise
        else:
            call(False)

        return
    return fail


@pytest.fixture()
def ic():
    """
    Fixture to provide icecream imports without requiring that the package exist
    """
    try:
        from icecream import ic
        return ic
    except ImportError:
        pass
    try:
        from IPython.lib.pretty import pprint
        return pprint
    except ImportError:
        from pprint import pprint
        return pprint


#these are used with the pprint fixture
try:
    import icecream
except ImportError:
    icecream = None
    pass
try:
    from IPython.lib.pretty import pprint, pretty
    pformat = pretty
except ImportError:
    from pprint import pprint, pformat


@pytest.fixture
def pprint(request, tpath_join):
    """
    This is a fixture providing a wrapper function for pretty printing. It uses
    the icecream module for pretty printing, falling back to ipythons pretty
    printer if needed, then to the python build in pretty printing module.

    Along with printing to stdout, this function prints into the tpath_folder to
    save all output into output.txt.
    """
    fname = tpath_join('output.txt')

    #pushes past the dot
    print('---------------:{}:--------------'.format(request.node.name))
    with open(fname, 'w') as F:
        def pprint(*args, F = F, pretty = True, **kwargs):
            outs = []
            if pretty:
                for arg in args:
                    outs.append(
                        pformat(arg)
                    )
            else:
                outs = args
            if F is not None:
                print(*outs, file = F)
            if icecream is not None:
                icecream.DEFAULT_OUTPUT_FUNCTION(' '.join(outs), **kwargs)
            else:
                print(*outs, **kwargs)

        yield pprint


def tpath_raw_make(request):
    if isinstance(request.node, pytest.Function):
        return relfile_test(request.node.function.__code__.co_filename, request, 'tresults')
    raise RuntimeError("TPath currently only works for functions")


def fpath_raw_make(request):
    if isinstance(request.node, pytest.Function):
        return os.path.split(request.node.function.__code__.co_filename)[0]
    raise RuntimeError("TPath currently only works for functions")


def relfile(_file_, *args, fname = None):
    fpath = path.split(_file_)[0]
    post = path.join(*args)
    fpath = path.join(fpath, post)
    #os.makedirs(fpath, exist_ok = True)
    #os.utime(fpath, None)

    if fname is None:
        return fpath
    else:
        return path.join(fpath, fname)


def relfile_test(_file_, request, pre = None, post = None, fname = None):
    """
    Generates a folder specific to py.test function
    (provided by using the "request" fixture in the test's arguments)
    """
    if isinstance(pre, (list, tuple)):
        pre = path.join(pre)

    testname = request.node.name
    if pre is not None:
        testname = path.join(pre, testname)

    if isinstance(post, (list, tuple)):
        post = path.join(post)
    if post is not None:
        return relfile(_file_, testname, post, fname = fname)
    else:
        return relfile(_file_, testname, fname = fname)

