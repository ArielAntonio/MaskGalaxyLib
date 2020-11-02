import sys

def _raise_build_error(e):
    # Raise a comprehensible error
    import os.path as osp
    local_dir = osp.split(__file__)[0]
    msg = _STANDARD_MSG
    if local_dir == "MaskGalaxy":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = _INPLACE_MSG
    raise ImportError("""%s
It seems that MaskGalaxy has not been built correctly.
%s""" % (e, msg))