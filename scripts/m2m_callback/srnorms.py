"""Identity normalisation for map2map's ``--in-norms`` / ``--tgt-norms``.

Our boxes are ALREADY in ``map2map.data.norms.cosmology.dis`` space. That norm
is ``x /= dis_std * D(z)`` with ``dis_std = 6000`` kpc/h, and ``D(0) = 1``, so
at z = 0 it divides by exactly the 6000 kpc/h our ``paired_catnorm`` boxes were
written with. Passing ``--in-norms cosmology.dis`` would therefore normalise a
second time and shrink every displacement by 6000x.

Reached through ``--callback-at``, which is map2map's own documented mechanism
for user-supplied norms (``utils/imp.py::import_attr``), so using it is not an
edit to their code.

The contract, from ``data/fields.py``::

    norm = import_attr(norm, norms, callback_at=self.callback_at)
    norm(x, a=style, **self.kwargs)

-- called for its SIDE EFFECT on ``x`` and its return value discarded, so a
no-op body is the identity. ``a`` and any other kwargs are accepted and ignored.
"""


def identity(x, undo=False, **kwargs):
    """Leave the field exactly as loaded. See the module docstring for why."""
    return
