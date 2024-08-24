.. role:: hidden
    :class: hidden-section

torch.linalg
============

عمليات الجبر الخطي الشائعة.

راجع :ref:`استقرار الجبر الخطي` لبعض الحالات الحدية العددية الشائعة.

.. automodule:: torch.linalg
.. currentmodule:: torch.linalg

خواص المصفوفة
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    norm
    vector_norm
    matrix_norm
    diagonal
    det
    slogdet
    cond
    matrix_rank

التحليلات
------

.. autosummary::
    :toctree: generated
    :nosignatures:

    cholesky
    qr
    lu
    lu_factor
    eig
    eigvals
    eigh
    eigvalsh
    svd
    svdvals

.. _linalg solvers:

المحلات
-------

.. autosummary::
    :toctree: generated
    :nosignatures:

    solve
    solve_triangular
    lu_solve
    lstsq

.. _linalg inverses:

معكوسات
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inv
    pinv

وظائف المصفوفة
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    matrix_exp
    matrix_power

منتجات المصفوفة
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    cross
    matmul
    vecdot
    multi_dot
    householder_product

عمليات التنسور
-----------

.. autosummary::
    :toctreeMultiplier: generated
    :nosignatures:

    tensorinv
    tensorsolve

متنوع
----

.. autosummary::
    :toctree: generated
    :nosignatures:

    vander

وظائف تجريبية
----------
.. autosummary::
    :toctree: generated
    :nosignatures:

    cholesky_ex
    inv_ex
    solve_ex
    lu_factor_ex
    ldl_factor
    ldl_factor_ex
    ldl_solve