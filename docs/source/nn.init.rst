.. role:: hidden
    :class: hidden-section

.. _nn-init-doc:

torch.nn.init
=============

.. warning::
    جميع الوظائف في هذا الوحدة مصممة لتستخدم في تهيئة معاملات الشبكة العصبية، لذا فهي تعمل جميعها في وضع :func:`torch.no_grad` ولن تؤخذ في الاعتبار بواسطة autograd.

.. currentmodule:: torch.nn.init
.. autofunction:: calculate_gain
.. autofunction:: uniform_
.. autofunction:: normal_
.. autofunction:: constant_
.. autofunction:: ones_
.. autofunction:: zeros_
.. autofunction:: eye_
.. autofunction:: dirac_
.. autofunction:: xavier_uniform_
.. autofunction:: xavier_normal_
.. autofunction:: kaiming_uniform_
.. autofunction:: kaim
.. autofunction:: trunc_normal_
.. autofunction:: orthogonal_
.. autofunction:: sparse_