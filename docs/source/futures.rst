.. currentmodule:: torch.futures

.. _futures-docs:

torch.futures
=============

توفر هذه الحزمة نوعًا :class:`~torch.futures.Future` يغلف عملية تنفيذ غير متزامنة ومجموعة من وظائف المنفعة لتبسيط العمليات على كائنات :class:`~torch.futures.Future`. حاليًا، يتم استخدام نوع :class:`~torch.futures.Future` بشكل أساسي بواسطة :ref:`distributed-rpc-framework`.

.. automodule:: torch.futures

.. autoclass:: Future
    :inherited-members:

.. autofunction:: collect_all
.. autofunction:: wait_all