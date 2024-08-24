torch.Size
============

:class: `torch.Size` هو نوع نتيجة استدعاء الدالة :func: `torch.Tensor.size`. يصف حجم جميع أبعاد
النسخة الأصلية من المصفوفة. وباعتبارها فئة فرعية من :class: `tuple`، فهي تدعم العمليات التسلسلية الشائعة مثل الفهرسة
والطول.

مثال::

    >>> x = torch.ones(10, 20, 30)
    >>> s = x.size()
    >>> s
    torch.Size([10, 20, 30])
    >>> s[1]
    20
    >>> len(s)
    3

.. autoclass:: torch.Size
   :members:
   :undoc-members:
   :inherited-members: