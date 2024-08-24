torch.Storage
===============

:class:`torch.Storage` هو مرادف لفئة التخزين التي تتوافق مع نوع البيانات الافتراضي (:func:`torch.get_default_dtype()`). على سبيل المثال، إذا كان نوع البيانات الافتراضي هو :attr:`torch.float`، فإن :class:`torch.Storage` يحل محل :class:`torch.FloatStorage`.

فئات :class:`torch.<type>Storage` و :class:`torch.cuda.<type>Storage`، مثل :class:`torch.FloatStorage`، :class:`torch.IntStorage`، وما إلى ذلك، لا يتم إنشاء مثيل منها فعليًا أبدًا. يؤدي استدعاء بنائيها إلى إنشاء :class:`torch.TypedStorage` مع :class:`torch.dtype` و :class:`torch.device` المناسبين. تحتوي فئات :class:`torch.<type>Storage` على نفس طرق الفئات التي لدى :class:`torch.TypedStorage`.

:class:`torch.TypedStorage` هو مصفوفة أحادية البعد ومتجاورة لعناصر من نوع :class:`torch.dtype` معين. يمكن إعطاؤه أي نوع :class:`torch.dtype`، وسيتم تفسير البيانات الداخلية بشكل مناسب. يحتوي :class:`torch.TypedStorage` على :class:`torch.UntypedStorage` والذي يحتفظ بالبيانات كمصفوفة غير معلمة من البايتات.

يحتوي كل :class:`torch.Tensor` متعدد الخطوات على :class:`torch.TypedStorage`، والذي يخزن جميع البيانات التي يعرضها :class:`torch.Tensor`.

.. warning::
  سيتم إزالة جميع فئات التخزين باستثناء :class:`torch.UntypedStorage` في المستقبل، وسيتم استخدام :class:`torch.UntypedStorage` في جميع الحالات.

.. autoclass:: torch.TypedStorage
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: torch.UntypedStorage
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: torch.DoubleStorage
   :members:
   :undoc-members:

.. autoclass:: torch.FloatStorage
   :members:
   :undoc-members:

.. autoclass:: torch.HalfStorage
   :members:
   :undoc-members:

.. autoclass:: torch.LongStorage
   :members:
   :undoc-members:

.. autoclass:: torch.IntStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ShortStorage
   :members:
   :undoc-members:

.. autoclass:: torch.CharStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ByteStorage
   :members:
   :undoc-members:

.. autoclass:: torch.BoolStorage
   :members:
   :undoc-members:

.. autoclass:: torch.BFloat16Storage
   :members:
   :undoc-members:

.. autoclass:: torch.ComplexDoubleStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ComplexFloatStorage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt8Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QInt8Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QInt32Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt4x2Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt2x4Storage
   :members:
   :undoc-members: