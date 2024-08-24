.. currentmodule:: الشعلة

.. _tensor-attributes-doc:

خصائص Tensor
=================

يمتلك كل ``torch.Tensor`` :class:`نوع بيانات <torch.dtype>`، و:class:`جهاز <torch.device>`، و:class:`تخطيط <torch.layout>`.

.. _dtype-doc:

torch.dtype
-----------

.. class:: dtype

:class:`torch.dtype` هو كائن يمثل نوع البيانات لـ :class:`torch.Tensor`. يوفر PyTorch اثني عشر نوعًا مختلفًا من البيانات:

========================== ===========================================   ===========================
نوع البيانات                  dtype                                         البناة التراثية
========================== ===========================================   ===========================
نقطة عائمة 32-بت      ``torch.float32`` أو ``torch.float``          ``torch.*.FloatTensor``
نقطة عائمة 64-بت      ``torch.float64`` أو ``torch.double``         ``torch.*.DoubleTensor``
معقد 64-بت             ``torch.complex64`` أو ``torch.cfloat``
معقد 128-بت            ``torch.complex128`` أو ``torch.cdouble``
نقطة عائمة 16-بت [1]_ ``torch.float16`` أو ``torch.half``           ``torch.*.HalfTensor``
نقطة عائمة 16-بت [2]_ ``torch.bfloat16``                            ``torch.*.BFloat16Tensor``
عدد صحيح 8-بت (بدون إشارة)   ``torch.uint8``                               ``torch.*.ByteTensor``
عدد صحيح 8-بت (موقّع)     ``torch.int8``                                ``torch.*.CharTensor``
عدد صحيح 16-بت (موقّع)    ``torch.int16`` أو ``torch.short``            ``torch.*.ShortTensor``
عدد صحيح 32-بت (موقّع)    ``torch.int32`` أو ``torch.int``              ``torch.*.IntTensor``
عدد صحيح 64-بت (موقّع)    ``torch.int64`` أو ``torch.long``             ``torch.*.LongTensor``
منطقي                    ``torch.bool``                                ``torch.*.BoolTensor``
========================== ===========================================   ===========================

.. [1] يشار إليه أحيانًا باسم binary16: يستخدم 1 بت للإشارة، و5 للأس، و10 للمقام. مفيد عندما تكون الدقة مهمة.

.. [2] يشار إليه أحيانًا باسم Brain Floating Point: يستخدم 1 بت للإشارة، و8 للأس، و7 للمقام. مفيد عندما يكون النطاق مهمًا، حيث أن له نفس عدد بتات الأس مثل ``float32``

للتحقق مما إذا كان :class:`torch.dtype` هو نوع بيانات النقطة العائمة، يمكن استخدام الخاصية :attr:`is_floating_point`، والتي تعيد ``True`` إذا كان نوع البيانات هو نوع بيانات النقطة العائمة.

للتحقق مما إذا كان :class:`torch.dtype` هو نوع بيانات معقد، يمكن استخدام الخاصية :attr:`is_complex`، والتي تعيد ``True`` إذا كان نوع البيانات هو نوع بيانات معقد.

.. _type-promotion-doc:

عندما تختلف أنواع بيانات المدخلات لعملية حسابية (`add`، `sub`، `div`، `mul`)، نقوم بالترقية عن طريق إيجاد أصغر نوع بيانات يفي بالقواعد التالية:

* إذا كان نوع بيانات كمية سلمية أعلى من نوع بيانات الكميات المتناسقة (حيث أن المعقد > العائم > الصحيح > المنطقي)، فإننا نرقي إلى نوع له حجم كافٍ لاحتواء جميع الكميات السلمية من تلك الفئة.
* إذا كان لكمية سلمية صفرية الأبعاد نوع بيانات أعلى من الكميات المتناسقة ذات الأبعاد، فإننا نرقي إلى نوع له حجم وفئة كافيين لاحتواء جميع الكميات السلمية الصفرية الأبعاد من تلك الفئة.
* إذا لم تكن هناك كميات سلمية صفرية الأبعاد ذات أولوية أعلى، فإننا نرقي إلى نوع له حجم وفئة كافيين لاحتواء جميع الكميات المتناسقة ذات الأبعاد.

تكون القيمة الافتراضية لنوع بيانات الكمية السلمية العائمة ``torch.get_default_dtype()`` وللكمية السلمية الصحيحة غير المنطقية ``torch.int64``. على عكس NumPy، فإننا لا نتفقد القيم عند تحديد أصغر أنواع بيانات الكميات السلمية. لا يتم دعم الأنواع الكمية المعقدة بعد.

أمثلة الترقية::

    >>> float_tensor = torch.ones(1, dtype=torch.float)
    >>> double_tensor = torch.ones(1, dtype=torch.double)
    >>> complex_float_tensor = torch.ones(1, dtype=torch.complex64)
    >>> complex_double_tensor = torch.ones(1, dtype=torch.complex128)
    >>> int_tensor = torch.ones(1, dtype=torch.int)
    >>> long_tensor = torch.ones(1, dtype=torch.long)
    >>> uint_tensor = torch.ones(1, dtype=torch.uint8)
    >>> double_tensor = torch.ones(1, dtype=torch.double)
    >>> bool_tensor = torch.ones(1, dtype=torch.bool)
    # الكميات السلمية الصفرية الأبعاد
    >>> long_zerodim = torch.tensor(1, dtype=torch.long)
    >>> int_zerodim = torch.tensor(1, dtype=torch.int)

    >>> torch.add(5, 5).dtype
    torch.int64
    # 5 هي كمية سلمية من النوع int64، ولكنها لا تمتلك فئة أعلى من int_tensor لذا لا يتم اعتبارها.
    >>> (int_tensor + 5).dtype
    torch.int32
    >>> (int_tensor + long_zerodim).dtype
    torch.int32
    >>> (long_tensor + int_tensor).dtype
    torch.int64
    >>> (bool_tensor + long_tensor).dtype
    torch.int64
    >>> (bool_tensor + uint_tensor).dtype
    torch.uint8
    >>> (float_tensor + double_tensor).dtype
    torch.float64
    >>> (complex_float_tensor + complex_double_tensor).dtype
    torch.complex128
    >>> (bool_tensor + int_tensor).dtype
    torch.int32
    # نظرًا لأن long من نوع مختلف عن float، فإن نوع نتيجة العملية لا يحتاج إلا أن يكون كبيرًا بما يكفي
    # لاحتواء float.
    >>> torch.add(long_tensor, float_tensor).dtype
    torch.float32

عندما يتم تحديد الكمية المتناسقة الناتجة عن عملية حسابية، فإننا نسمح بالتحويل إلى نوع بياناتها باستثناء ما يلي:
  * لا يمكن للكمية المتناسقة الناتجة ذات النوع الصحيح أن تقبل كمية متناسقة ذات نوع بيانات عائم.
  * لا يمكن للكمية المتناسقة الناتجة ذات النوع المنطقي أن تقبل كمية متناسقة ذات نوع بيانات غير منطقي.
  * لا يمكن للكمية المتناسقة الناتجة ذات النوع غير المعقد أن تقبل كمية متناسقة ذات نوع بيانات معقد.

أمثلة التحويل::

    # مسموح:
    >>> float_tensor *= float_tensor
    >>> float_tensor *= int_tensor
    >>> float_tensor *= uint_tensor
    >>> float_tensor *= bool_tensor
    >>> float_tensor *= double_tensor
    >>> int_tensor *= long_tensor
    >>> int_tensor *= uint_tensor
    >>> uint_tensor *= int_tensor

    # غير مسموح (RuntimeError: لا يمكن تحويل نوع النتيجة إلى نوع الإخراج المطلوب):
    >>> int_tensor *= float_tensor
    >>> bool_tensor *= int_tensor
    >>> bool_tensor *= uint_tensor
    >>> float_tensor *= complex_float_tensor


.. _device-doc:

torch.device
------------

.. class:: device

:class:`torch.device` هو كائن يمثل الجهاز الذي تم أو سيتم تخصيص :class:`torch.Tensor` عليه.

يحتوي :class:`torch.device` على نوع جهاز (غالبًا ما يكون "cpu" أو
"cuda"، ولكن أيضًا قد يكون :doc:`"mps" <mps>`، أو :doc:`"xpu" <xpu>`،
أو `"xla" <https://github.com/pytorch/xla/>`_ أو :doc:`"meta" <meta>`) ورقم جهاز اختياري لنوع الجهاز. إذا لم يكن رقم الجهاز موجودًا، فسيُمثل هذا الكائن دائمًا الجهاز الحالي لنوع الجهاز، حتى بعد استدعاء :func:`torch.cuda.set_device()`؛ على سبيل المثال،
تكون الكمية المتناسقة المُنشأة بجهاز ``'cuda'`` مكافئة لـ ``'cuda:X'`` حيث X هي
نتيجة :func:`torch.cuda.current_device()`.

يمكن الوصول إلى جهاز الكمية المتناسقة عبر الخاصية :attr:`Tensor.device`.

يمكن إنشاء :class:`torch.device` عبر سلسلة أو عبر سلسلة ورقم جهاز

عبر سلسلة:
::

    >>> torch.device('cuda:0')
    device(type='cuda', index=0)

    >>> torch.device('cpu')
    device(type='cpu')

    >>> torch.device('mps')
    device(type='mps')

    >>> torch.device('cuda')  # جهاز cuda الحالي
    device(type='cuda')

عبر سلسلة ورقم جهاز:

::

    >>> torch.device('cuda', 0)
    device(type='cuda', index=0)

    >>> torchMultiplier.device('mps', 0)
    device(type='mps', index=0)

    >>> torch.device('cpu', 0)
    device(type='cpu', index=0)

يمكن أيضًا استخدام كائن الجهاز كسياق للتحكم في تغيير الجهاز الافتراضي
الذي يتم تخصيص الكميات المتناسقة عليه:

::

    >>> with torch.device('cuda:1'):
    ...     r = torch.randn(2, 3)
    >>> r.device
    device(type='cuda', index=1)

ليس لهذا السياق أي تأثير إذا تم تمرير دالة مصنع وسيطة جهاز صريحة وغير فارغة.  لرؤية كيفية التغيير الشامل للجهاز الافتراضي، راجع أيضًا
:func:`torch.set_default_device`.

.. warning::

    تفرض هذه الدالة تكلفة أداء طفيفة على كل استدعاء Python
    لواجهة برمجة تطبيقات PyTorch (ليس فقط لوظائف المصنع).  إذا كان
    هذا يسبب مشاكل لك، يرجى التعليق على
    https://github.com/pytorch/pytorch/issues/92701

.. note::
   يمكن استبدال وسيطة :class:`torch.device` في الدوال بشكل عام بسلسلة.
   يسمح ذلك بالتصميم السريع لبروتوكول التشفير.

   >>> # مثال لوظيفة تأخذ وسيطة torch.device
   >>> cuda1 = torch.device('cuda:1')
   >>> torch.randn((2,3), device=cuda1)

   >>> # يمكنك استبدال وسيطة torch.device بسلسلة
   >>> torch.randn((2,3), device='cuda:1')

.. note::
   لأسباب تتعلق بالتوافق مع الإصدارات الأقدم، يمكن إنشاء جهاز عبر رقم جهاز واحد، والذي يتم التعامل معه
   كنوع :ref:`accelerator<accelerators>` الحالي.
   وهذا يتوافق مع :meth:`Tensor.get_device`، والذي يعيد رقمًا لكميات tensor ذات الجهاز
   ولا يتم دعمه لكميات tensor ذات الجهاز cpu.

   >>> torch.device(1)
   device(type='cuda', index=1)

.. note::
   تقبل الطرق التي تأخذ جهازًا بشكل عام (سلسلة مُنسقة بشكل صحيح) أو (رقم جهاز صحيح) كوسيطة للجهاز، أي أن ما يلي متكافئ:

   >>> torch.randn((2,3), device=torch.device('cuda:1'))
   >>> torch.randn((2,3), device='cuda:1')
   >>> torch.randn((2,3), device=1)  # الإصدار الأقدم

.. note::
   لا يتم نقل الكميات المتناسقة مطلقًا تلقائيًا بين الأجهزة ويتطلب الأمر استدعاءً صريحًا من المستخدم. الكميات المتناسقة القياسية (مع tensor.dim()==0) هي الاستثناء الوحيد لهذه القاعدة حيث يتم نقلها تلقائيًا من CPU إلى GPU عندما تكون هناك حاجة إلى ذلك لأن هذه العملية يمكن إجراؤها "مجانًا".
   مثال:

   >>> # كميتان قياسيتان
   >>> torch.ones(()) + torch.ones(()).cuda()  # موافق، يتم نقل الكمية المتناسقة القياسية تلقائيًا من CPU إلى GPU
   >>> torch.ones(()).cuda() + torch.ones(())  # موافق، يتم نقل الكمية المتناسقة القياسية تلقائيًا من CPU إلى GPU

   >>> # كمية متناسقة قياسية واحدة (CPU)، وكمية متناسقة ناقلة واحدة (GPU)
   >>> torch.ones(()) + torch.ones(1).cuda()  # موافق، يتم نقل الكمية المتناسقة القياسية تلقائيًا من CPU إلى GPU
   >>> torch.ones(1).cuda() + torch.ones(())  # موافق، يتم نقل الكمية المتناسقة القياسية تلقائيًا من CPU إلى GPU

   >>> # كمية متناسقة قياسية واحدة (GPU)، وكمية متناسقة ناقلة واحدة (CPU)
   >>> torch.ones(()).cuda() + torch.ones(1)  # فشل، لا يتم نقل الكمية المتناسقة القياسية تلقائيًا من GPU إلى CPU ولا يتم نقل الكمية المتناسقة غير القياسية تلقائيًا من CPU إلى GPU
   >>> torch.ones(1) + torch.ones(()).cuda()  # فشل، لا يتم نقل الكمية المتناسقة القياسية تلقائيًا من GPU إلى CPU ولا يتم نقل الكمية المتناسقة غير القياسية تلقائيًا من CPU إلى GPU

.. _layout-doc:

torch.layout
------------

.. class:: layout

.. warning::
  ``torch.layout`` هو في مرحلة البيتا وقد يخضع للتغيير.

:class:`torch.layout` هو كائن يمثل التخطيط الذاكري لـ :class:`torch.Tensor`. حاليًا، ندعم ``torch.strided`` (كميات tensor الكثيفة)
ولدينا دعم تجريبي لـ ``torch.sparse_coo`` (كميات tensor المتناثرة COO).

يمثل ``torch.strided`` كميات tensor الكثيفة وهو تخطيط الذاكرة الذي
يتم استخدامه بشكل أكثر شيوعًا. تمتلك كل كمية tensor كثيفة مرتبطة بها
:class:`torch.Storage`، والذي يحتفظ ببياناتها. توفر هذه الكميات المتناسقة عرضًا متعدد الأبعاد، `متناسق <https://en.wikipedia.org/wiki/Stride_of_an_array>`_
للتخزين. تكون الخطوات قائمة من الأعداد الصحيحة: تمثل الخطوة k-th
القفزة الضرورية في الذاكرة للانتقال من عنصر إلى العنصر التالي في البعد k-th للكمية المتناسقة. يجعل هذا المفهوم من الممكن
أداء العديد من عمليات الكمية المتناسقة بكفاءة.

مثال::

    >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)

    >>> x.t().stride()
    (1, 5)

للحصول على مزيد من المعلومات حول ``torch.sparse_coo`` tensors، راجع :ref:`sparse-docs`.

torch.memory_format
.. class:: memory_format

إن :class:`torch.memory_format` هو كائن يمثل تنسيق الذاكرة الذي تم أو سيتم تخصيص :class:`torch.Tensor` له.

القيم الممكنة هي:

- ``torch.contiguous_format``:
  سيتم تخصيص نسيج أو تم تخصيصه في ذاكرة غير متداخلة كثيفة. الخطوات التي تمثلها القيم بترتيب تنازلي.

- ``torch.channels_last``:
  سيتم تخصيص نسيج أو تم تخصيصه في ذاكرة غير متداخلة كثيفة. الخطوات التي تمثلها القيم في
  ``strides[0] > strides[2] > strides[3] > strides[1] == 1`` المعروف باسم ترتيب NHWC.

- ``torch.channels_last_3d``:
  سيتم تخصيص نسيج أو تم تخصيصه في ذاكرة غير متداخلة كثيفة. الخطوات التي تمثلها القيم في
  ``strides[0] > strides[2] > strides[3] > strides[4] > strides[1] == 1`` المعروف باسم ترتيب NDHWC.

- ``torch.preserve_format``:
  يستخدم في وظائف مثل 'clone' للحفاظ على تنسيق الذاكرة للنسيج المدخلات. إذا تم تخصيص نسيج الإدخال
  في ذاكرة غير متداخلة كثيفة، سيتم نسخ خطوات الإخراج من الإدخال.
  وإلا فإن الخطوات الإخراجية ستتبع ``torch.contiguous_format``