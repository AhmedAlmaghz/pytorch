.. currentmodule:: torch

.. _named_tensors-doc:

التوابع المسماة
==========

تسمح التوابع المسماة للمستخدمين بإعطاء أسماء صريحة لأبعاد التوابع.
في معظم الحالات، تقبل العمليات التي تأخذ معلمات الأبعاد أسماء الأبعاد، مما يجنب الحاجة إلى تتبع الأبعاد حسب الموضع.
بالإضافة إلى ذلك، تستخدم التوابع المسماة الأسماء للتحقق تلقائيًا من أن واجهات برمجة التطبيقات (APIs)
يتم استخدامها بشكل صحيح في وقت التشغيل، مما يوفر أمانًا إضافيًا. يمكن أيضًا استخدام الأسماء لإعادة ترتيب الأبعاد،
على سبيل المثال، لدعم "البث حسب الاسم" بدلاً من "البث حسب الموضع".

.. warning::
    واجهة برمجة تطبيقات التوابع المسماة هي ميزة تجريبية وقد تتغير.

إنشاء التوابع المسماة
-------------

تأخذ دالات المصنع الآن وسيطًا جديدًا :attr: `names` الذي يربط اسمًا بكل بعد.

::

    >>> torch.zeros(2, 3, names=('N', 'C'))
    tensor([[0., 0., 0.],
            [0., 0., 0.]], names=('N', 'C'))

الأبعاد المسماة، مثل أبعاد التوابع العادية، مرتبة.
``tensor.names[i]`` هو اسم البعد ``i`` من ``tensor``.

دالات المصنع التالية تدعم التوابع المسماة:

- :func: `torch.empty`
- :func: `torch.rand`
- :func: `torch.randn`
- :func: `torch.ones`
- :func: `torch.tensor`
- :func: `torch.zeros`

الأبعاد المسماة
----------

راجع :attr: `~ Tensor.names` للقيود المفروضة على أسماء التوابع.

استخدم :attr: `~ Tensor.names` للوصول إلى أسماء أبعاد التابع واستخدم :meth: `~ Tensor.rename` لإعادة تسمية الأبعاد المسماة.

::

    >>> imgs = torch.randn(1, 2, 2, 3, names=('N', 'C', 'H', 'W'))
    >>> imgs.names
    ('N', 'C', 'H', 'W')

    >>> renamed_imgs = imgs.rename(H='height', W='width')
    >>> renamed_imgs.names
    ('N', 'C', 'height', 'width')

يمكن للتوابع المسماة التعايش مع التوابع غير المسماة؛ التوابع المسماة هي مثيلات من
:class: `torch.Tensor`. الأبعاد غير المسماة لها أسماء ``None``. لا تتطلب التوابع المسماة تسمية جميع الأبعاد.

::

    >>> imgs = torch.randn(1, 2, 2, 3, names=(None, 'C', 'H', 'W'))
    >>> imgs.names
    (None, 'C', 'H', 'W')

دلالة انتشار الاسم
--------------

تستخدم التوابع المسماة الأسماء للتحقق تلقائيًا مما إذا كانت واجهات برمجة التطبيقات (APIs)
يتم استدعاؤها بشكل صحيح في وقت التشغيل. يحدث هذا في عملية تسمى *استنتاج الاسم*.
بشكل أكثر رسمية، يتكون استنتاج الاسم من الخطوتين التاليتين:

- **التحقق من الأسماء**: قد يؤدي المشغل عمليات فحص تلقائية في وقت التشغيل للتأكد من أن أسماء أبعاد معينة يجب أن تتطابق.
- **نشر الأسماء**: ينشر استدلال الاسم الأسماء إلى التوابع الناتجة.

تدعم جميع العمليات التي تدعم التوابع المسماة نشر الأسماء.

::

    >>> x = torch.randn(3, 3, names=('N', 'C'))
    >>> x.abs().names
    ('N', 'C')

.. _match_semantics-doc:

دلالات المطابقة
^^^^^^^^^^^

*تتطابق* الأسماء إذا كانت متساوية (تساوي السلاسل) أو إذا كان أحدهما على الأقل ``None``.
إن "Nones" هي في الأساس اسم "برية" خاص.

تحدد ``unify(A، B)`` أي من الأسماء ``A`` و ``B`` التي سيتم نشرها إلى المخرجات.
إنه يعيد الأكثر *تحديدًا* من الاسمين، إذا تطابقا. إذا لم تتطابق الأسماء،
ثم يرتكب خطأ.

.. note::
    من الناحية العملية، عند العمل مع التوابع المسماة، يجب تجنب وجود أبعاد غير مسماة لأن التعامل معها قد يكون معقدًا. يوصى برفع
    جميع الأبعاد غير المسماة لتصبح أبعادًا مسماة باستخدام :meth: `~ Tensor.refine_names`.

قواعد استدلال الاسم الأساسية
^^^^^^^^^^^^^^^^^

دعونا نرى كيف يتم استخدام "المطابقة" و "unify" في استدلال الاسم في حالة
إضافة اثنين من التوابع ذات البعد الواحد بدون البث.

::

    x = torch.randn(3, names=('X',))
    y = torch.randn(3)
    z = torch.randn(3, names=('Z',))

**التحقق من الأسماء**: التحقق من أن أسماء التابعين *تتطابق*.

بالنسبة للأمثلة التالية:

::

    >>> # x + y  # match('X', None) is True
    >>> # x + z  # match('X', 'Z') is False
    >>> # x + x  # match('X', 'X') is True

    >>> x + z
    خطأ عند محاولة بث الأبعاد ['X'] والأبعاد ['Z']: البعد 'X' والبعد 'Z' في نفس الموضع من اليمين ولكن لا تتطابق.

**نشر الأسماء**: *توحيد* الأسماء لاختيار أيهما ينتشر.
في حالة ``x + y``، ``unify('X', None) = 'X'`` لأن ``'X'`` أكثر
تحديدًا من ``None``.

::

    >>> (x + y).names
    ('X',)
    >>> (x + x).names
    ('X',)

للاطلاع على قائمة شاملة بقواعد استدلال الاسم، راجع :ref: `name_inference_reference-doc`.
فيما يلي عمليتان شائعتان قد يكون من المفيد مراجعتهما:

- العمليات الحسابية الثنائية: :ref: `unifies_names_from_inputs-doc`
- عمليات الضرب المصفوفة: :ref: `contracts_away_dims-doc`

المواءمة الصريحة حسب الأسماء
---------------------------

استخدم :meth: `~ Tensor.align_as` أو :meth: `~ Tensor.align_to` لمواءمة أبعاد التابع
حسب الاسم إلى ترتيب محدد. هذا مفيد لأداء "البث حسب الأسماء".

::

    # هذه الدالة لا تعتمد على ترتيب الأبعاد لـ `input`،
    # طالما أن لديها بعد `C` في مكان ما.
    def scale_channels(input، scale):
        scale = scale.refine_names('C')
        return input * scale.align_as(input)

    >>> num_channels = 3
    >>> scale = torch.randn(num_channels, names=('C',))
    >>> imgs = torch.rand(3, 3, 3, num_channels, names=('N', 'H', 'W', 'C'))
    >>> more_imgs = torch.rand(3, num_channels, 3, 3, names=('N', 'C', 'H', 'W'))
    >>> videos = torch.randn(3، num_channels، 3، 3، 3، names=('N'، 'C'، 'H'، 'W'، 'D')

    >>> scale_channels(imgs، scale)
    >>> scale_channels(more_imgs، scale)
    >>> scale_channels(videos، scale)

التلاعب بالأبعاد
----------

استخدم :meth: `~ Tensor.align_to` لإعادة ترتيب كميات كبيرة من الأبعاد دون
ذكرها جميعًا كما هو مطلوب بواسطة :meth: `~ Tensor.permute`.

::

    >>> tensor = torch.randn(2, 2, 2, 2, 2, 2)
    >>> named_tensor = tensor.refine_names('A', 'B', 'C', 'D', 'E', 'F')

    # حرك البعد F (البعد 5) والبعد E (البعد 4) إلى الأمام مع الحفاظ
    # الباقي في نفس الترتيب
    >>> tensor.permute(5, 4, 0, 1, 2, 3)
    >>> named_tensor.align_to('F'، 'E'، ...)

استخدم :meth: `~ Tensor.flatten` و :meth: `~ Tensor.unflatten` لتقسيم الأبعاد وإلغاء تقسيمها،
على التوالي. هذه الطرق أكثر تفصيلاً من :meth: `~ Tensor.view`
و :meth: `~ Tensor.reshape`، ولكن لها معنى دلالي أكثر للشخص الذي يقرأ الكود.

::

    >>> imgs = torch.randn(32, 3, 128, 128)
    >>> named_imgs = imgs.refine_names('N', 'C', 'H', 'W')

    >>> flat_imgs = imgs.view(32, -1)
    >>> named_flat_imgs = named_imgs.flatten(['C', 'H', 'W'], 'features')
    >>> named_flat_imgs.names
    ('N'، 'features')

    >>> unflattened_named_imgs = named_flat_imgs.unflatten('features'، [('C'، 3)، ('H'، 128)، ('W'، 128)])
    >>> unflattened_named_imgs.names
    ('N'، 'C'، 'H'، 'W')

.. _named_tensors_autograd-doc:

دعم Autograd
--------------

يدعم Autograd حاليًا التوابع المسماة بطريقة محدودة: يتجاهل Autograd
الأسماء على جميع التوابع. لا يزال حساب التدرج صحيحًا ولكننا نفقد الأمان الذي توفره لنا الأسماء.

::

    >>> x = torch.randn(3, names=('D',))
    >>> weight = torch.randn(3, names=('D',)، requires_grad=True)
    >>> loss = (x - weight).abs()
    >>> grad_loss = torch.randn(3)
    >>> loss.backward(grad_loss)
    >>> weight.grad  # غير مسمى الآن. ستكون مسماة في المستقبل
    tensor([-1.8107، -0.6357، 0.0783])

    >>> weight.grad.zero_()
    >>> grad_loss = grad_loss.refine_names('C')
    >>> loss = (x - weight).abs()
    # من الناحية المثالية، يجب أن نتحقق من أن أسماء الخسارة و grad_loss تتطابق ولكننا لا نفعل ذلك بعد.
    >>> loss.backward(grad_loss)
    >>> weight.grad
    tensor([-1.8107، -0.6357، 0.0783])

العمليات والنظم الفرعية المدعومة حاليًا
------------------------

المشغلات
^^^^^^

راجع :ref: `name_inference_reference-doc` للحصول على قائمة كاملة بالمشغلات المدعومة
و :ref: `tensor operations`. لا ندعم ما يلي:

- الفهرسة، الفهرسة المتقدمة.

بالنسبة لمشغلات ``torch.nn.functional``، ندعم ما يلي:

- :func: `torch.nn.functional.relu`
- :func: `torch.nn.functional.softmax`
- :func: `torch.nn.functional.log_softmax`
- :func: `torch.nn.functional.tanh`
- :func: `torch.nn.functional.sigmoid`
- :func: `torch.nn.functional.dropout`

النظم الفرعية
^^^^^^^^^^

يتم دعم Autograd، راجع :ref: `named_tensors_autograd-doc`.
نظرًا لأن التدرجات غير مسماة حاليًا، فقد تعمل المحسنات ولكنها غير مختبرة.

وحدات NN غير مدعومة حاليًا. قد يؤدي هذا إلى ما يلي عند استدعاء
وحدات نمطية ذات مدخلات مسماة:

- معلمات الوحدة النمطية NN غير مسماة، لذا فقد تكون المخرجات مسماة جزئيًا.
- تحتوي تمريرات الوحدة النمطية NN للأمام على تعليمات برمجية لا تدعم التوابع المسماة وستخطئ بشكل مناسب.

لا ندعم أيضًا الأنظمة الفرعية التالية، على الرغم من أن بعضها قد يعمل
من الصندوق:

- التوزيعات
- التسلسل (:func: `torch.load`، :func: `torch.save`)
- المعالجة المتعددة
- JIT
- الموزعة
- ONNX

إذا كان أي من هذه الأمور مفيدًا لحالتك، يرجى البحث
`تم رفع قضية بالفعل <https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22>`_
وإذا لم يكن الأمر كذلك، `قم برفع واحدة <https://github.com/pytorch/pytorch/issues/new/choose>`_.

مرجع واجهة برمجة تطبيقات التوابع المسماة
--------------------------

في هذا القسم، يرجى الاطلاع على الوثائق الخاصة بواجهات برمجة التطبيقات المحددة للتوابع المسماة.
للمرجع الشامل حول كيفية انتشار الأسماء عبر مشغلات PyTorch الأخرى، راجع :ref: `name_inference_reference-doc`.

.. class:: Tensor()
   :noindex:

   .. autoattribute:: names
   .. automethod:: rename
   .. automethod:: rename_
   .. automethod:: refine_names

   .. automethod:: align_as
   .. automethod:: align_to

   .. py:method:: flatten(dims، out_dim) -> Tensor
      :noindex:

      يسطّح :attr: `dims` في بعد واحد باسم :attr: `out_dim`.

      يجب أن تكون جميع `dims` متتالية في الترتيب في التابع :attr: `self`،
      ولكن ليس من الضروري أن تكون متجاورة في الذاكرة.

      الأمثلة::

          >>> imgs = torch.randn(32, 3, 128, 128, names=('N'، 'C'، 'H'، 'W'))
          >>> flat_imgs = imgs.flatten(['C'، 'H'، 'W']، 'features')
          >>> flat_imgs.names، flat_imgs.shape
          (('N'، 'features')، torch.Size ([32، 49152]))

      .. warning::
          واجهة برمجة تطبيقات التوابع المسماة تجريبية وقد تتغير.