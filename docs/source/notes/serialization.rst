.. _serialization:

دلاليات التسلسل
-----------
هذا النص يشرح كيفية حفظ وتحميل تنسورات PyTorch وحالات الوحدات النمطية في بايثون، وكيفية تسلسل الوحدات النمطية في بايثون بحيث يمكن تحميلها في سي++.

.. contents:: جدول المحتويات

.. _saving-loading-tensors:

حفظ وتحميل التنسورات
-----------------

:func:`torch.save` و :func:`torch.load` يتيحان لك حفظ وتحميل التنسورات بسهولة:

::

    >>> t = torch.tensor([1., 2.])
    >>> torch.save(t, 'tensor.pt')
    >>> torch.load('tensor.pt')
    tensor([1., 2.])

بحسب الاتفاقية، يتم كتابة ملفات PyTorch عادةً مع امتداد '.pt' أو '.pth'.

:func:`torch.save` و :func:`torch.load` يستخدمان Python’s pickle بشكل افتراضي،
لذلك يمكنك أيضًا حفظ تنسورات متعددة كجزء من كائنات بايثون مثل tuples،
القوائم، والقواميس:

::

    >>> d = {'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])}
    >>> torch.save(d, 'tensor_dict.pt')
    >>> torch.load('tensor_dict.pt')
    {'a': tensor([1., 2.]), 'b': tensor([3., 4.])}

يمكن أيضًا حفظ البنى البيانات المخصصة التي تتضمن تنسورات PyTorch إذا كانت
بنية البيانات قابلة للتخليل.

.. _preserve-storage-sharing:

حفظ وتحميل التنسورات يحافظ على العروض
----------------------------------

يحافظ حفظ التنسورات على علاقات العرض الخاصة بها:

::

    >>> numbers = torch.arange(1, 10)
    >>> evens = numbers[1::2]
    >>> torch.save([numbers, evens], 'tensors.pt')
    >>> loaded_numbers, loaded_evens = torch.load('tensors.pt')
    >>> loaded_evens *= 2
    >>> loaded_numbers
    tensor([ 1,  4,  3,  8,  5, 12,  7, 16,  9])

في الخلفية، تشترك هذه التنسورات في نفس "التخزين". راجع
`عروض التنسورات <https://pytorch.org/docs/main/tensor_view.html>`_ للمزيد
من المعلومات حول العروض والتخزين.

عندما يحفظ PyTorch التنسورات، فإنه يحفظ كائنات التخزين الخاصة بها وبيانات
تعريف التنسور بشكل منفصل. هذه تفاصيل تنفيذ قد تتغير في المستقبل، ولكنها
عادة ما توفر المساحة وتسمح لـ PyTorch بإعادة بناء علاقات العرض بين
التنسورات المحملة بسهولة. في المقطع أعلاه، على سبيل المثال، يتم كتابة
تخزين واحد فقط في 'tensors.pt'.

ومع ذلك، في بعض الحالات، قد يكون حفظ كائنات التخزين الحالية غير ضروري
ويؤدي إلى إنشاء ملفات كبيرة بشكل محظور. في المقطع التالي، يتم كتابة
تخزين أكبر بكثير من التنسور إلى ملف:

::

    >>> large = torch.arange(1, 1000)
    >>> small = large[0:5]
    >>> torch.save(small, 'small.pt')
    >>> loaded_small = torch.load('small.pt')
    >>> loaded_small.storage().size()
    999

بدلاً من حفظ القيم الخمسة فقط في تنسور "small" في "small.pt"، تم حفظ
999 قيمة في التخزين الذي يشترك فيه مع "large" وتحميلها.

عند حفظ التنسورات التي تحتوي على عدد أقل من العناصر من كائنات التخزين الخاصة
بها، يمكن تقليل حجم الملف المحفوظ عن طريق استنساخ التنسورات أولاً. ينتج
عن استنساخ تنسور تنسور جديد بكائن تخزين جديد يحتوي فقط على القيم الموجودة
في التنسور:

::

    >>> large = torch.arange(1, 1000)
    >>> small = large[0:5]
    >>> torch.save(small.clone(), 'small.pt')  # saves a clone of small
    >>> loaded_small = torch.load('small.pt')
    >>> loaded_small.storage().size()
    5

نظرًا لأن التنسورات المستنسخة مستقلة عن بعضها البعض، فإنها لا تحتوي على
أي من علاقات العرض التي كانت موجودة في التنسورات الأصلية. إذا كان حجم
الملف وعلاقات العرض مهمة عند حفظ التنسورات الأصغر من كائنات التخزين
الخاصة بها، فيجب توخي الحذر لإنشاء تنسورات جديدة تقلل من حجم كائنات
التخزين الخاصة بها ولكن لا تزال تحتوي على علاقات العرض المرغوبة قبل الحفظ.

.. _saving-loading-python-modules:

حفظ وتحميل torch.nn.Modules
--------------------------

راجع أيضًا: `التدريب: حفظ وتحميل الوحدات النمطية <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_

في PyTorch، يتم تسلسل حالة الوحدة النمطية غالبًا باستخدام "قاموس الحالة".
يحتوي قاموس حالة الوحدة النمطية على جميع معلماتها وذاكرتها المؤقتة الثابتة:

::

    >>> bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
    >>> list(bn.named_parameters())
    [('weight', Parameter containing: tensor([1., 1., 1.], requires_grad=True)),
     ('bias', Parameter containing: tensor([0., 0., 0.], requires_grad=True))]

    >>> list(bn.named_buffers())
    [('running_mean', tensor([0., 0., 0.])),
     ('running_var', tensor([1., 1., 1.])),
     ('num_batches_tracked', tensor(0))]

    >>> bn.state_dict()
    OrderedDict([('weight', tensor([1., 1., 1.])),
                 ('bias', tensor([0., 0., 0.])),
                 ('running_mean', tensor([0., 0., 0.])),
                 ('running_var', tensor([1., 1., 1.])),
                 ('num_batches_tracked', tensor(0))])

بدلاً من حفظ الوحدة النمطية مباشرةً، يوصى بحفظ قاموس الحالة فقط لأسباب
متعلقة بالتوافق. تحتوي الوحدات النمطية في بايثون حتى على دالة،
:meth:`~torch.nn.Module.load_state_dict`، لاستعادة حالتها من قاموس الحالة:

::

    >>> torch.save(bn.state_dict(), 'bn.pt')
    >>> bn_state_dict = torch.load('bn.pt')
    >>> new_bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
    >>> new_bn.load_state_dict(bn_state_dict)
    <All keys matched successfully>

لاحظ أنه يتم تحميل قاموس الحالة أولاً من ملفه باستخدام :func:`torch.load`
ويتم بعد ذلك استعادة الحالة باستخدام :meth:`~torch.nn.Module.load_state_dict`.

حتى الوحدات النمطية المخصصة والوحدات النمطية التي تحتوي على وحدات نمطية
أخرى لديها قواميس الحالة ويمكنها استخدام هذا النمط:

::

    # وحدة نمطية بطبقتين خطيتين
    >>> class MyModule(torch.nn.Module):
          def __init__(self):
            super().__init__()
            self.l0 = torch.nn.Linear(4, 2)
            self.l1 = torch.nn.Linear(2, 1)

          def forward(self, input):
            out0 = self.l0(input)
            out0_relu = torch.nn.functional.relu(out0)
            return self.l1(out0_relu)

    >>> m = MyModule()
    >>> m.state_dict()
    OrderedDict([('l0.weight', tensor([[ 0.1400, 0.4563, -0.0271, -0.4406],
                                       [-0.3289, 0.2827, 0.4588, 0.2031]])),
                 ('l0.bias', tensor([ 0.0300, -0.1316])),
                 ('l1.weight', tensor([[0.6533, 0.3413]])),
                 ('l1.bias', tensor([-0.1112]))])

    >>> torch.save(m.state_dict(), 'mymodule.pt')
    >>> m_state_dict = torch.load('mymodule.pt')
    >>> new_m = MyModule()
    >>> new_m.load_state_dict(m_state_dict)
    <All keys matched successfully>

.. _serialized-file-format:

تنسيق الملف المسلسل لـ ``torch.save``
-------------------------------

منذ PyTorch 1.6.0، يقوم ``torch.save`` بشكل افتراضي بإرجاع أرشيف ZIP64 غير مضغوط
ما لم يحدد المستخدم ``_use_new_zipfile_serialization=False``.

في هذا الأرشيف، يتم ترتيب الملفات على النحو التالي

.. code-block:: text

    checkpoint.pth
    ├── data.pkl
    ├── byteorder  # تمت إضافته في PyTorch 2.1.0
    ├── data/
    │   ├── 0
    │   ├── 1
    │   ├── 2
    │   └── …
    └── version

الإدخالات هي كما يلي:
  * ``data.pkl`` هي نتيجة تخليل الكائن الذي تم تمريره إلى ``torch.save``
    باستثناء كائنات "torch.Storage" التي يحتوي عليها
  * ``byteorder`` يحتوي على سلسلة مع ``sys.byteorder`` عند الحفظ ("little" أو "big")
  * ``data/`` يحتوي على جميع التخزين في الكائن، حيث يكون كل تخزين ملفًا منفصلاً
  * ``version`` يحتوي على رقم إصدار في وقت الحفظ والذي يمكن استخدامه في وقت التحميل

عند الحفظ، سيضمن PyTorch أن يكون رأس الملف المحلي لكل ملف مضبوطًا إلى
إزاحة مضاعفة لـ 64 بايت، مما يضمن أن يكون إزاحة كل ملف مضبوطة إلى 64 بايت.

.. note::
    يتم تسلسل التنسورات على أجهزة معينة مثل XLA كصفيفات نومبي مخللة. وبالتالي،
    لا يتم تسلسل تخزينها. في هذه الحالات، قد لا يوجد "data/" في نقطة المراقبة.

.. _serializing-python-modules:

تسلسل الوحدات النمطية في torch.nn.Modules وتحميلها في C++
----------------------------------------------------

راجع أيضًا: `التدريب: تحميل نموذج TorchScript في C++ <https://pytorch.org/tutorials/advanced/cpp_export.html>`_

يمكن تسلسل الوحدات النمطية النصية كبرنامج TorchScript وتحميلها
باستخدام :func:`torch.jit.load`.
يقوم هذا التسلسل بتشفير جميع طرق الوحدة النمطية، والوحدات النمطية الفرعية،
والمعلمات، والسمات، ويتيح تحميل البرنامج المسلسل في C++
(أي بدون بايثون).

قد لا يكون التمييز بين :func:`torch.jit.save` و :func:`torch.save` واضحًا
على الفور. :func:`torch.save` يحفظ كائنات بايثون باستخدام pickle.
هذا مفيد بشكل خاص للنماذج الأولية والبحث والتدريب. :func:`torch.jit.save`،
من ناحية أخرى، يقوم بتوصيل الوحدات النمطية النصية إلى تنسيق يمكن تحميله
في بايثون أو C++. هذا مفيد عند حفظ وتحميل الوحدات النمطية في C++ أو لتشغيل
الوحدات النمطية المدربة في بايثون مع C++، وهي ممارسة شائعة عند نشر نماذج
PyTorch.

لتوصيل وحدة نمطية وتسلسلها وتحميلها في بايثون:

::

    >>> scripted_module = torch.jit.script(MyModule())
    >>> torch.jit.save(scripted_module, 'mymodule.pt')
    >>> torch.jit.load('mymodule.pt')
    RecursiveScriptModule( original_name=MyModule
                          (l0): RecursiveScriptModule(original_name=Linear)
                          (l1): RecursiveScriptModule(original_name=Linear) )


يمكن أيضًا حفظ الوحدات النمطية المتبعة باستخدام :func:`torch.jit.save`،
مع ملاحظة أن مسار التعليمات البرمجية المتبعة فقط هو المسلسل. يوضح المثال
التالي ذلك:

::

    # وحدة نمطية مع التحكم في التدفق
    >>> class ControlFlowModule(torch.nn.Module):
          def __init__(self):
            super().__init__()
            self.l0 = torch.nn.Linear(4, 2)
            self.l1 = torch.nn.Linear(2, 1)

          def forward(self, input):
            if input.dim() > 1:
                return torch.tensor(0)

            out0 = self.l0(input)
            out0_relu = torch.nn.functional.relu(out0)
            return self.l1(out0_relu)

    >>> traced_module = torch.jit.trace(ControlFlowModule(), torch.randn(4))
    >>> torch.jit.save(traced_module, 'controlflowmodule_traced.pt')
    >>> loaded = torch.jit.load('controlflowmodule_traced.pt')
    >>> loaded(torch.randn(2, 4)))
    tensor([[-0.1571], [-0.3793]], grad_fn=<AddBackward0>)

    >>> scripted_module = torch.jit.script(ControlFlowModule(), torch.randn(4))
    >>> torch.jit.save(scripted_module, 'controlflowmodule_scripted.pt')
    >>> loaded = torch.jit.load('controlflowmodule_scripted.pt')
    >> loaded(torch.randn(2, 4))
    tensor(0)

تحتوي الوحدة النمطية أعلاه على عبارة if لا يتم تشغيلها بواسطة المدخلات المتبعة،
لذلك فهي ليست جزءًا من الوحدة النمطية المتبعة ولا يتم تسلسلها معها.
تحتوي الوحدة النمطية النصية، من ناحية أخرى، على عبارة if ويتم تسلسلها معها.
راجع وثائق `TorchScript <https://pytorch.org/docs/stable/jit.html>`_
للمزيد من المعلومات حول النصوص النصية والتعقب.

أخيرًا، لتحميل الوحدة النمطية في C++:

::

    >>> torch::jit::script::Module module;
    >>> module = torch::jit::load('controlflowmodule_scripted.pt');

راجع وثائق `API لـ PyTorch C++ <https://pytorch.org/cppdocs/>`_
للحصول على التفاصيل حول كيفية استخدام الوحدات النمطية PyTorch في C++.

.. _saving-loading-across-versions:

حفظ وتحميل الوحدات النمطية النصية عبر إصدارات PyTorch
توصي فريق PyTorch بحفظ وتحميل الوحدات النمطية باستخدام نفس الإصدار من PyTorch. وقد لا تدعم الإصدارات القديمة من PyTorch الوحدات النمطية الأحدث، وقد تزيل الإصدارات الأحدث أو تعدل السلوكيات الأقدم. وتوصف هذه التغييرات بوضوح في "ملاحظات الإصدار" الخاصة بـ PyTorch، وقد تحتاج الوحدات النمطية التي تعتمد على الوظائف التي تم تغييرها إلى التحديث للاستمرار في العمل بشكل صحيح. في حالات محدودة موضحة أدناه، سيحافظ PyTorch على السلوك التاريخي للوحدات النصية المسلسلة بحيث لا تحتاج إلى تحديث.

قيام torch.div بإجراء القسمة الصحيحة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

في PyTorch 1.5 والإصدارات الأقدم، تقوم دالة :func:`torch.div` بإجراء قسمة صحيحة عندما يتم تمرير مدخلين صحيحين لها:

::

    # PyTorch 1.5 (والإصدارات الأقدم)
    >>> a = torch.tensor(5)
    >>> b = torch.tensor(3)
    >>> a / b
    tensor(1)

ولكن في PyTorch 1.7، تقوم دالة :func:`torch.div` دائمًا بإجراء قسمة حقيقية لمدخلاتها، تمامًا مثل القسمة في Python 3:

::

    # PyTorch 1.7
    >>> a = torch.tensor(5)
    >>> b = torch.tensor(3)
    >>> a / b
    tensor(1.6667)

يتم الحفاظ على سلوك دالة :func:`torch.div` في الوحدات النصية المسلسلة. وهذا يعني أن الوحدات النصية المسلسلة بإصدارات PyTorch الأقدم من 1.6 ستستمر في رؤية قيام دالة :func:`torch.div` بإجراء قسمة صحيحة عند تمرير مدخلين صحيحين لها، حتى عند تحميلها بإصدارات أحدث من PyTorch. لا يمكن تحميل الوحدات النصية التي تستخدم دالة :func:`torch.div` والمسلسلة على PyTorch 1.6 أو إصدار أحدث في الإصدارات الأقدم من PyTorch، لأن تلك الإصدارات الأقدم لا تفهم السلوك الجديد.

قيام دالة torch.full دائمًا باستنتاج نوع بيانات float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

في PyTorch 1.5 والإصدارات الأقدم، تقوم دالة :func:`torch.full` دائمًا بإرجاع مصفوفة ذات نوع بيانات float، بغض النظر عن قيمة الملء التي يتم تمريرها لها:

::

    # PyTorch 1.5 والإصدارات الأقدم
    >>> torch.full((3,), 1)  # لاحظ قيمة الملء الصحيحة...
    tensor([1., 1., 1.])  # ...ولكن مصفوفة ذات نوع بيانات float!

ولكن في PyTorch 1.7، تقوم دالة :func:`torch.full` باستنتاج نوع بيانات المصفوفة التي يتم إرجاعها من قيمة الملء:

::

    # PyTorch 1.7
    >>> torch.full((3,), 1)
    tensor([1, 1, 1])

    >>> torch.full((3,), True)
    tensor([True, True, True])

    >>> torch.full((3,), 1.)
    tensor([1., 1., 1.])

    >>> torch.full((3,), 1 + 1j)
    tensor([1.+1.j, 1.+1.j, 1.+1.j])

يتم الحفاظ على سلوك دالة :func:`torch.full` في الوحدات النصية المسلسلة. وهذا يعني أن الوحدات النصية المسلسلة بإصدارات PyTorch الأقدم من 1.6 ستستمر في رؤية قيام دالة :func:`torch.full` بإرجاع مصفوفة ذات نوع بيانات float بشكل افتراضي، حتى عند تمرير قيم ملء صحيحة أو منطقية. لا يمكن تحميل الوحدات النصية التي تستخدم دالة :func:`torch.full` والمسلسلة على PyTorch 1.6 أو إصدار أحدث في الإصدارات الأقدم من PyTorch، لأن تلك الإصدارات الأقدم لا تفهم السلوك الجديد.

.. _دالات المنفعة:

دالات المنفعة
----------

الدالات التالية هي دالات منفعة مرتبطة بالتسلسل:

.. currentmodule:: torch.serialization

.. autofunction:: register_package
.. autofunction:: get_default_load_endianness
.. autofunction:: set_default_load_endianness
.. autofunction:: get_default_mmap_options
.. autofunction:: set_default_mmap_options
.. autofunction:: add_safe_globals
.. autofunction:: clear_safe_globals
.. autofunction:: get_safe_globals
.. autoclass:: safe_globals