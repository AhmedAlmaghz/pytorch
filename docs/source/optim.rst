.. class:: torch.optim

   ويوفر حزم torch.optim تطبيقًا للخوارزميات الشائعة لتحسين المحتوى.

   .. currentmodule:: torch.optim

   .. autoclass:: Optimizer
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Adam
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: AdamW
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Adadelta
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Adagrad
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Adamax
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: ASGD
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: RMSprop
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Rprop
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: SGD
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: LBFGS
      :members:
      :undoc-members:
      :show-inheritance:

.. function:: torch.optim.lr_scheduler.step(closure=None)

   .. currentmodule:: torch.optim.lr_scheduler

   .. autofunction:: step

.. function:: torch.optim.lr_scheduler.get_last_lr()

   .. autofunction:: get_last_lr

.. function:: torch.optim.lr_scheduler.get_lr()

   .. autofunction:: get_lr

.. function:: torchMultiplier.torch.optim.lr_scheduler.state_dict()

   .. autofunction:: state_dict

.. function:: torch.optim.lr_scheduler.load_state_dict(state_dict)

   .. autofunction:: load_state_dict

حزمة torch.optim
===============

يوفر حزم torch.optim تطبيقًا للخوارزميات الشائعة لتحسين المحتوى.

الصنف Optimizer
---------------

.. autoclass:: torch.optim.Optimizer
   :members:
   :undoc-members:
   :show-inheritance:

خوارزميات التحسين
------------------

هذه هي خوارزميات التحسين المتوفرة في حزمة torch.optim.

آدم
^^^^

.. autoclass:: torch.optim.Adam
   :members:
   :undoc-members:
   :show-inheritance:

آدم دبليو
^^^^^^^^^^

.. autoclass:: torch.optim.AdamW
   :members:
   :undoc-members:
   :show-inheritance:

أدالديلتا
^^^^^^^^^^

.. autoclass:: torch.optim.Adadelta
   :members:
   :undoc-members:
   :show-inheritance:

أداجراد
^^^^^^^^

.. autoclass:: torch.optim.Adagrad
   :members:
   :undoc-members:
   :show-inheritance:

أداماكس
^^^^^^^^

.. autoclass:: torch.optim.Adamax
   :members:
   :undoc-members:
   :show-inheritance:

إيه إس جي دي
^^^^^^^^^^^

.. autoclass:: torch.optim.ASGD
   :members:
   :undoc-members:
   :show-inheritance:

آر إم إس بروب
^^^^^^^^^^^^^

.. autoclass:: torch.optim.RMSprop
   :members:
   :undoc-members:
   :show-inheritance:

آر بروب
^^^^^^^^

.. autoclass:: torch.optim.Rprop
   :members:
   :undoc-members:
   :show-inheritance:

إس جي دي
^^^^^^^^

.. autoclass:: torch.optim.SGD
   :members:
   :undoc-members:
   :show-inheritance:

إل بي إف جي إس
^^^^^^^^^^^^^

.. autoclass:: torch.optim.LBFGS
   :members:
   :undoc-members:
   :show-inheritance:

وظائف الجدولة
----------------

هذه هي وظائف الجدولة المتوفرة في حزمة torch.optim.lr_scheduler.

خطوة
^^^^

.. autofunction:: torch.optim.lr_scheduler.step

جيت_لاست_ال
^^^^^^^^^^^^^

.. autofunction:: torch.optim.lr_scheduler.get_last_lr

جيت_ال
^^^^^^^^

.. autofunction:: torch.optim.lr_scheduler.get_lr

ستيت_ديكت
^^^^^^^^^^^^

.. autofunction:: torch.optim.lr_scheduler.state_dict

لود_ستيت_ديكت
^^^^^^^^^^^^^^^^^

.. autofunction:: torch.optim.lr_scheduler.load_state_dict
.. automodule:: torch.optim

كيفية استخدام المحسن
-----------------------

لاستخدام :mod:`torch.optim` ، يجب عليك إنشاء كائن محسن يحتفظ
بالحالة الحالية وسيقوم بتحديث المعلمات بناءً على التدرجات المحسوبة.

إنشاؤه
^^^^^^^^^^^^^^^

لإنشاء :class:`Optimizer` ، يجب عليك منحه iterable يحتوي على
المعلمات (يجب أن تكون جميعها :class:`~torch.autograd.Variable` s) للتحسين. بعد ذلك،
يمكنك تحديد خيارات خاصة بالمحسن مثل معدل التعلم، وتناقص الوزن، وما إلى ذلك.

مثال::

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)

خيارات لكل معلمة
^^^^^^^^^^^^^^^^^^^^^

:class:`Optimizer` s تدعم أيضًا تحديد خيارات لكل معلمة. للقيام بذلك، بدلاً من تمرير iterable من :class:`~torch.autograd.Variable` s، قم بتمرير iterable من
:class:`dict` s. سيحدد كل منها مجموعة معلمات منفصلة، ويجب أن يحتوي على
مفتاح "params"، يحتوي على قائمة من المعلمات التي تنتمي إليه. يجب أن تتطابق المفاتيح الأخرى
مع الحجج الكلمة التي يقبلها المحسنون، وسيتم استخدامها
كخيارات تحسين لهذه المجموعة.

على سبيل المثال، هذا مفيد جدًا عندما يريد المرء تحديد معدلات تعلم لكل طبقة::

    optim.SGD([
                    {'params': model.base.parameters(), 'lr': 1e-2},
                    {'params': model.classifier.parameters()}
                ], lr=1e-3, momentum=0.9)

هذا يعني أن معلمات "model.base" ستستخدم معدل تعلم "1e-2"، في حين
ستلتزم معلمات "model.classifier" بمعدل التعلم الافتراضي "1e-3".
أخيرًا، سيتم استخدام زخم "0.9" لجميع المعلمات.

.. note::

    يمكنك أيضًا تمرير الخيارات كحجج كلمة رئيسية. سيتم استخدامها كـ
    الافتراضيات، في المجموعات التي لم تتجاوزها. هذا مفيد عندما
    تريد فقط تنويع خيار واحد، مع الحفاظ على جميع الخيارات الأخرى متسقة
    بين مجموعات المعلمات.

ضع في اعتبارك أيضًا المثال التالي المتعلق بالعقوبة المميزة للمعلمات.
تذكر أن :func:`~torch.nn.Module.parameters` يعيد iterable يحتوي على
جميع المعلمات القابلة للتعلم، بما في ذلك الانحيازات والمعلمات الأخرى
التي قد تفضل العقوبة المميزة. لمعالجة ذلك، يمكن للمرء تحديد
أوزان العقوبة الفردية لكل مجموعة معلمات::

    bias_params = [p for name, p in self.named_parameters() if 'bias' in name]
    others = [p for name, p in self.named_parameters() if 'bias' not in name]

    optim.SGD([
                    {'params': others},
                    {'params': bias_params, 'weight_decay': 0}
                ], weight_decay=1e-2, lr=1e-2)

بهذه الطريقة، يتم عزل مصطلحات الانحياز عن المصطلحات غير الانحيازية، ويتم تعيين "weight_decay"
من "0" بشكل محدد لمصطلحات الانحياز، وذلك لتجنب أي عقوبة لهذه المجموعة.


اتخاذ خطوة تحسين
^^^^^^^^^^^^^^^^^^^^^^^^^^^

تنفذ جميع المحسنات طريقة :func:`~Optimizer.step` ، والتي تحدِّث
المعلمات. يمكن استخدامه بطريقتين:

``optimizer.step()``
~~~~~~~~~~~~~~~~~~~~

هذه هي النسخة المبسطة التي يدعمها معظم المحسنين. يمكن استدعاء الدالة مرة واحدة
تم حساب التدرجات باستخدام على سبيل المثال
:func:`~torch.autograd.Variable.backward`.

مثال::

    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

``optimizer.step(closure)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

تحتاج بعض خوارزميات التحسين مثل Conjugate Gradient وLBFGS إلى
إعادة تقييم الدالة عدة مرات، لذلك يجب عليك تمرير إغلاق يسمح لهم
إعادة حساب نموذجك. يجب أن يقوم الإغلاق بمسح التدرجات،
حساب الخسارة، وإعادتها.

مثال::

    for input, target in dataset:
        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss
        optimizer.step(closure)

.. _optimizer-algorithms:

الفئة الأساسية
----------

.. autoclass:: Optimizer

.. autosummary::
    :toctree: generated
    :nosignatures:

    Optimizer.add_param_group
    Optimizer.load_state_dict
    Optimizer.register_load_state_dict_pre_hook
    Optimizer.register_load_state_dict_post_hook
    Optimizer.state_dict
    Optimizer.register_state_dict_pre_hook
    Optimizer.register_state_dict_post_hook
    Optimizer.step
    Optimizer.register_step_pre_hook
    Optimizer.register_step_post_hook
    Optimizer.zero_grad

الخوارزميات
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Adadelta
    Adafactor
    Adagrad
    Adam
    AdamW
    SparseAdam
    Adamax
    ASGD
    LBFGS
    NAdam
    RAdam
    RMSprop
    Rprop
    SGD

الكثير من خوارزمياتنا لها العديد من التنفيذات التي تم تحسينها للأداء
وإمكانية القراءة و/أو العمومية، لذلك نحاول الافتراض إلى التنفيذ الأسرع بشكل عام
للجهاز الحالي إذا لم يحدد المستخدم تنفيذًا معينًا.

لدينا 3 فئات رئيسية من التنفيذات: for-loop، وforeach (multi-tensor)، و
مدمج. أكثر التنفيذات مباشرة هي حلقات for-loop على المعلمات مع
أجزاء كبيرة من الحساب. عادةً ما يكون التكرار أبطأ من تنفيذات foreach الخاصة بنا، والتي
تدمج المعلمات في multi-tensor وتشغل أجزاء كبيرة
من الحساب دفعة واحدة، وبالتالي توفير العديد من مكالمات kernel التسلسلية. لدى بعض محسناتنا
تنفيذات مدمجة أسرع حتى الآن، والتي تدمج أجزاء كبيرة من
الحساب في kernel واحد. يمكننا أن نفكر في تنفيذات foreach كدمج
أفقيًا والتنفيذات المدمجة كدمج رأسيًا فوق ذلك.

بشكل عام، ترتيب الأداء للتنفيذات الثلاثة هو fused > foreach > for-loop.
لذلك عندما ينطبق، فإننا نعتمد افتراضيًا على foreach على for-loop. يعني قابل للتطبيق أن تنفيذ foreach
متوفر، ولم يحدد المستخدم أي تنفيذات خاصة kwargs (مثل المدمج، foreach، differentiable)، وجميع tensors أصلية. لاحظ أنه في حين يجب أن يكون المدمج
أسرع حتى من foreach، فإن التنفيذات أحدث ونود منحها
مزيد من الوقت للخبز قبل التبديل في كل مكان. نلخص حالة الاستقرار
لكل تنفيذ في الجدول الثاني أدناه، يمكنك تجربتها!

فيما يلي جدول يظهر التنفيذات المتاحة والافتراضية لكل خوارزمية:

.. csv-table::
    :header: "Algorithm", "Default", "Has foreach?", "Has fused?"
    :widths: 25, 25, 25, 25
    :delim: ;

    :class:`Adadelta`;foreach;نعم;لا
    :class:`Adafactor`;for-loop;لا;لا
    :class:`Adagrad`;foreach;نعم;نعم (cpu فقط)
    :class:`Adam`;foreach;نعم;نعم
    :class:`AdamW`;foreach;نعم;نعم
    :class:`SparseAdam`;for-loop;لا;لا
    :class:`Adamax`;foreach;نعم;لا
    :class:`ASGD`;foreach;نعم;لا
    :class:`LBFGS`;for-loop;لا;لا
    :class:`NAdam`;foreach;نعم;لا
    :class:`RAdam`;foreach;نعم;لا
    :class:`RMSprop`;foreach;نعم;لا
    :class:`Rprop`;foreach;نعم;لا
    :class:`SGD`;foreach;نعم;نعم

يوضح الجدول التالي حالة الاستقرار للتنفيذات المدمجة:

.. csv-table::
    :header: "Algorithm", "CPU", "CUDA"، "MPS"
    :widths: 25, 25, 25, 25
    :delim: ;

    :class:`Adadelta`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`Adafactor`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`Adagrad`;بيتا;غير مدعوم;غير مدعوم
    :class:`Adam`;بيتا;مستقر;بيتا
    :class:`AdamW`;بيتا;مستقر;بيتا
    :class:`SparseAdam`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`Adamax`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`ASGD`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`LBFGS`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`NAdam`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`RAdam`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`RMSprop`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`Rprop`;غير مدعوم;غير مدعوم;غير مدعوم
    :class:`SGD`;بيتا;بيتا;بيتا

كيفية ضبط معدل التعلم
---------------------------

:class:`torch.optim.lr_scheduler.LRScheduler` يوفر عدة طرق لتعديل معدل التعلم
بناءً على عدد العصور. :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`
يسمح بخفض معدل التعلم الديناميكي بناءً على بعض قياسات التحقق من الصحة.

يجب تطبيق جدولة معدل التعلم بعد تحديث المحسن؛ على سبيل المثال، يجب عليك
كتابة الكود بهذه الطريقة:

مثال::

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(20):
        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

يمكن استدعاء معظم مخططات معدل التعلم الواحد تلو الآخر (يُشار إليها أيضًا باسم
تسلسل المخططات). النتيجة هي أن كل مخطط يتم تطبيقه الواحد تلو الآخر على
معدل التعلم الذي حصل عليه من الذي يسبقه.

مثال::

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    for epoch in range(20):
        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler1.step()
        scheduler2.step()

في العديد من الأماكن في الوثائق، سنستخدم القالب التالي للإشارة إلى خوارزميات المخطط.

    >>> scheduler = ...
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()

.. warning::
  قبل PyTorch 1.1.0، كان من المتوقع استدعاء مخطط معدل التعلم قبل
  تحديث المحسن؛ 1.1.0 غير هذا السلوك بطريقة غير متوافقة مع الإصدارات السابقة. إذا كنت تستخدم
  مخطط معدل التعلم (استدعاء "scheduler.step()") قبل تحديث المحسن
  (استدعاء "optimizer.step()"))، فسيتم تخطي القيمة الأولى لجدول معدل التعلم.
  إذا كنت غير قادر على إعادة إنتاج النتائج بعد الترقية إلى PyTorch 1.1.0، يرجى التحقق
  ما إذا كنت تستدعي "scheduler.step()" في الوقت غير المناسب.


.. autosummary::
    :toctree: generated
    :nosignatures:

    lr_scheduler.LRScheduler
    lr_scheduler.LambdaLR
    lr_scheduler.MultiplicativeLR
    lr_scheduler.StepLR
    lr_scheduler.MultiStepLR
    lr_scheduler.ConstantLR
    lr_scheduler.LinearLR
    lr_scheduler.ExponentialLR
    lr_scheduler.PolynomialLR
    lr_scheduler.CosineAnnealingLR
    lr_scheduler.ChainedScheduler
    lr_scheduler.SequentialLR
    lr_scheduler.ReduceLROnPlateau
    lr_scheduler.CyclicLR
    lr_scheduler.OneCycleLR
    lr_scheduler.CosineAnnealingWarmRestarts

متوسط ​​الوزن (SWA وEMA)
بالتأكيد! فيما يلي النص المترجم بتنسيق ReStructuredText:

------------------------------

:class:`torch.optim.swa_utils.AveragedModel` ينفذ المتوسط ​​الستوكاستيكي للوزن (SWA) والمتوسط ​​المتحرك الأسّي (EMA)،
:class:`torch.optim.swa_utils.SWALR` ينفذ جدول معدل التعلم SWA
:func:`torch.optim.swa_utils.update_bn` هي دالة مساعدة تستخدم لتحديث إحصائيات التطبيع الدفعي SWA/EMA
في نهاية التدريب.

تم اقتراح SWA في "يؤدي متوسط الأوزان إلى Optima أوسع وتعميم أفضل"_.

EMA هي تقنية معروفة على نطاق واسع لتقليل وقت التدريب عن طريق تقليل عدد تحديثات الأوزان المطلوبة. إنه تنوع لـ `Polyak averaging`_، ولكن باستخدام أوزان أسية بدلاً من أوزان متساوية عبر التكرارات.

.. _`يؤدي متوسط الأوزان إلى Optima أوسع وتعميم أفضل`: https://arxiv.org/abs/1803.05407

.. _`Polyak averaging`: https://paperswithcode.com/method/polyak-averaging

بناء نماذج متوسطة
^^^^^^^^^^^^^^^^^^^^

تستخدم فئة `AveragedModel` لحساب أوزان نموذج SWA أو EMA.

يمكنك إنشاء نموذج SWA بمتوسط تشغيل الأمر التالي:

>>> averaged_model = AveragedModel(model)

يتم بناء نماذج EMA عن طريق تحديد وسيط ``multi_avg_fn`` كما يلي:

>>> decay = 0.999
>>> averaged_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))

التدهور هو معلمة بين 0 و 1 تتحكم في سرعة تدهور المعلمات المتوسطة. إذا لم يتم توفيره إلى :func:`torch.optim.swa_utils.get_ema_multi_avg_fn`، يكون الافتراضي 0.999.

:func:`torch.optim.swa_utils.get_ema_multi_avg_fn` تعيد دالة تطبق معادلة EMA التالية على الأوزان:

.. math:: W^\textrm{EMA}_{t+1} = \alpha W^\textrm{EMA}_{t} + (1 - \alpha) W^\textrm{model}_t

حيث ألفا هو التدهور EMA.

هنا، يمكن أن يكون النموذج "نموذج" كائنًا تعسفيًا لـ :class:`torch.nn.Module`. سوف "averaged_model"
تتبع المتوسطات الجارية لمعلمات "النموذج". لتحديث هذه
المتوسطات، يجب عليك استخدام دالة :func:`update_parameters` بعد `optimizer.step()`:

>>> averaged_model.update_parameters(model)

بالنسبة لـ SWA و EMA، تتم هذه المكالمة عادةً مباشرة بعد "خطوة" المحسن. في حالة SWA، يتم عادةً تخطي هذا الأمر لبعض الأرقام من الخطوات في بداية التدريب.

استراتيجيات المتوسط ​​المخصصة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

بشكل افتراضي، :class:`torch.optim.swa_utils.AveragedModel` يحسب متوسط تشغيل متساوٍ
من المعلمات التي تقدمها، ولكن يمكنك أيضًا استخدام دالات متوسط مخصصة مع
وسيطة "avg_fn" أو "multi_avg_fn":

- ``avg_fn`` يسمح بتعريف دالة تعمل على كل زوج من المعلمات (معلمة متوسطة، معلمة نموذج) ويجب أن تعيد المعلمة المتوسطة الجديدة.
- ``multi_avg_fn`` يسمح بتعريف عمليات أكثر كفاءة تعمل على زوج من قوائم المعلمات، (قائمة المعلمات المتوسطة، قائمة معلمات النموذج)، في نفس الوقت، على سبيل المثال باستخدام وظائف ``torch._foreach*``. يجب أن تقوم هذه الدالة بتحديث المعلمات المتوسطة في مكانها.

في المثال التالي، يحسب "ema_model" متوسطًا متحركًا أسيًا باستخدام وسيط "avg_fn":

>>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
>>>     0.9 * averaged_model_parameter + 0.1 * model_parameter
>>> ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)


في المثال التالي، يحسب "ema_model" متوسطًا متحركًا أسيًا باستخدام وسيط "multi_avg_fn" الأكثر كفاءة:

>>> ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9))


جدول التعلم SWA
^^^^^^^^^^^^^^^^^^^

عادة، في SWA، يتم تعيين معدل التعلم إلى قيمة ثابتة عالية. :class:`SWALR` هو
جدول معدلات التعلم الذي يقلل معدل التعلم إلى قيمة ثابتة، ثم يحافظ عليه
ثابت. على سبيل المثال، يقوم الكود التالي بإنشاء جدول مواعيد يقلل معدل التعلم خطيًا من قيمته الأولية إلى 0.05 في 5 حقبات داخل كل مجموعة من المعلمات:

>>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, \
>>>         anneal_strategy="linear"، anneal_epochs=5، swa_lr=0.05)

يمكنك أيضًا استخدام التلاشي التلقائي إلى قيمة ثابتة بدلاً من التلاشي الخطي عن طريق تعيين
``anneal_strategy="cos"``.


الاهتمام بتطبيع الدُفعات
^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`update_bn` هي دالة مساعدة تسمح بحساب إحصائيات التطبيع الدفعي للنموذج SWA
على محمل البيانات المحدد ``loader`` في نهاية التدريب:

>>> torch.optim.swa_utils.update_bn(loader, swa_model)

:func:`update_bn` يطبق ``swa_model`` على كل عنصر في محمل البيانات ويحسب إحصائيات التنشيط
لكل طبقة تطبيع الدفعة في النموذج.

.. تحذير ::
يفترض :func:`update_bn` أن كل دفعة في محمل البيانات ``loader`` هي إما تنسورات أو قائمة من
التنسورات حيث العنصر الأول هو التنسور الذي يجب تطبيق الشبكة عليه ``swa_model``.
إذا كان محمل البيانات لديك له بنية مختلفة، فيمكنك تحديث إحصائيات التطبيع الدفعي لـ
"swa_model" عن طريق إجراء تمرير للأمام باستخدام "swa_model" على كل عنصر في مجموعة البيانات.



وضع كل شيء معًا: SWA
^^^^^^^^^^^^^^^^^^^^^^^^

في المثال أدناه، "swa_model" هو نموذج SWA الذي يتراكم متوسطات الأوزان.
نحن نقوم بتدريب النموذج لمدة 300 حقبة وننتقل إلى جدول معدل التعلم SWA
وابدأ في جمع متوسطات SWA للمعلمات في الحقبة 160:

>>> loader, optimizer, model, loss_fn = ...
>>> swa_model = torch.optim.swa_utils.AveragedModel(model)
>>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
>>> swa_start = 160
>>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
>>>
>>> for epoch in range(300):
>>>       for input, target in loader:
>>>           optimizer.zero_grad()
>>>           loss_fn(model(input), target).backward()
>>>           optimizer.step()
>>>       if epoch > swa_start:
>>>           swa_model.update_parameters(model)
>>>           swa_scheduler.step()
>>>       else:
>>>           scheduler.step()
>>>
>>> # تحديث إحصائيات التطبيع الدفعي للنموذج swa_model في النهاية
>>> torch.optim.swa_utils.update_bn(loader, swa_model)
>>> # استخدم النموذج swa_model للتنبؤ ببيانات الاختبار
>>> preds = swa_model(test_input)


وضع كل شيء معًا: EMA
^^^^^^^^^^^^^^^^^^^^^^^^

في المثال أدناه، "ema_model" هو نموذج EMA الذي يتراكم متوسطات الأوزان المتدهورة أسيًا بمعدل تدهور يبلغ 0.999.
نقوم بتدريب النموذج لمدة 300 حقبة ونبدأ في جمع متوسطات EMA على الفور.

>>> loader, optimizer, model, loss_fn = ...
>>> ema_model = torch.optim.swa_utils.AveragedModel(model, \
>>>             multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
>>>
>>> for epoch in range(300):
>>>       for input, target in loader:
>>>           optimizer.zero_grad()
>>>           loss_fn(model(input), target).backward()
>>>           optimizer.step()
>>>           ema_model.update_parameters(model)
>>>
>>> # تحديث إحصائيات التطبيع الدفعي للنموذج ema_model في النهاية
>>> torch.optim.swa_utils.update_bn(loader, ema_model)
>>> # استخدم النموذج ema_model للتنبؤ ببيانات الاختبار
>>> preds = ema_model(test_input)

.. autosummary::
:toctree: generated
:nosignatures:

swa_utils.AveragedModel
swa_utils.SWALR


.. autofunction:: torch.optim.swa_utils.get_ema_multi_avg_fn
.. autofunction:: torch.optim.swa_utils.update_bn


.. تحتاج هذه الوحدة إلى توثيق. إضافة هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.optim.adadelta
.. py:module:: torch.optim.adagrad
.. py:module:: torch.optim.adam
.. py:module:: torch.optim.adamax
.. py:module:: torch.optim.adamw
.. py:module:: torch.optim.asgd
.. py:module:: torch.optim.lbfgs
.. py:module:: torch.optim.lr_scheduler
.. py:module:: torch.optim.nadam
.. py:module:: torch.optim.optimizer
.. py:module:: torch.optim.radam
.. py:module:: torch.optim.rmsprop
.. py:module:: torch.optim.rprop
.. py:module:: torch.optim.sgd
.. py:module:: torch.optim.sparse_adam
.. py:module:: torch.optim.swa_utils