.. _troubleshooting:

استكشاف أخطاء PyTorch 2.0 وإصلاحها
===============================

هذه الصفحة مخصصة لمساعدتك في استكشاف الأخطاء وإصلاحها في PyTorch. إذا واجهتك مشكلة، يرجى الاطلاع على الأسئلة الشائعة التالية. إذا لم تجد حلاً لمشكلتك هنا، فيرجى فتح مشكلة على صفحة `GitHub Issues <https://github.com/pytorch/pytorch/issues>`_ الخاصة بنا.

.. contents:: جدول المحتويات
    :local:

أسئلة عامة
----------

ما هي متطلبات النظام لتشغيل PyTorch؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  PyTorch يدعم Windows وLinux وmacOS.
-  يتطلب PyTorch Python 3.6 أو أعلى.
-  يتطلب PyTorch `CUDA <https://developer.nvidia.com/cuda-downloads>`_ 9.2 أو أعلى إذا كنت ترغب في استخدام PyTorch مع GPU.
-  يتطلب PyTorch `cuDNN <https://developer.nvidia.com/cudnn>`_ 7.4.2 أو أعلى.

كيف يمكنني تثبيت PyTorch؟
^^^^^^^^^^^^^^^^^^^^^^

يمكنك العثور على تعليمات التثبيت التفصيلية في `صفحة التثبيت الخاصة بنا <https://pytorch.org/get-started/locally/>`_.

ما هي إصدارات PyTorch المدعومة حالياً؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يمكنك العثور على معلومات حول الإصدارات المدعومة حاليًا في `سياسة دعم الإصدارات الخاصة بنا <https://pytorch.org/resources/stable/version_policy.html>`_.

ما هي أفضل طريقة للتحديث إلى أحدث إصدار من PyTorch؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يرجى الاطلاع على `دليل الترقية <https://pytorch.org/resources/stable/notes/upgrade.html>`_ للحصول على إرشادات حول كيفية الترقية إلى أحدث إصدار من PyTorch.

أين يمكنني العثور على أحدث إصدار من PyTorch؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يمكنك دائمًا العثور على أحدث إصدار من PyTorch على موقعنا على الويب في `https://pytorch.org/get-started/ <https://pytorch.org/get-started/>`_.

ما هي الطريقة الموصى بها للتعامل مع الأخطاء في PyTorch؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا صادفت خطأ أثناء استخدام PyTorch، فيرجى فتح مشكلة على صفحة `GitHub Issues <https://github.com/pytorch/pytorch/issues>`_ الخاصة بنا. يرجى تضمين أكبر قدر ممكن من التفاصيل حول الخطأ، بما في ذلك أي رسائل خطأ أو آثار، بالإضافة إلى التعليمات البرمجية أو الخطوات اللازمة لإعادة إنتاج الخطأ.

أسئلة محددة
---------

لماذا أحصل على خطأ "لا يمكن استيراد الاسم 'XYZ'" عند محاولة استيراد PyTorch؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا واجهت خطأ مثل "لا يمكن استيراد الاسم 'XYZ'"، فهذا يعني أن PyTorch غير مثبت بشكل صحيح. يرجى التأكد من اتباعك `تعليمات التثبيت <https://pytorch.org/get-started/locally/>`_ بشكل صحيح، والتأكد من أنك قمت بتنشيط بيئة Python الخاصة بك بشكل صحيح.

لماذا لا يمكنني استيراد PyTorch في Jupyter Notebook؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا كنت تستخدم Jupyter Notebook، فتأكد من تشغيل دفتر الملاحظات في نفس البيئة التي قمت بتثبيت PyTorch فيها. يمكنك أيضًا محاولة إعادة تشغيل الخادم وإعادة فتح دفتر الملاحظات.

لماذا أحصل على خطأ "لا يمكن تحميل وحدة CUDA" عند تشغيل التعليمات البرمجية على GPU؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا واجهت خطأ "لا يمكن تحميل وحدة CUDA"، فتأكد من تثبيت CUDA وcuDNN بشكل صحيح ومتاح لـ PyTorch. يرجى الاطلاع على `متطلبات النظام <#what-are-the-system-requirements-to-run-pytorch>`_ للتأكد من أن لديك الإصدارات المدعومة من CUDA وcuDNN.

لماذا أحصل على خطأ "لا يوجد جهاز CUDA متوفر" عند محاولة تشغيل التعليمات البرمجية على GPU؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا واجهت خطأ "لا يوجد جهاز CUDA متوفر"، فتأكد من أن لديك بطاقة GPU متوافقة مع CUDA وأن PyTorch يمكنه الوصول إليها. يمكنك استخدام ``torch.cuda.is_available()`` للتحقق مما إذا كان PyTorch قادرًا على اكتشاف GPU.

لماذا أحصل على خطأ "ذاكرة CUDA خارج النطاق" عند تشغيل التعليمات البرمجية على GPU؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا واجهت خطأ "ذاكرة CUDA خارج النطاق"، فهذا يعني أنك تستخدم المزيد من ذاكرة GPU أكثر مما هو متاح. حاول تقليل حجم دفعة التعليمات البرمجية الخاصة بك أو تقليل عدد المتغيرات التي يتم تخزينها في GPU. يمكنك أيضًا محاولة استخدام بطاقة GPU ذات سعة ذاكرة أكبر.

لماذا أحصل على نتائج مختلفة عند تشغيل نفس التعليمات البرمجية على CPU وGPU؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

قد تحدث اختلافات طفيفة في النتائج عند تشغيل التعليمات البرمجية على CPU وGPU بسبب الاختلافات في الدقة العددية وتوازي الحسابات. في معظم الحالات، يجب ألا يكون لهذه الاختلافات تأثير كبير على الأداء العام لنموذجك.

لماذا بطء PyTorch عند تشغيله على CPU؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch مصمم للاستفادة من قدرات معالجة GPU، لذلك قد يكون أبطأ عند تشغيله على CPU. إذا كنت تواجه مشكلات في الأداء عند تشغيل PyTorch على CPU، فتأكد من أنك تستخدم أحدث إصدار من PyTorch، والذي يتضمن تحسينات للأداء على CPU. يمكنك أيضًا محاولة استخدام مكتبات مثل `Numba <https://numba.readthedocs.io/en/stable/cuda/overview.html>`_ أو `Cython <https://cython.org/>`_ لتسريع أجزاء حرجة من التعليمات البرمجية الخاصة بك.

لماذا لا يمكنني تثبيت PyTorch على Windows؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا كنت تواجه مشكلات في تثبيت PyTorch على Windows، فتأكد من أنك تستخدم `مفسر Python مدعوم <https://pytorch.org/get-started/locally/#windows-install-with-pip>`_ وأن لديك `Microsoft Visual C++ Redistributable <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ مثبتًا. إذا استمرت المشكلة، فيرجى الرجوع إلى `دليل استكشاف الأخطاء وإصلاحها الخاص بنا <https://pytorch.org/get-started/locally/#windows-troubleshooting>`_ لمزيد من الاقتراحات.

لماذا لا يمكنني تثبيت PyTorch على macOS؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا كنت تواجه مشكلات في تثبيت PyTorch على macOS، فتأكد من أن لديك الإصدار الصحيح من `Xcode <https://developer.apple.com/xcode/>`_ و `Xcode Command Line Tools <https://developer.apple.
com/downloads/all/>`_ مثبتًا. إذا استمرت المشكلة، فيرجى الرجوع إلى `دليل استكشاف الأخطاء وإصلاحها الخاص بنا <https://pytorch.org/get-started/locally/#macos-troubleshooting>`_ لمزيد من الاقتراحات.
**المؤلف**: `مايكل لازوس <https://github.com/mlazos>`_

نقوم حاليًا بتطوير أدوات التصحيح، ومُصَحِّحات الأخطاء، وتحسين رسائل الخطأ والتحذير لدينا. فيما يلي جدول بالأدوات المتاحة واستخداماتها النموذجية. للحصول على مساعدة إضافية، راجع `تشخيص أخطاء وقت التشغيل <#diagnosing-runtime-errors>`__.

.. list-table:: العنوان
   :widths: 25 25 50
   :header-rows: 1

   * - الأداة
     - الغرض
     - الاستخدام
   * - تسجيل المعلومات
     - عرض الخطوات الملخصة للتجميع
     - ``torch._logging.set_logs(dynamo = logging.INFO)`` أو ``TORCH_LOGS="dynamo"``
   * - تسجيل التصحيح
     - عرض الخطوات التفصيلية للتجميع (طباعة كل تعليمة تم تتبعها)
     - ``torch._logging.set_logs(dynamo = logging.DEBUG)`` و
       ``torch._dynamo.config.verbose = True``، أو ``TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1``
   * - المُصَغِّر لأي backend
     - البحث عن أصغر subgraph الذي يكرر الأخطاء لأي backend
     - قم بتعيين متغير البيئة ``TORCHDYNAMO_REPRO_AFTER="dynamo"``
   * - المُصَغِّر لـ ``TorchInductor``
     - إذا كان الخطأ معروفًا بعد حدوثه ``AOTAutograd`` ابحث
       أصغر subgraph الذي يكرر الأخطاء أثناء ``TorchInductor`` lowering
     - قم بتعيين متغير البيئة ``TORCHDYNAMO_REPRO_AFTER="aot"``
   * - مُصَغِّر دقة Dynamo
     - يجد أصغر subgraph الذي يكرر مشكلة دقة
       بين نموذج الوضع الحريص ونموذج مُحَسَّن، عندما تشتبه
       المشكلة في ``AOTAutograd``
     - ``TORCHDYNAMO_REPRO_AFTER="dynamo" TORCHDYNAMO_REPRO_LEVEL=4``
   * - مُصَغِّر دقة Inductor
     - يجد أصغر subgraph الذي يكرر مشكلة دقة
       بين نموذج الوضع الحريص ونموذج مُحَسَّن، عندما تشتبه
       المشكلة في backend (على سبيل المثال، inductor).
       إذا لم ينجح هذا، جرب مُصَغِّر دقة Dynamo
       بدلا من ذلك.
     - ``TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4``
   * - ``torch._dynamo.explain``
     - العثور على كسور الرسم البياني وعرض التبرير لها
     - ``torch._dynamo.explain(fn)(*inputs)``
   * - التسجيل/إعادة التشغيل
     - تسجيل وإعادة تشغيل الإطارات التي تتكرر الأخطاء أثناء التقاط الرسم البياني
     - ``torch._dynamo.config.replay_record_enabled = True``
   * - تصفية أسماء وظائف TorchDynamo
     - قم بتجميع الوظائف فقط مع الاسم المحدد لتقليل الضوضاء عند
       تصحيح مشكلة
     - قم بتعيين متغير البيئة ``TORCHDYNAMO_DEBUG_FUNCTION=<name>``
   * - تسجيل تصحيح TorchInductor
     - طباعة معلومات تصحيح الأخطاء العامة لـ TorchInductor والرمز المولد لـ Triton/C++
     - ``torch._inductor.config.debug = True``
   * - تتبع TorchInductor
     - عرض الوقت المستغرق في كل مرحلة من مراحل TorchInductor + إخراج التعليمات البرمجية وتصور الرسم البياني
     - قم بتعيين متغير البيئة TORCH_COMPILE_DEBUG=1 أو
       ``torch._inductor.config.trace.enabled = True``

بالإضافة إلى تسجيل المعلومات وتسجيل التصحيح،
يمكنك استخدام `torch._logging <https://pytorch.org/docs/main/logging.html>`__
للحصول على تسجيل أكثر تفصيلاً.

تشخيص أخطاء وقت التشغيل
~~~~~~~~~~~~~~~~~~~~~~~~~

على مستوى عالٍ، تتكون مكدس TorchDynamo من التقاط الرسم البياني
من رمز Python (TorchDynamo) ومُجَمِّع backend. على سبيل المثال، قد يتكون مُجَمِّع backend من تتبع الرسم البياني العكسي (AOTAutograd)
وتقليل الرسم البياني (TorchInductor)*. يمكن أن تحدث الأخطاء في أي مكون
من المكدس وسيوفر آثار المكدس الكاملة.

لتحديد المكون الذي حدث فيه الخطأ،
يمكنك استخدام تسجيل مستوى المعلومات
``torch._logging.set_logs(dynamo = logging.INFO)`` أو ``TORCH_LOGS="dynamo"``
والبحث عن مخرجات ``Step #: ...``. يتم إجراء السجلات في بداية ونهاية
كل خطوة، لذلك فإن الخطوة التي يجب أن يقابلها الخطأ هي أحدث خطوة تم تسجيلها
التي لم يتم تسجيل نهايتها بعد. تتوافق الخطوات مع
أجزاء المكدس التالية:

==== ================
الخطوة المكون
==== ================
1    TorchDynamo
2    مُجَمِّع backend
3    TorchInductor
==== ================

إذا كان تسجيل المعلومات غير كافٍ، فيمكنك استخدام خيارات backend
المتاحة. تشمل هذه الخيارات:

-  ``"eager"``: تشغيل TorchDynamo التقاط الرسم البياني للأمام فقط
   ثم قم بتشغيل الرسم البياني الذي تم التقاطه باستخدام PyTorch. يوفر هذا مؤشرا على
   ما إذا كان TorchDynamo يرفع الخطأ.

-  ``"aot_eager"``: تشغيل TorchDynamo لالتقاط الرسم البياني للأمام، و
   ثم AOTAutograd لتعقب الرسم البياني العكسي دون أي خطوات إضافية
   مُجَمِّع backend. سيتم بعد ذلك استخدام PyTorch eager لتشغيل
   الرسوم البيانية للأمام والخلف. هذا مفيد لتضييق المشكلة
   إلى AOTAutograd.

الإجراء العام لتضييق نطاق المشكلة هو ما يلي:

1. قم بتشغيل برنامجك باستخدام backend "eager". إذا لم يعد الخطأ يحدث،
   فإن المشكلة تكمن في مُجَمِّع backend الذي يتم استخدامه (إذا
   استخدام TorchInductor، انتقل إلى الخطوة 2. إذا لم يكن الأمر كذلك، راجع `هذا
   القسم <#minifying-backend-compiler-errors>`__). إذا استمر الخطأ
   مع backend "eager"، فهو `خطأ أثناء تشغيل torchdynamo <#torchdynamo-errors>`__.

2. هذه الخطوة ضرورية فقط إذا تم استخدام ``TorchInductor`` كمُجَمِّع backend
   . قم بتشغيل النموذج باستخدام backend "aot_eager". إذا قام هذا
   backend برفع خطأ، فإن الخطأ يحدث أثناء
   تتبع AOTAutograd. إذا لم يعد الخطأ يحدث مع هذا backend،
   ثم `الخطأ في
   TorchInductor\* <#minifying-torchinductor-errors>`__.

يتم تحليل كل من هذه الحالات في الفروع التالية.

.. note:: يتكون backend TorchInductor من
   كل من تتبع AOTAutograd ومُجَمِّع TorchInductor نفسه. سنقوم
   إزالة الغموض عن طريق الإشارة إلى ``TorchInductor`` كمُجَمِّع backend، و
   تقليل TorchInductor كمرحلة تقلل الرسم البياني الذي تم تتبع بواسطة
   AOTAutograd.

أخطاء Torchdynamo
------------------

إذا حدث الخطأ الذي تم إنشاؤه مع backend "eager"، فإن
TorchDynamo هو على الأرجح مصدر الخطأ. فيما يلي رمز العينة
التي ستولد خطأ.

.. code-block:: py

   import torch

   import torch._dynamo as dynamo


   def test_assertion_error():
       y = torch.ones(200, 200)
       z = {y: 5}
       return z

   compiled_test_assertion_error = torch.compile(test_assertion_error, backend="eager")

   compiled_test_assertion_error()

تولد الشفرة أعلاه الخطأ التالي:

::

   torch._dynamo.convert_frame: [ERROR] WON'T CONVERT test_assertion_error /scratch/mlazos/torchdynamo/../test/errors.py line 26
   due to:
   Traceback (most recent call last):
     File "/scratch/mlazos/torchdynamo/torchdynamo/symbolic_convert.py", line 837, in BUILD_MAP
       assert isinstance(k, ConstantVariable) or (
   AssertionError

   from user code:
      File "/scratch/mlazos/torchdynamo/../test/errors.py", line 34, in test_assertion_error
       z = {y: 5}

   Set torch._dynamo.config.verbose=True for more information
   ==========

كما يوحي الرسالة، يمكنك تعيين
``torch._dynamo.config.verbose=True`` للحصول على أثر المكدس الكامل للخطأ
في TorchDynamo ورمز المستخدم. بالإضافة إلى هذا العلم،
يمكنك أيضًا تعيين مستوى تسجيل الدخول لـ TorchDynamo من خلال
``torch._logging.set_logs(dynamo = logging.INFO)`` أو ``TORCH_LOGS="dynamo"``. تشمل هذه المستويات:

- ``logging.DEBUG`` أو ``TORCH_LOGS="+dynamo"``: طباعة كل تعليمة
   يتم مواجهتها بالإضافة إلى جميع مستويات تسجيل الدخول المدرجة أدناه.
- ``logging.INFO``:
   طباعة كل دالة يتم تجميعها (رمز البايت الأصلي والمعدل)
   والرسوم البيانية التي يتم التقاطها بالإضافة إلى جميع مستويات تسجيل الدخول المدرجة أدناه.
- ``logging.WARNING`` (افتراضي): طباعة كسور الرسم البياني بالإضافة إلى جميع
   مستويات تسجيل الدخول المدرجة أدناه.
- ``logging.ERROR``: طباعة الأخطاء فقط.

إذا كان النموذج كبيرًا جدًا، فقد تصبح السجلات ساحقة. إذا
حدث خطأ عميقًا داخل رمز Python للنموذج، فقد يكون من المفيد
تنفيذ الإطار الذي يحدث فيه الخطأ فقط لتمكين تصحيح الأخطاء بشكل أسهل. هناك أداتان متاحتان لتمكين ذلك:

- قم بتعيين متغير البيئة ``TORCHDYNAMO_DEBUG_FUNCTION``
  إلى اسم الدالة المطلوبة لتشغيل torchdynamo فقط على الوظائف التي تحمل هذا الاسم.

- تمكين أداة التسجيل/إعادة التشغيل (تعيين ``torch._dynamo.config.replay_record_enabled = True``)
  الذي يقوم بإلقاء سجل التنفيذ عند مواجهة خطأ. يمكن بعد ذلك إعادة تشغيل هذا السجل
  لتشغيل الإطار فقط حيث حدث خطأ.

تشخيص أخطاء TorchInductor
-------------------------------

إذا لم يحدث الخطأ مع backend "eager"، فإن
مُجَمِّع backend هو مصدر الخطأ (`مثال
خطأ <https://gist.github.com/mlazos/2f13681e3cc6c43b3911f336327032de%5D>`__).
هناك `خيارات مختلفة <./torch.compiler.rst>`__
لمُجَمِّعات backend لـ TorchDynamo، مع TorchInductor
تلبية احتياجات معظم المستخدمين. يركز هذا القسم على TorchInductor
كمثال توضيحي، ولكن يمكن استخدام بعض الأدوات أيضًا مع مُجَمِّعات backend الأخرى.

فيما يلي الجزء من المكدس الذي نركز عليه:

مع اختيار TorchInductor كمُجَمِّع backend، يتم استخدام AOTAutograd
لإنشاء الرسم البياني العكسي من الرسم البياني للأمام الذي تم التقاطه بواسطة
torchdynamo. من المهم ملاحظة أنه يمكن أن تحدث أخطاء أثناء هذا
التتبع وأيضًا أثناء قيام TorchInductor بتقليل الرسوم البيانية للأمام والخلف إلى رمز GPU أو C++. غالبًا ما يتكون النموذج من مئات أو
آلاف عقد FX، لذلك قد يكون من الصعب تضييق العقدة التي حدثت فيها المشكلة
بشكل كبير. لحسن الحظ، هناك أدوات متاحة ل
تصغير هذه الرسوم البيانية المدخلة تلقائيًا إلى العقد التي تسبب
المشكلة. تتمثل الخطوة الأولى في تحديد ما إذا كان الخطأ يحدث
أثناء تتبع الرسم البياني العكسي باستخدام AOTAutograd أو أثناء تقليل TorchInductor. كما ذكرنا أعلاه في الخطوة 2، يمكن استخدام backend
"aot_eager" لتشغيل AOTAutograd في عزلة دون تقليل. إذا استمر الخطأ في الحدوث مع هذا backend،
فذلك يشير إلى أن الخطأ يحدث أثناء تتبع AOTAutograd.

فيما يلي مثال:

.. code-block:: py

   import torch

   import torch._dynamo as dynamo

   model = torch.nn.Sequential(*[torch.nn.Linear(200, 200) for _ in range(5)])

   def test_backend_error():

       y = torch.ones(200, 200)
       x = torch.ones(200, 200)
       z = x + y
       a = torch.ops.aten._foobar(z)  # وظيفة وهمية بها خطأ
       return model(a)


   compiled_test_backend_error = torch.compile(test_backend_error, backend="inductor")
   compiled_test_backend_error()

يجب أن يعطيك هذا الخطأ مع أثر المكدس الأطول أدناه

::

   Traceback (most recent call last):
     File "/scratch/mlazos/torchdynamo/torchinductor/graph.py", line 246, in call_function
       return lowerings[target](*args, **kwargs)
     File "/scratch/mlazos/torchdynamo/torchinductor/lowering.py", line 185, in wrapped
       return decomp_fn(*args, **kwargs)
     File "/scratch/mlazos/torchdynamo/torchinductor/lowering.py", line 810, in _foobar
       assert False
   AssertionError
   ...

`خطأ مع أثر المكدس الكامل
<https://gist.github.com/mlazos/d6947854aa56d686800259a164c62100>`__

إذا قمت بعد ذلك بتغيير ``torch.compile(backend="inductor")`` إلى
``torch.compile(backend="aot_eager")``، فسيتم تشغيله دون خطأ، لأن
`القضية <https://github.com/pytorch/torchdynamo/blob/d09e50fbee388d466b5252a63045643166006f77/torchinductor/lowering.py#:~:text=%23%20This%20shouldn%27t%20be,assert%20False>`__
في عملية تقليل TorchInductor، وليس في AOTAutograd.

تصغير أخطاء TorchInductor
-------------------------

من هنا، دعنا نقوم بتشغيل أداة التبسيط (minifier) للحصول على نسخة مبسطة يمكن إعادة إنتاجها. سيؤدي تعيين متغير البيئة ``TORCHDYNAMO_REPRO_AFTER="aot"`` (أو تعيين ``torch._dynamo.config.repro_after="aot"`` مباشرةً) إلى إنشاء برنامج Python يقوم بتبسيط الرسم البياني الذي ينتجه AOTAutograd إلى أصغر رسم فرعي يعيد إنتاج الخطأ. (انظر أدناه لمثال حيث نقوم بتبسيط الرسم البياني الذي ينتجه TorchDynamo) يجب أن يؤدي تشغيل البرنامج مع متغير البيئة هذا إلى إخراج متطابق تقريبًا، مع إضافة سطر يشير إلى المكان الذي تم فيه كتابة ``minifier_launcher.py``. يمكن تكوين دليل الإخراج عن طريق تعيين ``torch._dynamo.config.base_dir`` إلى اسم دليل صالح. الخطوة الأخيرة هي تشغيل أداة التبسيط والتحقق من تشغيلها بنجاح. يبدو التشغيل الناجح مثل
`هذا <https://gist.github.com/mlazos/e6ea41ccce68a7b1b8a7a09acb1b206a>`__.
إذا نجحت أداة التبسيط، فستولد كود Python قابل للتنفيذ يعيد إنتاج الخطأ بالضبط. بالنسبة لمثالنا، يكون الكود على النحو التالي:

.. code-block:: python

   import torch
   from torch import tensor, device
   import torch.fx as fx
   from torch._dynamo.testing import rand_strided
   from math import inf
   from torch.fx.experimental.proxy_tensor import make_fx

   # نسخة PyTorch: 1.13.0a0+gitfddfc44
   # نسخة CUDA في PyTorch: 11.6
   # نسخة Git في PyTorch: fddfc4488afb207971c54ad4bf58130fdc8a4dc5


   # معلومات CUDA:
   # nvcc: برنامج تجميع CUDA من NVIDIA (R)
   # حقوق النشر (ج) 2005-2022 لشركة NVIDIA Corporation
   # تم البناء يوم: Thu_Feb_10_18:23:41_PST_2022
   # أدوات التجميع CUDA، الإصدار 11.6، V11.6.112
   # إنشاء cuda_11.6.r11.6/compiler.30978841_0

   # معلومات عتاد GPU:
   # NVIDIA A100-SXM4-40GB: 8

   من torch.nn استورد *

   class Repro(torch.nn.Module):
       def __init__(self):
           super().__init__()

       def forward(self, add):
           _foobar = torch.ops.aten._foobar.default(add);  add = None
           return (_foobar,)

   args = [((200, 200), (200, 1), torch.float32, 'cpu')]
   args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]
   mod = make_fx(Repro())(*args)
   from torch._inductor.compile_fx import compile_fx_inner

   compiled = compile_fx_inner(mod, args)
   compiled(*args)

تحتوي طريقة ``forward`` في وحدة ``Repro`` على العملية الدقيقة التي تسبب المشكلة. عند إرسال مشكلة، يرجى تضمين أي نسخ مبسطة يمكن إعادة إنتاجها للمساعدة في تصحيح الأخطاء.

تبسيط أخطاء مجمع المؤخرة
--------------------

مع مجمعات المؤخرة الأخرى بخلاف TorchInductor، تكون عملية العثور على الرسم الفرعي الذي يسبب الخطأ مماثلة تقريبًا للإجراء المتبع في `الأخطاء في TorchInductor <#torchinductor-errors>`__ مع تحذير مهم واحد. وهو أن أداة التبسيط ستعمل الآن على الرسم البياني الذي يتم تتبعه بواسطة TorchDynamo، وليس الرسم البياني الناتج عن AOTAutograd. دعنا نتعمق في مثال.

.. code-block:: py

   import torch

   import torch._dynamo as dynamo

   model = torch.nn.Sequential(*[torch.nn.Linear(200, 200) for _ in range(5)])
   # مجمع تجريبي يفشل إذا احتوى الرسم البياني على relu
   def toy_compiler(gm: torch.fx.GraphModule, _):
       for node in gm.graph.nodes:
           if node.target == torch.relu:
               assert False

       return gm


   def test_backend_error():
       y = torch.ones(200, 200)
       x = torch.ones(200, 200)
       z = x + y
       a = torch.relu(z)
       return model(a)


   compiled_test_backend_error = torch.compile(test_backend_error, backend=toy_compiler)
   compiled_test_backend_error()

لتشغيل الكود بعد أن يقوم TorchDynamo بتتبع الرسم البياني للأمام، يمكنك استخدام متغير البيئة ``TORCHDYNAMO_REPRO_AFTER``. يجب أن يؤدي تشغيل هذا البرنامج مع ``TORCHDYNAMO_REPRO_AFTER="dynamo"`` (أو ``torch._dynamo.config.repro_after="dynamo"``) إلى إنتاج `هذا
الإخراج <https://gist.github.com/mlazos/244e3d5b53667e44078e194762c0c92b>`__\ وكود على النحو التالي في ``{torch._dynamo.config.base_dir}/repro.py``.

.. note:: الخيار الآخر لـ TORCHDYNAMO_REPRO_AFTER هو ``"aot"``، والذي
   سيقوم بتشغيل أداة التبسيط بعد إنشاء الرسم البياني للخلف.

.. code-block:: python

   import torch
   import torch._dynamo as dynamo
   from torch import tensor, device
   import torch.fx as fx
   from torch._dynamo.testing import rand_strided
   from math import inf
   from torch._dynamo.debug_utils import run_fwd_maybe_bwd

   from torch.nn import *

   class Repro(torch.nn.Module):
       def __init__(self):
           super().__init__()

       def forward(self, add):
           relu = torch.relu(add);  add = None
           return (relu,)


   mod = Repro().cuda()
   opt_mod = torch.compile(mod, backend="None")


   args = [((200, 200), (200, 1), torch.float32, 'cpu', False)]
   args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


   with torch.cuda.amp.autocast(enabled=False):
       ref = run_fwd_maybe_bwd(mod, args)
       res = run_fwd_maybe_bwd(opt_mod, args)

نجحت أداة التبسيط في تقليل الرسم البياني إلى العملية التي تسبب الخطأ في ``toy_compiler``. والاختلاف الآخر عن الإجراء المتبع في `أخطاء TorchInductor <#torchinductor-errors>`__ هو أن أداة التبسيط تعمل تلقائيًا بعد مواجهة خطأ في مجمع المؤخرة. بعد تشغيل ناجح، تقوم أداة التبسيط بكتابة ``repro.py`` إلى ``torch._dynamo.config.base_dir``.

تحليل الأداء
~~~~~~~~~~

الوصول إلى ملف تعريف TorchDynamo
------------------------------

يحتوي TorchDynamo على دالة إحصائيات مدمجة لجمع وعرض الوقت المستغرق في كل مرحلة من مراحل التجميع. يمكن الوصول إلى هذه الإحصائيات عن طريق استدعاء ``torch._dynamo.utils.compile_times()`` بعد تنفيذ Torch._Dynamo. بشكل افتراضي، تقوم الدالة بإرجاع تمثيل نصي لأوقات التجميع المستغرقة في كل دالة TorchDynamo حسب الاسم.

تصحيح الأخطاء في TorchInductor باستخدام TORCH_COMPILE_DEBUG
-----------------------------------------------------

يحتوي TorchInductor على دالة مدمجة للتعقب والإحصاءات لعرض الوقت المستغرق في كل مرحلة من مراحل التجميع، وكود الإخراج، وتصوير الرسم البياني للإخراج، وإلقاء IR. تعد هذه الأداة مصممة لتسهيل فهم واستكشاف أخطاء TorchInductor الداخلية.

دعنا نتعمق في مثال باستخدام برنامج الاختبار التالي (``repro.py``):

::

  import torch

  @torch.compile()
  def test_model(x):
      model = torch.nn.Sequential(
          torch.nn.Linear(10, 10),
          torch.nn.LayerNorm(10),
          torch.nn.ReLU(),
      )
      return model(x)


  y = test_model(torch.ones(10, 10))

سيؤدي تعيين متغير البيئة ``TORCH_COMPILE_DEBUG=1`` إلى إنشاء دليل تعقب للتصحيح، بشكل افتراضي، سيكون هذا الدليل في الدليل الحالي ويُسمى torch_compile_debug (يمكن تجاوز هذا الإعداد في حقل التكوين torchdynamo ``debug_dir_root`` ومتغير البيئة ``TORCH_COMPILE_DEBUG_DIR`` أيضًا). داخل هذا الدليل، سيكون لكل عملية تشغيل مجلد منفصل باسم ختم التاريخ والوقت وعملية التعريف:

::

   $ env TORCH_COMPILE_DEBUG=1 python repro.py
   $ cd torch_compile_debug
   $ ls
   run_2023_03_01_08_20_52_143510-pid_180167

في مجلد العملية، سيكون هناك مجلد ``torchdynamo`` يحتوي على سجلات التصحيح، ومجلد ``torchinductor`` يحتوي على مجلد فرعي لكل نواة مجمعة مع آثار تصحيح الأخطاء في Inductor.

::

   $ cd
   run_2023_03_01_08_20_52_143510-pid_180167
   $ ls
   torchinductor  torchdynamo

عند الدخول أكثر إلى مجلد ``torchinductor``، تكون ملفات ``\*.log`` هي سجلات من مرحلة AOT Autograd من التجميع، ويحتوي ``model__0_forward_1.0`` على آثار تصحيح الأخطاء في Inductor.

::

   $ cd torchinductor
   $ ls
   aot_model___0_debug.log  model__0_forward_1.0
   $ cd model__0_forward_1.0
   $ ls
   debug.log  fx_graph_readable.py  fx_graph_runnable.py  fx_graph_transformed.py  ir_post_fusion.txt  ir_pre_fusion.txt  output_code.py

فيما يلي ملخص للمحتويات:

- ``fx_graph_readable.py`` و ``fx_graph_runnable.py`` هما الإصدارات القابلة للقراءة والقابلة للتنفيذ من ``fx_graph`` التي تلقاها inductor.
- ``fx_graph_transformed.py`` هو الرسم البياني fx بعد أن قام inductor بتشغيل جميع تمريرات fx.
- ``ir\*.txt`` هو IR inductor قبل وبعد الانصهار.
- ``output_code.py`` هو نواة Triton المجمعة للرسم البياني الفرعي.

فيما يلي `محتويات دليل التصحيح
المثال <https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396>`__
لبرنامج الاختبار:

::

  import torch

  @torch.compile()
  def test_model(x):
      model = torch.nn.Sequential(
          torch.nn.Linear(10, 10),
          torch.nn.LayerNorm(10),
          torch.nn.ReLU(),
      )
      return model(x)


  y = test_model(torch.ones(10, 10))

يمكن تمكين كل ملف في تنسيق التصحيح الجديد هذا وتعطيله من خلال ``torch._inductor.config.trace.*``. يتم تعطيل المخطط والتعريف بشكل افتراضي لأنهما مكلفان في الإنشاء.

تبدو العقدة الفردية في تنسيق التصحيح الجديد هذا كما يلي:

::

   buf1: SchedulerNode(ComputedBuffer)
   buf1.writes =
       {   MemoryDep(name='buf1', index=0, size=()),
           MemoryDep(name='buf1', index=0, size=(s0,))}
   buf1.unmet_dependencies = {MemoryDep(name='buf0', index=c0, size=(s0,))}
   buf1.met_dependencies = {MemoryDep(name='primals_2', index=c0, size=(s0,))}
   buf1.group.device = cuda:0
   buf1.group.iteration = (1, s0)
   buf1.sizes = ([], [s0])
   class buf1_loop_body:
       var_ranges = {z0: s0}
       index0 = z0
       index1 = 0
       def body(self, ops):
           get_index = self.get_index('index0')
           load = ops.load('buf0', get_index, False)
           get_index_1 = self.get_index('index0')
           load_1 = ops.load('primals_2', get_index_1, False)
           add = ops.add(load, load_1)
           get_index_2 = self.get_index('index1')
           reduction = ops.reduction('buf1', torch.float32, torch.float32, 'sum', get_index_2, add)
           return reduction

راجع `إخراج دليل التصحيح
المثال <https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396>`__
لمزيد من الأمثلة.

..  _Memory Profiling
  ----------------

  TBD كسر الرسم البياني
---------------------

بالنسبة لبرنامج مثل هذا:

.. code-block:: python

   def some_fun(x):
       ...

   compiled_fun = torch.compile(some_fun, ...)
   ...

سيحاول TorchDynamo تجميع جميع عمليات torch/tensor داخل some_fun في مخطط FX واحد، ولكنه قد يفشل في التقاط كل شيء في مخطط واحد.

قد تكون بعض أسباب كسر المخطط غير قابلة للتغلب عليها بالنسبة لـ TorchDynamo، ولا يمكن إصلاحها بسهولة. - تعتبر الاستدعاءات إلى امتداد C بخلاف torch غير مرئية لـ torchdynamo، وقد تقوم بأشياء عشوائية دون أن يتمكن TorchDynamo من تقديم الضمانات اللازمة (راجع :ref: making-dynamo-sound-guards) ) لضمان أن يكون البرنامج المجمع آمنًا لإعادة الاستخدام. يمكن أن تعوق كسور المخطط الأداء إذا كانت الشظايا الناتجة صغيرة. ولتحقيق أقصى قدر من الأداء، من المهم تقليل عدد كسور المخطط إلى الحد الأدنى.

تحديد سبب كسر المخطط
~~~~~~~~~~~~~~~~~~~~~~

لتحديد جميع كسور المخطط في برنامج وأسباب الكسور المرتبطة، يمكن استخدام "torch._dynamo.explain". تقوم هذه الأداة بتشغيل TorchDynamo على الدالة المقدمة وتجميع كسور المخطط التي يتم مواجهتها. فيما يلي مثال على الاستخدام:

.. code-block:: python

   import torch
   import torch._dynamo as dynamo
   def toy_example(a, b):
       x = a / (torch.abs(a) + 1)
       print("woo")
       if b.sum() < 0:
           b = b * -1
       return x * b
   explanation = dynamo.explain(toy_example)(torch.randn(10), torch.randn(10))
   print(explanation_verbose)
   """
   عدد المخططات: 3
   عدد كسور المخطط: 2
   عدد العمليات: 5
   أسباب الكسر:
     سبب الكسر 1:
       السبب: builtin: print [<class 'torch._dynamo.variables.constant.ConstantVariable'>] False
       مكدس المستخدم:
         <FrameSummary file foo.py, line 5 in toy_example>
     سبب الكسر 2:
       السبب: generic_jump TensorVariable()
       مكدس المستخدم:
         <FrameSummary file foo.py, line 6 in torch_dynamo_resume_in_toy_example_at_5>
   العمليات لكل مخطط:
     ...
   الضمانات الخارجية:
     ...
   """

تشمل النواتج ما يلي:

- ``out_guards`` - قائمة قوائم حيث تحتوي كل قائمة فرعية على الضمانات التي يجب أن تتحقق لضمان صحة المخططات التي تم تتبعها.
- ``graphs`` - قائمة وحدات المخطط التي تم تتبعها بنجاح.
- ``ops_per_graph`` - قائمة قوائم حيث تحتوي كل قائمة فرعية على العمليات التي يتم تشغيلها في المخطط.

لإلقاء خطأ عند أول كسر مخطط يتم مواجهته، استخدم وضع "fullgraph". يعطل هذا الوضع عملية الرجوع إلى Python في TorchDynamo، ولا ينجح إلا إذا كان البرنامج بأكمله قابلًا للتحويل إلى مخطط واحد. مثال على الاستخدام:

.. code-block:: python

   def toy_example(a, b):
      ...

   compiled_toy = torch.compile(toy_example, fullgraph=True, backend=<compiler>)(a, b)

إعادة التجميع المفرطة
----------------

عندما يقوم TorchDynamo بتجميع دالة (أو جزء منها)، فإنه يقوم بعمل افتراضات معينة حول المتغيرات المحلية والعالمية للسماح بتحسين المترجم، ويعبر عن هذه الافتراضات على أنها ضمانات للتحقق من قيم معينة في وقت التشغيل. إذا فشل أي من هذه الضمانات، فسوف يقوم Dynamo بإعادة تجميع تلك الدالة (أو الجزء) حتى
``torch._dynamo.config.cache_size_limit`` مرات. إذا كان برنامجك يصل إلى حد ذاكرة التخزين المؤقت، فستحتاج أولاً إلى تحديد الضمان الذي فشل والجزء من برنامجك الذي يفعله.

يقوم "مُحسِّن التجميع" <https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/utils.py>`__ بتشغيل عملية
ضبط حد ذاكرة التخزين المؤقت لـ TorchDynamo إلى 1 وتشغيل برنامجك في ظل "مُجمِّع" للمراقبة فقط يقوم بتسجيل أسباب أي فشل في الضمان. يجب التأكد من تشغيل برنامجك لمدة لا تقل عن المدة (عدد التكرارات) التي كنت تعمل خلالها عندما واجهت مشكلة، وسيقوم المحسن بتجميع الإحصائيات خلال هذه المدة.

إذا أظهر برنامجك قدرًا محدودًا من الديناميكية، فقد تتمكن من ضبط حد ذاكرة التخزين المؤقت لـ TorchDynamo للسماح بتجميع وتخزين كل تنوع في الذاكرة المؤقتة في الذاكرة، ولكن إذا كان حد ذاكرة التخزين المؤقت مرتفعًا جدًا، فقد تجد أن تكلفة إعادة التجميع تفوق فوائد التحسين.

::

   torch._dynamo.config.cache_size_limit = <your desired cache limit>

يخطط TorchDynamo لدعم العديد من الحالات الشائعة للأشكال الديناميكية للموتر، مثل حجم الدفعة المتغير أو طول التسلسل. ولا يخطط لدعم ديناميكية الرتبة. في الوقت نفسه، يمكن استخدام تعيين حد ذاكرة تخزين مؤقت محدد بالتنسيق مع تقنيات التجميع لتحقيق عدد مقبول من عمليات إعادة التجميع لبعض النماذج الديناميكية.

.. code-block:: python

   from torch._dynamo.utils import CompileProfiler

   def my_model():
       ...

   with CompileProfiler() as prof:
       profiler_model = torch.compile(my_model, backend=prof)
       profiler_model()
       print(prof.report())

تصحيح الأخطاء الدقيقة
~~~~~~~~~~~~~~~~

يمكن أيضًا تصغير مشكلات الدقة إذا قمت بتعيين متغير البيئة
``TORCHDYNAMO_REPRO_LEVEL=4``، فهو يعمل بنموذج git bisect مشابه وقد يكون إعادة إنتاج كاملة شيئًا مثل
``TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4`` السبب
نحن بحاجة إلى هذا هو أن المترجمين في الأسفل سيقومون بتوليد التعليمات البرمجية سواء كان ذلك
رمز Triton أو backend C++، ويمكن أن تختلف الأرقام من هذه المترجمات في الأسفل
طرق دقيقة بعد أن يكون لها تأثير كبير على استقرار التدريب الخاص بك. لذلك، فإن مصحح الأخطاء الدقيق مفيد جدًا لنا
لاكتشاف الأخطاء في توليد التعليمات البرمجية الخاصة بنا أو مع مترجم backend.

إذا كنت ترغب في التأكد من أن توليد الأرقام العشوائية هو نفسه عبر كل من torch وtriton، فيمكنك تمكين ``torch._inductor.config.fallback_random = True``

تصحيح الأخطاء الممتد
~~~~~~~~~~~~~~~~

يمكن تمكين تصحيح الأخطاء الممتد باستخدام الأعلام التجريبية التالية.

``TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED`` - يوفر معلومات تصحيح الأخطاء الممتدة إذا تطابق التمثيل النصي للضمان قيمة هذا العلم. على سبيل المثال، قم بتعيينه على "Ne(s0, 10)" لتوليد تتبع مكدس Python وC++ الكامل كلما تم إصدار الضمان.
``TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL`` - يوفر معلومات تصحيح الأخطاء الممتدة عند تخصيص رمز معين. على سبيل المثال، قم بتعيين هذا إلى "u2" لتوليد تتبع مكدس Python وC++ الكامل كلما تم إنشاء هذا الرمز.
``TORCHDYNAMO_EXTENDED_DEBUG_CPP`` - يوفر معلومات تصحيح الأخطاء الممتدة (تتبع مكدس C++)
لجميع إعدادات تصحيح الأخطاء الممتدة بالإضافة إلى الأخطاء. على سبيل المثال، قم بتعيين هذا إلى "1". تتبع مكدس C++ بطيء جدًا ومزعج للغاية، لذلك لا يتم تضمينه بشكل افتراضي مع تصحيح الأخطاء الممتد.

تصحيح أخطاء توقيت بدء التشغيل البارد والفساد في ذاكرة التخزين المؤقت
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

من أجل قياس وقت تجميع بدء التشغيل البارد أو تصحيح أخطاء تلف ذاكرة التخزين المؤقت،
من الممكن تمرير ``TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`` أو تعيين
``torch._inductor.config.force_disable_caches = True`` الذي سيؤدي إلى تجاوز أي
خيار تكوين ذاكرة التخزين المؤقت الأخرى وتعطيل جميع ذاكرات التخزين المؤقت في وقت التجميع.