.. _torchscript:

TorchScript
===========

TorchScript هو نموذج البرمجة في PyTorch الذي يسمح لك بتشغيل نماذجك في بيئة الإنتاج. TorchScript مبني على نفس محرك التنفيذ المستخدم أثناء التدريب، لذلك يمكنك الانتقال من النماذج النصية إلى النشر دون أي تغييرات.

هناك طريقتان رئيسيتان للعمل مع TorchScript:

1. تتبع النموذج: يمكنك ببساطة تتبع نموذج PyTorch الخاص بك باستخدام `` torch.jit.trace `` ، والذي يسجل عمليات النموذج ويبني تمثيلا قابلا للتنفيذ. هذا مفيد للنماذج البسيطة أو إذا كان لديك نموذج تم تدريبه بالفعل وتريد تشغيله في بيئة الإنتاج.

2. النماذج النصية: يمكنك أيضًا كتابة نماذج PyTorch الخاصة بك باستخدام بناء جملة النص الفريد، والذي يسمح لك بتعريف وظائف وسلوكيات مخصصة. يوفر هذا المزيد من المرونة في كيفية تعريف النماذج الخاصة بك ويمكن أن يحسن الأداء.

يوفر TorchScript العديد من المزايا لتشغيل نماذج PyTorch في الإنتاج، بما في ذلك:

- الأداء: يمكن لنماذج TorchScript الاستفادة من التحسينات التي تمكنها من العمل بشكل أسرع، خاصة على أجهزة GPU.

- النشر: يمكنك نشر نماذج TorchScript على أجهزة مختلفة، بما في ذلك الخوادم وأجهزة الجوال والأجهزة المدمجة، دون الحاجة إلى الاعتماد على وقت تشغيل Python.

- التوافق: يعمل TorchScript مع معظم وحدات PyTorch، مما يتيح لك استخدام مجموعة واسعة من الأدوات والوظائف في نماذجك.

- قابلية التوسع: يدعم TorchScript التوزيع التلقائي، مما يسهل تشغيل نماذجك عبر أجهزة متعددة لتحسين الأداء.

لمزيد من المعلومات حول TorchScript، يرجى الاطلاع على `دليل TorchScript <https://pytorch.org/docs/stable/jit.html>`_ في وثائق PyTorch الرسمية.

مثال على استخدام TorchScript:

.. code:: python

   import torch
   from torch import nn

   class MyModel(nn.Module):
       def __init__(self):
           super(MyModel, self).__init__()
           self.linear = nn.Linear(10، 1)

       def forward(self، x):
           y = self.linear(x)
           return y

   # قم بإنشاء مثيل للنموذج
   model = MyModel()

   # قم بتعريف مدخلات نموذج عينة
   example_input = torch.rand(2, 10)

   # قم بتتبع النموذج لإنشاء نموذج TorchScript
   traced_model = torch.jit.trace(model، example_input)

   # قم بتشغيل نموذج TorchScript
   traced_model(example_input)

في هذا المثال، نقوم أولاً بتحديد نموذج PyTorch بسيط باستخدام `` nn.Module ``. ثم نقوم بإنشاء مثيل للنموذج ونحدد مدخلات نموذج عينة. بعد ذلك، نقوم بتتبع النموذج باستخدام `` torch.jit.trace ``، والذي يسجل العمليات التي يتم تنفيذها على النموذج وإنشاء تمثيل TorchScript القابل للتنفيذ. وأخيرًا، يمكننا تشغيل نموذج TorchScript هذا باستخدام نفس مدخلات نموذج العينة.

TorchScript هو أداة قوية تتيح لك الاستفادة من نماذج PyTorch في بيئة الإنتاج، وتقديم مزايا الأداء وقابلية النشر مع الحفاظ على التوافق والمرونة في تعريف النماذج الخاصة بك.


===========

.. toctree::
   :maxdepth: 1
   :caption: الوظائف المدمجة
   :hidden:

   torch.jit.supported_ops <jit_builtin_functions>

.. toctree::
   :maxdepth: 1
   :caption: مرجع اللغة
   :hidden:

   jit_language_reference

.. toctree::
   :maxdepth: 1

   jit_language_reference_v2

.. contents:: :local:
   :depth: 2

.. automodule:: torch.jit
.. currentmodule:: torch.jit

TorchScript هي طريقة لإنشاء نماذج قابلة للتسلسل والتحسين من كود PyTorch.
يمكن حفظ أي برنامج TorchScript من عملية Python
وتحميله في عملية لا يوجد بها اعتماد على Python.

نقدم أدوات للانتقال التدريجي لنموذج من برنامج Python نقي
إلى برنامج TorchScript يمكن تشغيله بشكل مستقل عن Python، كما هو الحال في برنامج C++ مستقل.
هذا يجعل من الممكن تدريب النماذج في PyTorch باستخدام أدوات مألوفة في Python ثم تصدير
النموذج عبر TorchScript إلى بيئة إنتاج حيث قد تكون برامج Python غير مناسبة
لأسباب تتعلق بالأداء وتعدد الخيوط.

للحصول على مقدمة سهلة إلى TorchScript، راجع البرنامج التعليمي `مقدمة إلى TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_.

للحصول على مثال شامل حول تحويل نموذج PyTorch إلى TorchScript وتشغيله في C++، راجع
البرنامج التعليمي `تحميل نموذج PyTorch في C++ <https://pytorch.org/tutorials/advanced/cpp_export.html>`_.

إنشاء كود TorchScript
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   script
   trace
   script_if_tracing
   trace_module
   fork
   wait
   ScriptModule
   ScriptFunction
   freeze
   optimize_for_inference
   enable_onednn_fusion
   onednn_fusion_enabled
   set_fusion_strategy
   strict_fusion
   save
   load
   ignore
   unused
   interface
   isinstance
   Attribute
   annotate

مزج التعقب والكتابة
---------------

في العديد من الحالات، يكون التتبع أو الكتابة أسهل نهج لتحويل نموذج إلى TorchScript.
يمكن تكوين التتبع والكتابة لتلبية المتطلبات الخاصة
جزء من نموذج.

يمكن لوظائف النص المكتوب استدعاء الوظائف المتبعة. هذا مفيد بشكل خاص عندما تحتاج
إلى استخدام التحكم في التدفق حول نموذج التغذية الأمامية البسيط. على سبيل المثال، ستتم كتابة البحث الشعاعي
لنموذج تسلسل إلى تسلسل بشكل نموذجي في النص، ولكنه يمكن أن يستدعي وحدة ترميز تم إنشاؤها باستخدام التتبع.

.. testsetup::

   # هذه مخفية من الوثائق، ولكنها ضرورية لـ `doctest`
   # لأن وحدة "التفتيش" لا تتوافق مع بيئة التنفيذ
   # لـ `doctest`
   import torch

   original_script = torch.jit.script
   def script_wrapper(obj, *args, **kwargs):
       obj.__module__ = 'FakeMod'
       return original_script(obj, *args, **kwargs)

   torch.jit.script = script_wrapper

   original_trace = torch.jit.trace
   def trace_wrapper(obj, *args, **kwargs):
       obj.__module__ = 'FakeMod'
       return original_trace(obj, *args, **kwargs)

   torch.jit.trace = trace_wrapper

مثال (استدعاء وظيفة متتبعة في النص):

.. testcode::

   import torch

   def foo(x, y):
       return 2 * x + y

   traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

   @torch.jit.script
   def bar(x):
       return traced_foo(x, x)

يمكن للوظائف المتبعة استدعاء وظائف النص. هذا مفيد عندما يتطلب جزء صغير من
النموذج بعض التحكم في التدفق على الرغم من أن معظم النموذج عبارة عن شبكة تغذية أمامية فقط. يتم الحفاظ على التحكم في التدفق داخل
وظيفة النص التي تستدعيها وظيفة متتبعة بشكل صحيح.

مثال (استدعاء وظيفة نص في وظيفة متتبعة):

.. testcode::

   import torch

   @torch.jit.script
   def foo(x, y):
       if x.max() > y.max():
           r = x
       else:
           r = y
       return r


   def bar(x, y, z):
       return foo(x, y) + z

   traced_bar = torch.jit.trace(bar, (torch.rand(3)، torch.rand(3)، torch.rand(3)))

يعمل هذا التكوين أيضًا مع الوحدات النمطية "nn" أيضًا، حيث يمكن استخدامه لإنشاء
وحدة فرعية باستخدام التتبع الذي يمكن استدعاؤه من أساليب الوحدة النمطية للنص.

مثال (استخدام وحدة متتبعة):

.. testcode::
   :skipif: torchvision is None

   import torch
   import torchvision

   class MyScriptModule(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                           .resize_(1, 3, 1, 1))
           self.resnet = torch.jit.trace(torchvision.models.resnet18(),
                                         torch.rand(1, 3, 224, 224))

       def forward(self, input):
           return self.resnet(input - self.means)

   my_script_module = torch.jit.script(MyScriptModule())

لغة TorchScript
-----------------

TorchScript هي مجموعة فرعية ثابتة النوع من Python، لذلك تنطبق العديد من ميزات Python
مباشرة إلى TorchScript. راجع مرجع اللغة الكامل: ref:`language-reference` للحصول على التفاصيل.

.. _الوظائف المدمجة:

الوظائف والوحدات النمطية المدمجة
----------------------

تدعم TorchScript استخدام معظم وظائف PyTorch والعديد من الوظائف المدمجة في Python.
راجع: ref:`builtin-functions` للحصول على مرجع كامل للوظائف المدعومة.

وظائف PyTorch ووحداتها النمطية
~~~~~~~~~~~~~~~~~~~~~~~~

تدعم TorchScript مجموعة فرعية من وظائف الشبكة العصبية ووظائف التنسور التي توفرها PyTorch. معظم الأساليب على Tensor وكذلك الوظائف في
مساحة الاسم "torch"، وجميع الوظائف في "torch.nn.functional" و
معظم الوحدات النمطية من "torch.nn" مدعومة في TorchScript.

راجع: ref:`jit_unsupported` للحصول على قائمة بوظائف PyTorch ووحداتها غير المدعومة.

وظائف Python ووحداتها النمطية
~~~~~~~~~~~~~~~~~~~~~~~~
يتم دعم العديد من `الوظائف المدمجة في Python <https://docs.python.org/3/library/functions.html>`_ في TorchScript.
كما يتم دعم وحدة: any:`math` (راجع: ref:`math-module` للحصول على التفاصيل)، ولكن لا يتم دعم أي وحدات نمطية Python أخرى
(مدمجة أو تابعة لجهات خارجية).

مقارنة مرجع لغة Python
~~~~~~~~~~~~~~~~~~~~

للحصول على قائمة كاملة بميزات Python المدعومة، راجع: ref:`python-language-reference`.

تصحيح الأخطاء

.. _`disable-TorchScript`:

تعطيل JIT لأغراض التصحيح
~~~~~~~~~~~~~~~~~~~~~
.. envvar:: PYTORCH_JIT

يؤدي تعيين متغير البيئة ``PYTORCH_JIT=0`` إلى تعطيل جميع تعليمات النص البرمجي والتعقب. إذا كان هناك خطأ يصعب تصحيحه في أحد نماذج TorchScript، فيمكنك استخدام هذا العلم لإجبار كل شيء على التشغيل باستخدام Python الأصلي. نظرًا لأن TorchScript (النص البرمجي والتعقب) يتم تعطيله باستخدام هذا العلم، فيمكنك استخدام أدوات مثل ``pdb`` لتصحيح أخطاء كود النموذج. على سبيل المثال::

    @torch.jit.script
    def scripted_fn(x : torch.Tensor):
        for i in range(12):
            x = x + x
        return x

    def fn(x):
        x = torch.neg(x)
        import pdb; pdb.set_trace()
        return scripted_fn(x)

    traced_fn = torch.jit.trace(fn, (torch.rand(4, 5),))
    traced_fn(torch.rand(3, 4))

يعمل تصحيح هذا النص البرمجي باستخدام ``pdb`` باستثناء عند استدعاء دالة :func:`@torch.jit.script <torch.jit.script>`. يمكننا تعطيل JIT بشكل عام، بحيث يمكننا استدعاء دالة :func:`@torch.jit.script <torch.jit.script>` كدالة Python عادية وعدم تجميعها. إذا تمت تسمية النص البرمجي أعلاه باسم ``disable_jit_example.py``، فيمكننا استدعاؤه على النحو التالي::

    $ PYTORCH_JIT=0 python disable_jit_example.py

وسنتمكن من الدخول إلى دالة :func:`@torch.jit.script <torch.jit.script>` كدالة Python عادية. ولتعطيل مترجم TorchScript لدالة معينة، راجع :func:`@torch.jit.ignore <torch.jit.ignore>`.

.. _تفقد-الكود:

تفقد الكود
~~~~~~~~~~

يوفر TorchScript أداة تنسيق كود لجميع حالات :class:`ScriptModule`. وتقدم هذه الأداة تفسيرًا لكود طريقة النص البرمجي ككود Python صالح. على سبيل المثال:

.. testcode::

    @torch.jit.script
    def foo(len):
        # type: (int) -> torch.Tensor
        rv = torch.zeros(3, 4)
        for i in range(len):
            if i < 10:
                rv = rv - 1.0
            else:
                rv = rv + 1.0
        return rv

    print(foo.code)

.. testoutput::
    :hide:

    ...

ستكون لدى :class:`ScriptModule` ذات طريقة ``forward`` واحدة سمة ``code``، والتي يمكنك استخدامها لتفقد كود :class:`ScriptModule`. إذا كان لدى :class:`ScriptModule` أكثر من طريقة واحدة، فستحتاج إلى الوصول إلى ``.code`` على الطريقة نفسها وليس على الوحدة النمطية. يمكننا تفقد كود طريقة تسمى ``foo`` على :class:`ScriptModule` عن طريق الوصول إلى ``.foo.code``.

ينتج المثال أعلاه هذا المخرج::

    def foo(len: int) -> Tensor:
        rv = torch.zeros([3, 4], dtype=None, layout=None, device=None, pin_memory=None)
        rv0 = rv
        for i in range(len):
            if torch.lt(i, 10):
                rv1 = torch.sub(rv0, 1., 1)
            else:
                rv1 = torch.add(rv0, 1., 1)
            rv0 = rv1
        return rv0

هذا هو تجميع TorchScript لكود طريقة ``forward``. يمكنك استخدام هذا للتأكد من أن TorchScript (التعقب أو النص البرمجي) قد التقط كود النموذج بشكل صحيح.


.. _تفسير-الرسوم-البيانية:

تفسير الرسوم البيانية
~~~~~~~~~~~~~~~~~~~
لدي TorchScript أيضًا تمثيل على مستوى أقل من أداة تنسيق الكود، في شكل رسوم بيانية IR.

يستخدم TorchScript تمثيل وسيط ثابت الأحادي التعيين (SSA) لتمثيل الحساب. وتتكون التعليمات في هذا التنسيق من مشغلات ATen (الجهة الخلفية لـ C++ في PyTorch) ومشغلات أخرى بدائية، بما في ذلك مشغلات تدفق التحكم للحلقات والعبارات الشرطية. كمثال:

.. testcode::

    @torch.jit.script
    def foo(len):
        # type: (int) -> torch.Tensor
        rv = torch.zeros(3, 4)
        for i in range(len):
            if i < 10:
                rv = rv - 1.0
            else:
                rv = rv + 1.0
        return rv

    print(foo.graph)

.. testoutput::
    :hide:

    ...

يتبع ``graph`` نفس القواعد الموضحة في قسم :ref:`تفقد-الكود` فيما يتعلق بالبحث عن طريقة ``forward``.

ينتج النص البرمجي أعلاه الرسم البياني التالي::

    graph(%len.1 : int):
      %24 : int = prim::Constant[value=1]()
      %17 : bool = prim::Constant[value=1]() # test.py:10:5
      %12 : bool? = prim::Constant()
      %10 : Device? = prim::Constant()
      %6 : int? = prim::Constant()
      %1 : int = prim::Constant[value=3]() # test.py:9:22
      %2 : int = prim::Constant[value=4]() # test.py:9:25
      %20 : int = prim::Constant[value=10]() # test.py:11:16
      %23 : float = prim::Constant[value=1]() # test.py:12:23
      %4 : int[] = prim::ListConstruct(%1, %2)
      %rv.1 : Tensor = aten::zeros(%4, %6, %6, %10, %12) # test.py:9:10
      %rv : Tensor = prim::Loop(%len.1, %17, %rv.1) # test.py:10:5
        block0(%i.1 : int, %rv.14 : Tensor):
          %21 : bool = aten::lt(%i.1, %20) # test.py:11:12
          %rv.13 : Tensor = prim::If(%21) # test.py:11:9
            block0():
              %rv.3 : Tensor = aten::sub(%rv.14, %23, %24) # test.py:12:18
              -> (%rv.3)
            block1():
              %rv.6 : Tensor = aten::add(%rv.14, %23, %24) # test.py:14:18
              -> (%rv.6)
          -> (%17, %rv.13)
      return (%rv)


خذ التعليمات ``%rv.1 : Tensor = aten::zeros(%4, %6, %6, %10, %12) # test.py:9:10`` كمثال.

* ``%rv.1 : Tensor`` يعني أننا نعين الإخراج لقيمة باسم ``rv.1`` (فريدة)، وأن تلك القيمة من نوع ``Tensor`` وأننا لا نعرف شكلها المحدد.
* ``aten::zeros`` هو المشغل (المعادل لـ ``torch.zeros``) وقائمة الإدخال ``(%4, %6، %6، %10، %12)`` تحدد القيم في النطاق التي يجب تمريرها كإدخالات. ويمكن العثور على المخطط للوظائف المدمجة مثل ``aten::zeros`` في `الوظائف المدمجة`_.
* ``# test.py:9:10`` هو الموقع في ملف المصدر الأصلي الذي أنتج هذه التعليمات. في هذه الحالة، هو ملف باسم `test.py`، في السطر 9، وفي الحرف 10.

لاحظ أن المشغلات يمكن أن يكون لها أيضًا كتل مرتبطة بها، وهي ``prim::Loop`` و ``prim::If`` على وجه التحديد. وفي إخراج الرسم البياني، يتم تنسيق هذه المشغلات بحيث تعكس أشكالها المكافئة في كود المصدر لتسهيل التصحيح.

يمكن تفقد الرسوم البيانية كما هو موضح للتأكد من أن الحساب الذي يصفه :class:`ScriptModule` صحيح، سواء بطريقة تلقائية أو يدوية، كما هو موضح أدناه.

المتعقب
~~~~~~


حالات حافة التعقب
^^^^^^^^^^^^^^^^
هناك بعض الحالات الحدية التي يكون فيها تعقب دالة أو وحدة نمطية Python معينة غير ممثِل للكود الأساسي. وقد تشمل هذه الحالات ما يلي:

* تعقب تدفق التحكم الذي يعتمد على الإدخالات (مثل أشكال التنسيق)
* تعقب العمليات في الموقع لعروض التنسيق (مثل الفهرسة على الجانب الأيسر من التعيين)

لاحظ أن هذه الحالات قد تكون في الواقع قابلة للتعقب في المستقبل.


التحقق التلقائي من التعقب
^^^^^^^^^^^^^^^^^^
تتمثل إحدى طرق اكتشاف العديد من الأخطاء في التعقب تلقائيًا في استخدام ``check_inputs`` في واجهة برمجة تطبيقات ``torch.jit.trace()``. حيث يأخذ ``check_inputs`` قائمة من توبلات الإدخالات التي سيتم استخدامها لإعادة تعقب الحساب والتحقق من النتائج. على سبيل المثال::

    def loop_in_traced_fn(x):
        result = x[0]
        for i in range(x.size(0)):
            result = result * x[i]
        return result

    inputs = (torch.rand(3, 4, 5),)
    check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

    traced = torch.jit.trace(loop_in_traced_fn, inputs, check_inputs=check_inputs)

يعطينا معلومات التشخيص التالية::

    ERROR: Graphs differed across invocations!
    Graph diff:

                graph(%x : Tensor) {
                %1 : int = prim::Constant[value=0]()
                %2 : int = prim::Constant[value=0]()
                %result.1 : Tensor = aten::select(%x, %1, %2)
                %4 : int = prim::Constant[value=0]()
                %5 : int = prim::Constant[value=0]()
                %6 : Tensor = aten::select(%x, %4, %5)
                %result.2 : Tensor = aten::mul(%result.1, %6)
                %8 : int = prim::Constant[value=0]()
                %9 : int = prim::Constant[value=1]()
                %10 : Tensor = aten::select(%x, %8, %9)
            -   %result : Tensor = aten::mul(%result.2, %10)
            +   %result.3 : Tensor = aten::mul(%result.2, %10)
            ?          ++
                %12 : int = prim::Constant[value=0]()
                %13 : int = prim::Constant[value=2]()
                %14 : Tensor = aten::select(%x, %12, %13)
            +   %result : Tensor = aten::mul(%result.3, %14)
            +   %16 : int = prim::Constant[value=0]()
            +   %17 : int = prim::Constant[value=3]()
            +   %18 : Tensor = aten::select(%x, %16, %17)
            -   %15 : Tensor = aten::mul(%result, %14)
            ?     ^                                 ^
            +   %19 : Tensor = aten::mul(%result, %18)
            ?     ^                                 ^
            -   return (%15);
            ?             ^
            +   return (%19);
            ?             ^
                }


يشير هذا الرسالة إلى أن الحساب اختلف بين عندما قمنا بتعقبه لأول مرة وعندما قمنا بتعقبه باستخدام ``check_inputs``. وبالفعل، تعتمد الحلقة داخل جسم ``loop_in_traced_fn`` على شكل الإدخال ``x``، وبالتالي عندما نحاول استخدام ``x`` آخر بشكل مختلف، يختلف التعقب.

في هذه الحالة، يمكن التقاط تدفق التحكم المعتمد على البيانات مثل هذا باستخدام :func:`torch.jit.script` بدلاً من ذلك:

.. testcode::

    def fn(x):
        result = x[0]
        for i in range(x.size(0)):
            result = result * x[i]
        return result

    inputs = (torch.rand(3, 4, 5),)
    check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

    scripted_fn = torch.jit.script(fn)
    print(scripted_fn.graph)
    #print(str(scripted_fn.graph).strip())

    for input_tuple in [inputs] + check_inputs:
        torch.testing.assert_close(fn(*input_tuple), scripted_fn(*input_tuple))

.. testoutput::
    :hide:

    ...


والذي ينتج::

    graph(%x : Tensor) {
        %5 : bool = prim::Constant[value=1]()
        %1 : int = prim::Constant[value=0]()
        %result.1 : Tensor = aten::select(%x, %1, %1)
        %4 : int = aten::size(%x, %1)
        %result : Tensor = prim::Loop(%4, %5, %result.1)
        block0(%i : int, %7 : Tensor) {
            %10 : Tensor = aten::select(%x, %1, %i)
            %result.2 : Tensor = aten::mul(%7, %10)
            -> (%5, %result.2)
        }
        return (%result);
    }

تحذيرات المتعقب
^^^^^^^^^^^^^^^
ينتج المتعقب تحذيرات لأنماط متعددة من الأنماط الإشكالية في الحساب المتعقب. كمثال، خذ تعقب دالة تحتوي على تعيين في الموقع على شريحة (عرض) من Tensor:

.. testcode::

    def fill_row_zero(x):
        x[0] = torch.rand(*x.shape[1:2])
        return x

    traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
    print(traced.graph)

.. testoutput::
    :hide:

    ...

ينتج العديد من التحذيرات ورسم بياني يعيد ببساطة الإدخال::

    fill_row_zero.py:4: TracerWarning: There are 2 live references to the data region being modified when tracing in-place operator copy_ (possibly due to an assignment). This might cause the trace to be incorrect, because all other views that also reference this data will not reflect this change in the trace! On the other hand, if all other views use the same memory chunk, but are disjoint (e.g. are outputs of torch.split), this might still be safe.
        x[0] = torch.rand(*x.shape[1:2])
    fill_row_zero.py:6: TracerWarning: Output nr 1. of the traced function does not match the corresponding output of the Python function. Detailed error:
    Not within tolerance rtol=1e-05 atol=1e-05 at input[0, 1] (0.09115803241729736 vs. 0.6782537698745728) and 3 other locations (33.00%)
        traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
    graph(%0 : Float(3, 4)) {
        return (%0);
    }

يمكننا إصلاح هذا عن طريق تعديل الكود لعدم استخدام التحديث في الموقع، ولكن بدلاً من ذلك بناء نتيجة Tensor خارج المكان باستخدام ``torch.cat``:

.. testcode::

    def fill_row_zero(x):
        x = torch.cat((torch.rand(1, *x.shape[1:2]), x[1:2]), dim=0)
        return x

    traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
    print(traced.graph)

.. testoutput::
    :hide:

    ...

الأسئلة الشائعة
------------

س: أود تدريب نموذج على وحدة معالجة الرسوميات (GPU) واستنتاجه على وحدة المعالجة المركزية (CPU). ما هي أفضل الممارسات؟

   أولاً، قم بتحويل نموذجك من وحدة معالجة الرسوميات إلى وحدة المعالجة المركزية ثم احفظه، كما هو موضح أدناه: ::

      cpu_model = gpu_model.cpu()
      sample_input_cpu = sample_input_gpu.cpu()
      traced_cpu = torch.jit.trace(cpu_model, sample_input_cpu)
      torch.jit.save(traced_cpu, "cpu.pt")

      traced_gpu = torch.jit.trace(gpu_model, sample_input_gpu)
      torch.jit.save(traced_gpu, "gpu.pt")

      # ... later, when using the model:

      if use_gpu:
        model = torch.jit.load("gpu.pt")
      else:
        model = torch.jit.load("cpu.pt")

      model(input)

   يوصى بهذا لأنه قد يشهد منشئ التعقب إنشاء tensor على جهاز محدد، لذا فقد يكون لتحويل نموذج محمل بالفعل آثار غير متوقعة. ويضمن تحويل النموذج *قبل* حفظه أن يمتلك منشئ التعقب معلومات الجهاز الصحيحة.

س: كيف يمكنني تخزين السمات على :class: `ScriptModule`؟

    لنفترض أن لدينا نموذجًا مثل:

    .. testcode::

        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = 2

            def forward(self):
                return self.x

        m = torch.jit.script(Model())



    إذا تم إنشاء مثيل لـ ``Model``، فسيؤدي ذلك إلى خطأ في التجميع
    لأن المترجم لا يعرف ``x``. هناك 4 طرق لإعلام المترجم
    بالسمات على :class: `ScriptModule`:

    1. ``nn.Parameter`` - ستعمل القيم الملفوفة في ``nn.Parameter`` كما تفعل
    في ``nn.Module``\s

    2. ``register_buffer`` - ستعمل القيم الملفوفة في ``register_buffer`` كما
    تفعل في ``nn.Module``\s. هذا يعادل سمة (انظر 4) من النوع
    ``Tensor``.

    3. الثوابت - يؤدي وضع علامة على عضو فئة باسم ``Final`` (أو إضافته إلى قائمة تسمى
    ``__constants__`` على مستوى تعريف الفئة) إلى وضع علامة على الأسماء
    المحتواة كثوابت. يتم حفظ الثوابت مباشرة في كود النموذج. راجع
    `الثوابت المدمجة <builtin-constants>` للحصول على التفاصيل.

    4. السمات - يمكن إضافة القيم التي تكون من نوع `مدعوم <supported type>` كسمات قابلة للتغيير. يمكن استنتاج معظم الأنواع ولكن قد يلزم تحديد البعض، راجع
    `سمات الوحدة النمطية <module attributes>` للحصول على التفاصيل.

س: أود تتبع طريقة وحدة نمطية ولكنني أواصل الحصول على هذا الخطأ:

``RuntimeError: لا يمكن إدراج tensor الذي يتطلب تدرجًا كقيمة ثابتة. ضع في اعتبارك جعله معلمة أو إدخالًا، أو فصل تدرجه``

    عادةً ما يعني هذا الخطأ أن الطريقة التي تقوم بتتبعها تستخدم معلمات الوحدة النمطية
    وتقوم بتمرير طريقة وحدة نمطية بدلاً من مثيل الوحدة النمطية (على سبيل المثال، ``my_module_instance.forward`` مقابل ``my_module_instance``).

      - يؤدي استدعاء ``trace`` بطريقة وحدة نمطية إلى التقاط معلمات الوحدة النمطية (التي قد تتطلب تدرجات) ك**ثوابت**.
      - من ناحية أخرى، يؤدي استدعاء ``trace`` بمثيل الوحدة النمطية (على سبيل المثال ``my_module``) إلى إنشاء وحدة نمطية جديدة ونسخ المعلمات إلى الوحدة النمطية الجديدة بشكل صحيح، بحيث يمكنها تراكم التدرجات إذا لزم الأمر.

    لتتبع طريقة محددة في وحدة نمطية، راجع :func: `torch.jit.trace_module <torch.jit.trace_module>`

مشكلات معروفة
-----------

إذا كنت تستخدم ``Sequential`` مع TorchScript، فقد يتم استنتاج إدخالات بعض
من الوحدات النمطية الفرعية ``Sequential`` بشكل خاطئ على أنها
``Tensor``، حتى إذا تم وضع علامة عليها على أنها غير ذلك. الحل الأساسي هو
إنشاء فئة فرعية من ``nn.Sequential`` وإعادة إعلان ``forward``
مع إدخال الكتابة بشكل صحيح.

التذييل
----

الهجرة إلى PyTorch 1.2 واجهة برمجة التطبيقات النصية المتكررة
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
يتضمن هذا القسم تفاصيل التغييرات التي تم إجراؤها على TorchScript في PyTorch 1.2. إذا كنت جديدًا في TorchScript، فيمكنك
تخطي هذا القسم. هناك تغييرين رئيسيين في واجهة برمجة التطبيقات TorchScript مع PyTorch 1.2.

1. :func: `torch.jit.script <torch.jit.script>` ستحاول الآن تجميع الدالات والطرق والفئات بشكل متكرر والتي تواجهها. بمجرد استدعاء ``torch.jit.script``،
   يصبح التجميع "اختياريًا"، بدلاً من "اختياريًا".

2. ``torch.jit.script(nn_module_instance)`` هي الآن الطريقة المفضلة لإنشاء
:class: `ScriptModule`\s، بدلاً من الوراثة من ``torch.jit.ScriptModule``.
تؤدي هذه التغييرات مجتمعة إلى توفير واجهة برمجة تطبيقات أبسط وأسهل للاستخدام لتحويل
وحدات ``nn.Module``\s الخاصة بك إلى :class: `ScriptModule`\s، جاهزة للتحسين والتنفيذ في
بيئة غير Python.

يبدو الاستخدام الجديد كما يلي:

.. testcode::

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    my_model = Model()
    my_scripted_model = torch.jit.script(my_model)


* يتم تجميع طريقة ``forward`` للوحدة النمطية بشكل افتراضي. يتم تجميع الطرق التي يتم استدعاؤها من ``forward``
بتراخٍ بالترتيب الذي يتم استخدامها به في ``forward``، وكذلك أي
طرق ``@torch.jit.export``.
* لتجميع طريقة أخرى غير ``forward`` لا يتم استدعاؤها من ``forward``، أضف ``@torch.jit.export``.
* لإيقاف المترجم من تجميع طريقة، أضف :func: `@torch.jit.ignore <torch.jit.ignore>` أو :func: `@torch.jit.unused <torch.jit.unused>`. ``@ignore`` يترك
* الطريقة كاستدعاء لـ Python، ويستبدل ``@unused`` بها باستثناء. لا يمكن تصدير ``@ignored``؛ يمكن تصدير ``@unused``.
* يمكن استنتاج معظم أنواع السمات، لذا فإن ``torch.jit.Attribute`` غير ضروري. بالنسبة لأنواع الحاويات الفارغة، قم بوضع علامة على أنواعها باستخدام تعليقات PEP 526-style <https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations>`_ class.
* يمكن وضع علامة على الثوابت باستخدام تعليق ``Final`` بدلاً من إضافة اسم العضو إلى ``__constants__``.
* يمكن استخدام تلميحات أنواع Python 3 بدلاً من ``torch.jit.annotate``

ونتيجة لهذه التغييرات، تعتبر العناصر التالية مهملة ولا يجب أن تظهر في التعليمات البرمجية الجديدة:
  * زخرفة ``@torch.jit.script_method``
  * الفئات التي ترث من ``torch.jit.ScriptModule``
  * فئة الغلاف ``torch.jit.Attribute``
  * صفيف ``__constants__``
  * دالة ``torch.jit.annotate``

الوحدات النمطية
^^^^^^^^^^^
.. warning::

    يتغير سلوك تعليق :func: `@torch.jit.ignore <torch.jit.ignore>` في
    PyTorch 1.2. قبل PyTorch 1.2، تم استخدام زخرفة @ignore لجعل دالة
    أو طريقة قابلة للاستدعاء من الكود المصدر. لاستعادة هذه الوظيفة،
    استخدم ``@torch.jit.unused()``. ``@torch.jit.ignore`` يعادل الآن
    ``@torch.jit.ignore(drop=False)``. راجع :func: `@torch.jit.ignore <torch.jit.ignore>`
    و:func: `@torch.jit.unused<torch.jit.unused>` للحصول على التفاصيل.

عند تمريرها إلى دالة :func: `torch.jit.script <torch.jit.script>`، يتم
نسخ بيانات ``torch.nn.Module`` إلى :class: `ScriptModule` ويقوم مترجم TorchScript بتجميع الوحدة النمطية.
يتم تجميع طريقة ``forward`` للوحدة النمطية بشكل افتراضي. يتم تجميع الطرق التي يتم استدعاؤها من ``forward``
بتراخٍ بالترتيب الذي يتم استخدامها به في ``forward``، وكذلك أي
طرق ``@torch.jit.export``.

.. autofunction:: export

الدوال
^^^^
لا تتغير الدوال كثيرًا، فيمكن تزيينها باستخدام :func: `@torch.jit.ignore <torch.jit.ignore>` أو :func: `torch.jit.unused <torch.jit.unused>` إذا لزم الأمر.

.. testcode::

    # نفس السلوك كما كان قبل PyTorch 1.2
    @torch.jit.script
    def some_fn():
        return 2

    # وضع علامة على دالة كمهملة، إذا لم يتم
    # استدعاؤها مطلقًا، فلن يكون لها أي تأثير
    @torch.jit.ignore
    def some_fn2():
        return 2

    # مثل ignore، إذا لم يتم استدعاؤها مطلقًا، فلن يكون لها أي تأثير.
    # إذا تم استدعاؤها في النص البرمجي، فسيتم استبدالها باستثناء.
    @torch.jit.unused
    def some_fn3():
      import pdb; pdb.set_trace()
      return 4

    # لا تفعل شيئًا، هذه الدالة هي بالفعل
    # نقطة الدخول الرئيسية
    @torch.jit.export
    def some_fn4():
        return 2

فئات TorchScript
^^^^^^^^^^^^^^^

.. warning::

    دعم فئات TorchScript تجريبي. حاليًا، فهو مناسب بشكل أفضل
    لأنواع السجلات البسيطة (تخيل ``NamedTuple`` مع أساليب
    مرفقة).

يتم تصدير كل شيء في فئة TorchScript <torchscript-class> التي يحددها المستخدم
بشكل افتراضي، ويمكن تزيين الدوال باستخدام :func: `@torch.jit.ignore
<torch.jit.ignore>` إذا لزم الأمر.

السمات
^^^^^^
يحتاج مترجم TorchScript إلى معرفة أنواع `سمات الوحدة النمطية`. يمكن استنتاج معظم الأنواع
من قيمة العضو. لا يمكن استنتاج قوائم ومقارنات القواميس الفارغة ويجب وضع علامة على أنواعها باستخدام تعليقات `PEP 526-style <https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations>`_ class.
إذا لم يتم استنتاج نوع ولم يتم تحديده بشكل صريح، فلن يتم إضافته كسمة
إلى :class: `ScriptModule` الناتج


الواجهة القديمة:

.. testcode::

    from typing import Dict
    import torch

    class MyModule(torch.jit.ScriptModule):
        def __init__(self):
            super().__init__()
            self.my_dict = torch.jit.Attribute({}, Dict[str, int])
            self.my_int = torch.jit.Attribute(20, int)

    m = MyModule()

الواجهة الجديدة:

.. testcode::

    from typing import Dict

    class MyModule(torch.nn.Module):
        my_dict: Dict[str, int]

        def __init__(self):
            super().__init__()
            # لا يمكن استنتاج هذا النوع ويجب تحديده
            self.my_dict = {}

            # يتم استنتاج نوع السمة هنا على أنه `int`
            self.my_int = 20

        def forward(self):
            pass

    m = torch.jit.script(MyModule())


الثوابت
^^^^^^
يمكن استخدام منشئ النوع ``Final`` لوضع علامة على الأعضاء كـ `ثوابت`. إذا لم يتم وضع علامة على الأعضاء كقيم ثابتة، فسيتم نسخها إلى :class: `ScriptModule` الناتج كسمة. ويفتح استخدام ``Final`` فرصًا للتحسين إذا كان القيمة ثابتة ويعطي أمان نوع إضافي.

الواجهة القديمة:

.. testcode::

    class MyModule(torch.jit.ScriptModule):
        __constants__ = ['my_constant']

        def __init__(self):
            super().__init__()
            self.my_constant = 2

        def forward(self):
            pass
    m = MyModule()

الواجهة الجديدة:

::

    from typing import Final

    class MyModule(torch.nn.Module):

        my_constant: Final[int]

        def __init__(self):
            super().__init__()
            self.my_constant = 2

        def forward(self):
            pass

    m = torch.jit.script(MyModule())

.. _Python 3 type hints:

المتغيرات
^^^^^^^
يُفترض أن تكون الحاويات من النوع ``Tensor`` وغير اختيارية (راجع
`الأنواع الافتراضية` لمزيد من المعلومات). تم سابقًا استخدام ``torch.jit.annotate``
لإخبار مترجم TorchScript بالنوع الذي يجب أن يكون عليه. يتم الآن دعم تلميحات أنواع Python 3.

.. testcode::

    import torch
    from typing import Dict, Optional

    @torch.jit.script
    def make_dict(flag: bool):
        x: Dict[str, int] = {}
        x['hi'] = 2
        b: Optional[int] = None
        if flag:
            b = 2
        return x, b

خلفيات الاندماج
~~~~~~~~~~~~~
تتوفر بعض خلفيات الاندماج لتنفيذ TorchScript. خلفية الاندماج الافتراضية على وحدات المعالجة المركزية هي NNC، والتي يمكنها تنفيذ عمليات الاندماج لكل من وحدات المعالجة المركزية ووحدات معالجة الرسوميات. وخلفية الاندماج الافتراضية على وحدات معالجة الرسوميات هي NVFuser، والتي تدعم مجموعة أوسع من المشغلات وقد أثبتت أن نواة الاندماج الناتجة لها معدل إنتاجية محسن. راجع وثائق NVFuser <https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/codegen/cuda/README.md>`_ لمزيد من التفاصيل حول الاستخدام والتصحيح.


مراجع
~~~~~~
.. toctree::
    :maxdepth: 1

    jit_python_reference
    jit_unsupported

.. لم يتم توثيق هذه الحزمة. تمت إضافتها هنا للتغطية
.. لا يضيف هذا أي شيء إلى الصفحة المقدمة.
.. py:module:: torch.jit.mobile
.. py:module:: torch.jit.annotations
.. py:module:: torch.jit.frontend
.. py:module:: torch.jit.generate_bytecode
.. py:module:: torch.jit.quantized