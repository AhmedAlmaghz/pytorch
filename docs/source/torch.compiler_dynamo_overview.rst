.. -*- rst -*-

نظرة عامة على دينامو
=====================

دينامو هو محرك قواعد بيانات موزعة عالي الأداء، تم تصميمه ليكون قابلاً للتطوير وقادراً على التعامل مع أحجام البيانات الضخمة. إنه يوفر واجهة مستخدم بديهية وسهلة الاستخدام، مما يتيح للمستخدمين إدارة وتلاعب البيانات بكفاءة.

الميزات الرئيسية
--------------

- **قابلية التوسع**: تم بناء دينامو مع وضع قابلية التوسع في الاعتبار، مما يجعله قادراً على التعامل مع أحجام البيانات الكبيرة دون التضحية بالأداء.

- **الأداء العالي**: يتميز دينامو بأداء عالي، مع أوقات استجابة سريعة وعمليات فعالة، مما يضمن كفاءة التعامل مع البيانات.

- **سهولة الاستخدام**: يوفر دينامو واجهة مستخدم بديهية وسهلة الاستخدام، مما يجعل إدارة البيانات والتعامل معها أمرًا مباشرًا وبسيطًا.

- **الموثوقية**: تم تصميم دينامو ليكون موثوقًا، مع ميزات مثل التكرار التلقائي والتعافي من الأخطاء، مما يضمن سلامة البيانات ومتانتها.

- **المرونة**: دينامو مرن وقابل للتكيف، مما يسمح له بالتعامل مع مجموعة متنوعة من حالات استخدام قواعد البيانات.

- **الأمان**: يأتي الأمان كأولوية قصوى في دينامو، مع ميزات تشفير وتصريح قوية لحماية البيانات.

الحالات الاستخدامية
---------------

- **البيانات الضخمة**: يعد دينامو مثاليًا لحالات استخدام البيانات الضخمة، حيث يمكنه التعامل مع مجموعات البيانات الكبيرة والمعقدة بكفاءة.

- **التطبيقات الموزعة**: يمكن استخدام دينامو في التطبيقات الموزعة التي تتطلب وصولًا متزامنًا إلى البيانات من مواقع متعددة.

- **البيانات الديناميكية**: يوفر دينامو حلًا قويًا للتطبيقات التي لديها بيانات ديناميكية ومتغيرة باستمرار.

- **الذكاء الاصطناعي والتعلم الآلي**: يمكن أن يكون دينامو مفيدًا في حالات الذكاء الاصطناعي والتعلم الآلي، حيث يمكنه التعامل مع مجموعات البيانات الكبيرة والمعقدة المطلوبة لهذه التطبيقات.

الخلاصة
--

دينامو هو محرك قواعد بيانات قوي وموثوق وقابل للتطوير، مصمم للتعامل مع متطلبات البيانات الضخمة والتطبيقات الموزعة. مع ميزاته القوية وسهولة استخدامه، فهو خيار رائع لحالات استخدام البيانات الحديثة والمعقدة.
هذا هو النص المترجم إلى اللغة العربية بتنسيق ReStructuredText:

===============

قبل قراءة هذا القسم، يرجى قراءة :ref: `torch.compiler_overview <torch.compiler_overview>`.

TorchDynamo (أو ببساطة Dynamo) هو مترجم Just-In-Time (JIT) على مستوى Python مصمم لجعل برامج PyTorch غير المعدلة أسرع. يرتبط Dynamo بواجهة برمجة التطبيقات لتقييم الإطارات في CPython (`PEP 523 <https://peps.python.org/pep-0523/>`__) لتعديل بايتكود Python ديناميكيًا مباشرة قبل تنفيذه. إنه يعيد كتابة بايتكود Python لاستخراج تسلسلات من عمليات PyTorch في رسم بياني لـ `FX <https://pytorch.org/docs/stable/fx.html>`__ يتم تجميعه بعد ذلك باستخدام backend قابل للتخصيص.

يتم إنشاء رسم FX هذا من خلال تحليل البايتكود، وهو مصمم لمزج تنفيذ Python مع backends المجمعة للحصول على أفضل ما في العالمين - قابلية الاستخدام والأداء.

يجعل Dynamo من السهل تجربة backends مترجم مختلفة لجعل كود PyTorch أسرع باستخدام ديكور من سطر واحد ``torch._dynamo.optimize()`` والذي يتم لفه بشكل مريح بواسطة ``torch.compile()``

يوضح المخطط التالي كيف يعمل PyTorch مع ``torch.compile`` وبدونه:

.. image:: _static/img/dynamo/TorchDynamo.png

`TorchInductor` هو أحد backends المدعومة بواسطة `Dynamo Graph <https://pytorch.org/docs/stable/fx.html>`__ في `Triton <https://github.com/openai/triton>`__ لـ GPUs أو `C++/OpenMP <https://www.openmp.org/>`__ لـ CPUs. لدينا `لوحة أداء التدريب <https://github.com/pytorch/torchdynamo/issues/681#issuecomment-1233828468>`__ التي توفر مقارنة أداء لمختلف backends التدريب. يمكنك قراءة المزيد في `منشور TorchInductor على PyTorch dev-discuss <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__.

للحصول على نظرة عامة متعمقة، اقرأ الأقسام أدناه، وشاهد الفيديو التفصيلي، وتحقق من مواضيع dev-discuss.

   * `فيديو الغوص العميق في Dynamo <https://www.youtube.com/watch?v=egZB5Uxki0I>`__
   * `مواضيع dev-discuss <https://dev-discuss.pytorch.org/search?q=TorchDynamo%20order%3Alatest>`__

داخل Dynamo
~~~~~~~~~~~~~
**المؤلف**: `Jason Ansel <https://github.com/jansel>`_ و `Kaichao You <https://github.com/youkaichao>`_

سيتناول هذا القسم بعضًا من الداخل Dynamo وسيوضح كيف يعمل Dynamo تحت الغطاء.

ما هو الحارس؟
----------------

يعمل Dynamo في الوقت المناسب ويتخصص في الرسوم البيانية بناءً على الخصائص الديناميكية. فيما يلي مثال أساسي حول كيفية استخدام Dynamo. يمكنك تزيين دالة أو طريقة باستخدام ``torchdynamo.optimize`` لتمكين تحسين Dynamo:

.. code-block:: python

   from typing import List
   import torch
   from torch import _dynamo as torchdynamo
   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       print("my_compiler() called with FX graph:")
       gm.graph.print_tabular()
       return gm.forward  # return a python callable

   @torchdynamo.optimize(my_compiler)
   def toy_example(a, b):
       x = a / (torch.abs(a) + 1)
       if b.sum() < 0:
           b = b * -1
       return x * b
   for _ in range(100):
       toy_example(torch.randn(10), torch.randn(10))

على سبيل المثال، تحتوي الرسوم البيانية الأولى أعلاه على الحرس التالي:

::

   GUARDS:
   hasattr(L['a'], '_dynamo_dynamic_indices') == False
   hasattr(L['b'], '_dynamo_dynamic_indices') == False
   utils_device.CURRENT_DEVICE == None
   ___skip_backend_check() or ___current_backend() == ___lookup_backend(140355900538256)
   check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])
   check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])

إذا فشل أي من هذه الحراس، فسيتم إعادة التقاط الرسم البياني وإعادة تجميعه. الحارس المثير للاهتمام هناك هو ``check_tensor``، والذي يتحقق من الخصائص التالية لـ ``torch.Tensor``:

- فئة Python للtensor (التفرع الفرعي للtensor، إلخ)
- dtype
- الجهاز
- requires_grad
- dispatch_key (مع الإضافات/الاستبعادات الخاصة بالخيط المطبق)
- ndim
- الأحجام\*
- الخطوات\*

تسمح طريقة التخصص الكامل لمترجم backend بافتراض رسم بياني ثابت تمامًا. لسوء الحظ، تتطلب معظم backends ذلك. ستؤدي المشغلات التي تعيد الأشكال الديناميكية إلى تشغيل رسم بياني عند عدم تمكين وضع الشكل الديناميكي.

ماذا يفعل Dynamo؟
---------------------

إذا كنت تريد فهم ما يفعله Dynamo بشكل أفضل، فيمكنك تشغيل كودك مع:

::

   TORCH_LOGS="+dynamo,guards,bytecode"

إذا لم تكن على دراية بـ بايتكود Python، فيمكنك إضافة خطاف فك التجميع لفك تجميع البايتكود إلى كود مصدري بشري قابل للقراءة. إحدى الأدوات المتاحة هي `depyf <https://github.com/youkaichao/depyf>`__. إذا لم يكن لديك ``depyf`` مثبتًا بالفعل، فقم بتشغيل ``pip install depyf``. بعد ذلك، أضف الكود التالي لتثبيت خطافات فك التجميع قبل تشغيل أي كود.

.. code-block:: python

   import depyf
   depyf.install()

يؤدي هذا الكود إلى تشغيل عمليات الطباعة المفيدة (ولكن المزعجة).

على سبيل المثال، تتمثل عمليات الطباعة للرسم البياني الأول في ``toy_example`` فيما يلي:

::

   __compiled_fn_0 <eval_with_key>.1
   opcode         name     target                                                  args              kwargs
   -------------  -------  ------------------------------------------------------  ----------------  --------
   placeholder    a        a                                                       ()                {}
   placeholder    b        b                                                       ()                {}
   call_function  abs_1    <built-in method abs of type object at 0x7f9ca082f8a0>  (a,)              {}
   call_function  add      <built-in function add>                                 (abs_1, 1)        {}
   call_function  truediv  <built-in function truediv>                             (a, add)          {}
   call_method    sum_1    sum                                                     (b,)              {}
   call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
   output         output   output                                                  ((truediv, lt),)  {}

   ORIGINAL BYTECODE toy_example example.py line 12
    14           0 LOAD_FAST                0 (a)
                 2 LOAD_GLOBAL              0 (torch)
                 4 LOAD_METHOD              1 (abs)
                 6 LOAD_FAST                0 (a)
                 8 CALL_METHOD              1
                10 LOAD_CONST               1 (1)
                12 BINARY_ADD
                14 BINARY_TRUE_DIVIDE
                16 STORE_FAST               2 (x)

    15          18 LOAD_FAST                1 (b)
                20 LOAD_METHOD              2 (sum)
                22 CALL_METHOD              0
                24 LOAD_CONST               2 (0)
                26 COMPARE_OP               0 (<)
                28 POP_JUMP_IF_FALSE       19 (to 38)

    16          30 LOAD_FAST                1 (b)
                32 LOAD_CONST               3 (-1)
                34 BINARY_MULTIPLY
                36 STORE_FAST               1 (b)

    17     >>   38 LOAD_FAST                2 (x)
                40 LOAD_FAST                1 (b)
                42 BINARY_MULTIPLY
                44 RETURN_VALUE


   MODIFIED BYTECODE toy_example example.py line 12
    12           0 LOAD_GLOBAL              3 (__compiled_fn_0)
                 2 LOAD_FAST                0 (a)
                 4 LOAD_FAST                1 (b)
                 6 CALL_FUNCTION            2
                 8 UNPACK_SEQUENCE          2
                10 STORE_FAST               2 (x)
                12 POP_JUMP_IF_FALSE       12 (to 24)
                14 LOAD_GLOBAL              4 (__resume_at_30_1)
                16 LOAD_FAST                1 (b)
                18 LOAD_FAST                2 (x)
                20 CALL_FUNCTION            2
                22 RETURN_VALUE
           >>   24 LOAD_GLOBAL              5 (__resume_at_38_2)
                26 LOAD_FAST                1 (b)
                28 LOADَََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََََ
للتعرف على الآثار المترتبة على الكود الذي تم تجميعه بواسطة Dynamo، هناك واجهة برمجة تطبيقات (API) ``torch._dynamo.eval_frame._debug_get_cache_entry_list`` والتي تسترد الكود المُجمّع والحراس من كائن ``__code__`` التابع للدالة. يمكن أن تحتوي الدالة المجمّعة على عدة إدخالات ذاكرة التخزين المؤقت، ويتكون كل إدخال ذاكرة تخزين مؤقت من دالة مُنشأة للتحقق من الحراس، وكائن ``types.CodeType`` للاحتفاظ بالكود المراد تنفيذه إذا تم استيفاء شروط الحماية.

.. code-block:: python

   from torch._dynamo.eval_frame import _debug_get_cache_entry_list, innermost_fn
   cache_entries = _debug_get_cache_entry_list(innermost_fn(toy_example))
   cache_entry = cache_entries[0]
   guard, code = cache_entry.check_fn, cache_entry.code
   # يقوم الحارس بأخذ المتغيرات المحلية لإطار الإدخال، ويشير إلى ما إذا كان يجب تشغيل إعادة التجميع.
   import dis
   dis.dis(guard)
   dis.dis(code)

إذا كنت تعرف بايت كود بايثون، فيمكنك فهم الإخراج أعلاه.

بالنسبة لدالة الحارس، لا توجد حاجة لتفحص بايت كود. يمكننا الوصول مباشرة إلى شروط الحماية الخاصة بها:

.. code-block:: python

   for code_part in guard.code_parts:
       print(code_part)

الإخراج هو:

::

   ___guarded_code.valid
   ___check_global_state()
   hasattr(L['a'], '_dynamo_dynamic_indices') == False
   hasattr(L['b'], '_dynamo_dynamic_indices') == False
   utils_device.CURRENT_DEVICE == None
   ___skip_backend_check() or ___current_backend() == ___lookup_backend(140215810860528)
   ___check_tensors(L['a'], L['b'], tensor_check_names=tensor_check_names)

فقط عندما يتم استيفاء جميع الشروط، تقوم دالة الحارس بإرجاع القيمة "صحيح"، ويتم تنفيذ الكود المجمّع.

بالنسبة للكود المُجمّع، لا يمكننا الوصول مباشرة إلى مصدره ولكن يجب علينا إلغاء تجميعه.

.. code-block:: python

   from depyf import decompile
   print(decompile(code))

الإخراج هو:

::

   def toy_example(a, b):
       __temp_1 = __compiled_fn_0(a, b)
       x = __temp_1[0]
       if __temp_1[1]:
           return __resume_at_30_1(b, x)
       return __resume_at_38_2(b, x)

بعض الأسماء التي تمت الإشارة إليها في الكود هي:

- الدوال المُجمّعة، المخزنة في المساحة الاسمية العالمية للوحدة التي تحتوي على الدالة الأصلية ``toy_example``. تتضمن هذه الأسماء مثل ``__compiled_fn_0`` / ``__resume_at_30_1`` / ``__resume_at_38_2``.

- متغيرات الإغلاق المستخدمة للتحقق من الحراس. يمكن الوصول إلى الأسماء من ``guard.__code__.co_freevars``، ويتم تخزين القيم في ``guard.__closure__``. تتضمن هذه الأسماء مثل ``___guarded_code`` / ``___is_grad_enabled`` / ``___are_deterministic_algorithms_enabled`` / ``___is_torch_function_enabled`` / ``utils_device`` / ``___check_tensors`` / ``tensor_check_names``.

- الحُجة ``L`` من دالة ``guard``. هذا عبارة عن قاموس يقوم بمخطط اسم الحجج من ``toy_example`` إلى قيمها. هذا متاح فقط عندما يتم استدعاء الدالة، حيث يتم استخدام واجهة برمجة تطبيقات تقييم الإطار. وباختصار، فإن "L" عبارة عن قاموس بهيكل ``{'a': value_a, 'b': value_b}``. لذلك، يمكنك رؤية الكود الذي يستخدم ``L['a']`` للإشارة إلى المتغير المدخل ``a``.

يتم عرض كسر الرسم البياني في كود "toy_example" المُجمّع، حيث يتعين علينا استخدام مفسّر بايثون لاختيار الرسم البياني التالي للتنفيذ.

لاحظ أننا نمرر دالة "my_compiler" البسيطة كمُجمّل خلفي، وبالتالي فإن كود الرسم البياني الفرعي ``__resume_at_38_2``، و ``__resume_at_30_1``، و ``__compiled_fn_0`` يظل كود بايثون. يمكن أيضًا فحص هذا (يرجى تجاهل اسم الدالة، واستخدام توقيع الدالة وكود جسم الدالة فقط):

.. code-block:: python

   print("source code of __compiled_fn_0:")
   print(innermost_fn(__compiled_fn_0).__self__.code)
   print("=" * 60)
   print("source code of __resume_at_30_1:")
   print(decompile(__resume_at_30_1))
   print("=" * 60)
   print("source code of __resume_at_38_2:")
   print(decompile(__resume_at_38_2))

::

   source code of __compiled_fn_0:

   def forward(self, L_a_ : torch.Tensor, L_b_ : torch.Tensor):
       l_a_ = L_a_
       l_b_ = L_b_
       abs_1 = torch.abs(l_a_)
       add = abs_1 + 1;  abs_Multiplier = None
       truediv = l_a_ / add;  l_a_ = add = None
       sum_1 = l_b_.sum();  l_b_ = None
       lt = sum_1 < 0;  sum_1 = None
       return (truediv, lt)

   # To see more debug info, please use ``graph_module.print_readable()``
   ============================================================
   source code of __resume_at_30_1:
   def <resume in toy_example>(b, x):
       b = b * -1
       return x * b

   ============================================================
   source code of __resume_at_38_2:
   def <resume in toy_example>(b, x):
       return x * b

ومع ذلك، إذا كنا نستخدم خلفيات أخرى مثل "inductor" المدمجة، فإن كود الرسم البياني الفرعي سيكون نواة CUDA مجمعة للوحدة المعالجة المركزية الرسومية (GPU) أو كود C++ لوحدة المعالجة المركزية (CPU).

لتلخيص، فإن الكود المجمّع مكافئ مفاهيميًا للكود أدناه:

.. code-block:: python

   def compiled_example(a, b):
       L = {'a': a, 'b': b}
       for guard, code in get_cache_entries():
           if guard(L):
               return code(a, b)
       recompile_and_add_another_cache_entry()

يوضح المخطط التالي كيف تقوم ``torch.compile`` بتحويل الكود الذي كتبه المستخدم وتحسينه: حيث تقوم أولاً باستخراج الرسوم البيانية للحساب من الدالة التي كتبها المستخدم، ثم تقوم بتجميع هذه الرسوم البيانية إلى دوال مُحسّنة، ثم تقوم بتجميعها في دالة جديدة، وهي مكافئة وظيفيًا للكود الذي كتبه المستخدم ولكنها مُحسّنة لتوفير سرعة حسابية جيدة.

.. image:: _static/img/dynamo/flowchart.jpg

لمعرفة المزيد حول كيفية تنفيذ كل هذا داخليًا، راجع :ref: `torch.compiler_dynamo_deepdive`.