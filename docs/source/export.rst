.. _torch.export:

torch.export
.. warning::
    هذه الميزة هي نموذج أولي قيد التطوير النشط وحدوث تغييرات جذرية في المستقبل.

نظرة عامة
--------

:func:`torch.export.export` يأخذ دالة Python قابلة للاستدعاء بشكل تعسفي ( :class:`torch.nn.Module` ، أو دالة، أو طريقة) وينتج مخططًا مُتتبعًا يمثل حساب Tensor فقط للدالة بطريقة مسبقة التجهيز (Ahead-of-Time)، والتي يمكن تنفيذها بعد ذلك باستخدام مخرجات مختلفة أو تسلسلها.

::

    import torch
    from torch.export import export

    class Mod(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            a = torch.sin(x)
            b = torch.cos(y)
            return a + b

    example_args = (torch.randn(10, 10), torch.randn(10, 10))

    exported_program: torch.export.ExportedProgram = export(
        Mod(), args=example_args
    )
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[10, 10], arg1_1: f32[10, 10]):
                # code: a = torch.sin(x)
                sin: f32[10, 10] = torch.ops.aten.sin.default(arg0_1);

                # code: b = torch.cos(y)
                cos: f32[10, 10] = torch.ops.aten.cos.default(arg1_1);

                # code: return a + b
                add: f32[10, 10] = torch.ops.aten.add.Tensor(sin, cos);
                return (add,)

        Graph signature: ExportGraphSignature(
            parameters=[],
            buffers=[],
            user_inputs=['arg0_1', 'arg1_1'],
            user_outputs=['add'],
            inputs_to_parameters={},
            inputs_to_buffers={},
            buffers_to_mutate={},
            backward_signature=None,
            assertion_dep_token=None,
        )
        Range constraints: {}

تنتج ``torch.export`` تمثيلًا وسيطًا (IR) نظيفًا مع الدعائم التالية. يمكن العثور على المزيد من المواصفات حول IR :ref:`هنا <export.ir_spec>`.

* **الصحة**: من المضمون أن يكون تمثيلًا صحيحًا للبرنامج الأصلي، ويحافظ على نفس اتفاقيات الاستدعاء للبرنامج الأصلي.

* **مُوحّد**: لا توجد دلالية Python داخل المخطط. يتم دمج الوحدات الفرعية من البرامج الأصلية لتشكيل مخطط حسابي مُسطّح بالكامل.

* **خصائص المخطط**: المخطط وظيفي بحت، مما يعني أنه لا يحتوي على عمليات ذات تأثيرات جانبية مثل الطفرات أو الإشارات المرجعية. ولا يغير أي قيم وسيطة أو معلمات أو مخازن مؤقتة.

* **البيانات الوصفية**: يحتوي المخطط على بيانات وصفية تم التقاطها أثناء التتبع، مثل تتبع المكدس من رمز المستخدم.

تحت الغطاء، تستخدم ``torch.export`` أحدث التقنيات التالية:

* **TorchDynamo (torch._dynamo)** هو واجهة برمجة تطبيقات (API) داخلية تستخدم ميزة CPython تسمى Frame Evaluation API لتتبع مخططات PyTorch بأمان. يوفر هذا تحسينًا هائلاً في تجربة التقاط المخطط، مع عدد أقل بكثير من عمليات إعادة الكتابة اللازمة لتتبع رمز PyTorch بالكامل.

* **التفاضل التلقائي AOT**: يوفر مخطط PyTorch الوظيفي ويضمن أن يتم تفكيك المخطط/تخفيضه إلى مجموعة مشغلي ATen.

* **Torch FX (torch.fx)**: هو التمثيل الأساسي للمخطط، مما يسمح بإجراء تحويلات مرنة تعتمد على Python.


أطر العمل الحالية
^^^^^^^^^^^^^^^^^^^

:func:`torch.compile` يستخدم أيضًا نفس المكدس PT2 مثل ``torch.export``، ولكنه يختلف قليلاً:

* **JIT مقابل AOT**: :func:`torch.compile` عبارة عن مترجم JIT في حين أن ``torch.export`` ليس المقصود منه إنتاج برامج مجمعة خارج النشر.

* **التقاط المخطط الجزئي مقابل الكامل**: عندما يواجه :func:`torch.compile` جزءًا غير قابل للتتبع من النموذج، فإنه "يكسر المخطط" ويعود إلى تشغيل البرنامج في وقت تشغيل Python المتلهف. في المقابل، تهدف ``torch.export`` إلى الحصول على تمثيل مخطط كامل لنموذج PyTorch، لذا فسوف يتعطل عند الوصول إلى شيء غير قابل للتتبع. نظرًا لأن ``torch.export`` ينتج مخططًا كاملًا منفصلاً عن أي ميزات أو وقت تشغيل Python، يمكن بعد ذلك حفظ هذا المخطط وتحميله وتشغيله في بيئات ولغات مختلفة.

* **مفاضلة قابلية الاستخدام**: نظرًا لأن :func:`torch.compile` قادر على العودة إلى وقت تشغيل Python عند الوصول إلى شيء غير قابل للتتبع، فهو أكثر مرونة. سيتطلب ``torch.export`` من المستخدمين توفير مزيد من المعلومات أو إعادة كتابة رمزهم لجعله قابلًا للتتبع.

بالمقارنة مع :func:`torch.fx.symbolic_trace`، فإن ``torch.export`` يتتبع باستخدام TorchDynamo الذي يعمل على مستوى بايتكود Python، مما يمنحه القدرة على تتبع البنيات Python التعسفية غير المحدودة بما يدعمه التحميل الزائد لمشغل Python. بالإضافة إلى ذلك، يحتفظ ``torch.export`` بتتبع دقيق لبيانات تعريف Tensor، بحيث لا تفشل التتبعيات الشرطية للأشياء مثل أشكال Tensor. بشكل عام، من المتوقع أن يعمل ``torch.export`` على المزيد من برامج المستخدم، وينتج مخططات منخفضة المستوى (على مستوى ``torch.ops.aten``). لاحظ أنه لا يزال بإمكان المستخدمين استخدام :func:`torch.fx.symbolic_trace` كخطوة ما قبل المعالجة قبل ``torch.export``.

بالمقارنة مع :func:`torch.jit.script`، فإن ``torch.export`` لا يلتقط تدفق التحكم في Python أو البنى البيانات، ولكنه يدعم المزيد من ميزات لغة Python أكثر من TorchScript (حيث من الأسهل الحصول على تغطية شاملة لبايتكودات Python). المخططات الناتجة أبسط ولها تدفق تحكم خطي مستقيم (باستثناء مشغلي تدفق التحكم الصريح).

بالمقارنة مع :func:`torch.jit.trace`، فإن ``torch.export`` سليم: فهو قادر على تتبع الرمز الذي يؤدي حسابات صحيحة على الأحجام ويسجل جميع الشروط الجانبية اللازمة لإثبات أن تتبعًا معينًا صالحًا لمدخلات أخرى.


تصدير نموذج PyTorch
مثال
^^^^^^^^^^

نقطة الدخول الرئيسية هي من خلال :func:`torch.export.export`، والتي تأخذ دالة قابلة للاستدعاء (:class:`torch.nn.Module`، أو دالة، أو طريقة) ومدخلات نموذجية، وتلتقط مخطط الحساب في :class:`torch.export.ExportedProgram`. مثال:

::

    import torch
    from torch.export import export

    # وحدة نمطية بسيطة للتوضيح
    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            )
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

        def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
            a = self.conv(x)
            a.add_(constant)
            return self.maxpool(self.relu(a))

    example_args = (torch.randn(1, 3, 256, 256),)
    example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

    exported_program: torch.export.ExportedProgram = export(
        M(), args=example_args, kwargs=example_kwargs
    )
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[16, 3, 3, 3], arg1_1: f32[16], arg2_1: f32[1, 3, 256, 256], arg3_1: f32[1, 16, 256, 256]):

                # code: a = self.conv(x)
                convolution: f32[1, 16, 256, 256] = torch.ops.aten.convolution.default(
                    arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
                );

                # code: a.add_(constant)
                add: f32[1, 16, 256, 256] = torch.ops.aten.add.Tensor(convolution, arg3_1);

                # code: return self.maxpool(self.relu(a))
                relu: f32[1, 16, 256, 256] = torch.ops.aten.relu.default(add);
                max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(
                    relu, [3, 3], [3, 3]
                );
                getitem: f32[1, 16, 85, 85] = max_pool2d_with_indices[0];
                return (getitem,)

        Graph signature: ExportGraphSignature(
            parameters=['L__self___conv.weight', 'L__self___conv.bias'],
            buffers=[],
            user_inputs=['arg2_1', 'arg3_1'],
            user_outputs=['getitem'],
            inputs_to_parameters={
                'arg0_1': 'L__self___conv.weight',
                'arg1_1': 'L__self___conv.bias',
            },
            inputs_to_buffers={},
            buffers_to_mutate={},
            backward_signature=None,
            assertion_dep_token=None,
        )
        Range constraints: {}

بعد فحص ``ExportedProgram``، يمكننا ملاحظة ما يلي:

* يحتوي :class:`torch.fx.Graph` على مخطط الحساب للبرنامج الأصلي، إلى جانب سجلات الكود الأصلي للتصحيح السهل.

* يحتوي المخطط على مشغلات ``torch.ops.aten`` فقط الموجودة `هنا <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>`__
  والمشغلات المخصصة، وهو يعمل بالكامل، بدون أي مشغلات ذات مواضع محددة
  مثل ``torch.add_``.

* يتم رفع المعلمات (الوزن والانحياز إلى الدالة) كمدخلات للمخطط، مما يؤدي إلى عدم وجود عقد ``get_attr`` في المخطط، والتي كانت موجودة سابقًا
  في نتيجة :func:`torch.fx.symbolic_trace`.

* يقوم :class:`torch.export.ExportGraphSignature` بتصميم توقيع الدخل والخرج، إلى جانب تحديد أي المدخلات هي معلمات.

* يتم ملاحظة الشكل الناتج ونوع البيانات الناتج عن كل عقدة في المخطط. على سبيل المثال، ستنتج عقدة ``convolution`` نوع بيانات
  ``torch.float32`` وشكل (1، 16، 256، 256).


.. _Non-Strict Export:

تصدير غير صارم
^^^^^^^^^^^^^^

في PyTorch 2.3، قمنا بتقديم وضع جديد للتعقب يسمى **الوضع غير الصارم**.
لا يزال يمر بمرحلة التصلب، لذا إذا واجهتك أي مشكلات، يرجى إرسالها إلى Github مع علامة "oncall: export".

في *الوضع غير الصارم*، نقوم بتعقب البرنامج باستخدام مفسر Python.
سينفذ كودك بالضبط كما هو في الوضع الحريص؛ والفرق الوحيد هو
أن جميع كائنات Tensor سيتم استبدالها بواسطة ProxyTensors، والتي ستسجل جميع
عملياتها في مخطط.

في *الوضع الصارم*، وهو الوضع الافتراضي حاليًا، نقوم أولاً بتعقب البرنامج باستخدام TorchDynamo، وهو محرك تحليل بايتكود. لا يقوم TorchDynamo بتنفيذ كود Python الخاص بك بالفعل. بدلاً من ذلك، فإنه يحللها رمزيًا ويبني مخططًا بناءً على النتائج. يسمح هذا التحليل لـ torch.export بتقديم ضمانات أقوى بشأن السلامة، ولكن ليس كل كود Python مدعوم.

مثال على حالة قد يرغب فيها المرء في استخدام الوضع غير الصارم هو إذا واجهت ميزة غير مدعومة من TorchDynamo قد لا يتم حلها بسهولة، وتعرف أن كود Python غير مطلوب بالضبط للحساب. على سبيل المثال:

::

    import contextlib
    import torch

    class ContextManager():
        def __init__(self):
            self.count = 0
        def __enter__(self):
            self.count += 1
        def __exit__(self, exc_type, exc_value, traceback):
            self.count -= 1

    class M(torch.nn.Module):
        def forward(self, x):
            with ContextManager():
                return x.sin() + x.cos()

    export(M(), (torch.ones(3, 3),), strict=False)  # تتبع غير صارم ناجح
    export(M(), (torch.ones(3, 3),))  # يفشل الوضع الصارم مع torch._dynamo.exc.Unsupported: ContextManager

في هذا المثال، ينجح أول استدعاء باستخدام الوضع غير الصارم (من خلال
علامة ``strict=False``) في حين أن الاستدعاء الثاني باستخدام الوضع الصارم (الافتراضي) ينتج عنه فشل، حيث لا يمكن لـ TorchDynamo دعم
مديري السياق. أحد الخيارات هو إعادة كتابة الكود (راجع :ref:`Limitations of torch.export <Limitations of
torch.export>`)، ولكن بالنظر إلى أن مدير السياق لا يؤثر على حسابات tensor في النموذج، يمكننا استخدام نتيجة الوضع غير الصارم.


التعبير عن الديناميكية
^^^^^^^^^^^^^^

بشكل افتراضي، يفترض ``torch.export`` أن جميع الأشكال المدخلة **ثابتة**، ويخصص البرنامج المصدر لتلك الأبعاد. ومع ذلك،
يمكن أن تكون بعض الأبعاد، مثل حجم الدفعة، ديناميكية وتختلف من تشغيل إلى آخر. يجب تحديد هذه الأبعاد باستخدام
:func:`torch.export.Dim` API لإنشائها ومن خلال تمريرها إلى
:func:`torch.export.export` من خلال حجة ``dynamic_shapes``. مثال:

::

    import torch
    from torch.export import Dim, export

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(
                torch.nn.Linear(64, 32), torch.nn.ReLU()
            )
            self.branch2 = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU()
            )
            self.buffer = torch.ones(32)

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1 + self.buffer, out2)

    example_args = (torch.randn(32, 64), torch.randn(32, 128))

    # إنشاء حجم دفعة ديناميكي
    batch = Dim("batch")
    # تحديد أن البعد الأول لكل إدخال هو حجم الدفعة هذا
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    exported_program: torch.export.ExportedProgram = export(
        M(), args=example_args, dynamic_shapes=dynamic_shapes
    )
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[32, 64], arg1_1: f32[32], arg2_1: f32[64, 128], arg3_1: f32[64], arg4_1: f32[32], arg5_1: f32[s0, 64], arg6_1: f32[s0, 128]):

                # code: out1 = self.branch1(x1)
                permute: f32[64, 32] = torch.ops.aten.permute.default(arg0_1, [1, 0]);
                addmm: f32[s0, 32] = torch.ops.aten.addmm.default(arg1_1, arg5_1, permute);
                relu: f32[s0, 32] = torch.ops.aten.relu.default(addmm);

                # code: out2 = self.branch2(x2)
                permute_1: f32[128, 64] = torch.ops.aten.permute.default(arg2_1, [1, 0]);
                addmm_1: f32[s0, 64] = torch.ops.aten.addmm.default(arg3_1, arg6_1, permute_1);
                relu_1: f32[s0, 64] = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None

                # code: return (out1 + self.buffer, out2)
                add: f32[s0, 32] = torch.ops.aten.add.Tensor(relu, arg4_1);
                return (add, relu_1)

        Graph signature: ExportGraphSignature(
            parameters=[
                'branch1.0.weight',
                'branch1.0.bias',
                'branch2.0.weight',
                'branch2.0.bias',
            ],
            buffers=['L__self___buffer'],
            user_inputs=['arg5_1', 'arg6_1'],
            user_outputs=['add', 'relu_1'],
            inputs_to_parameters={
                'arg0_1': 'branch1.0.weight',
                'arg1_1': 'branch1.0.bias',
                'arg2_1': 'branch2.0.weight',
                'arg3_1': 'branch2.0.bias',
            },
            inputs_to_buffers={'arg4_1': 'L__self___buffer'},
            buffers_to_mutate={},
            backward_signature=None,
            assertion_dep_token=None,
        )
        Range constraints: {s0: RangeConstraint(min_val=2, max_val=9223372036854775806)}

بعض الأشياء الإضافية التي يجب ملاحظتها:

* من خلال :func:`torch.export.Dim` API وحجة ``dynamic_shapes``، حددنا البعد الأول لكل إدخال ليكون ديناميكيًا. بالنظر إلى المدخلات ``arg5_1`` و
  ``arg6_1``، لديهما شكل رمزي من (s0، 64) و (s0، 128)، بدلاً من
  أشكال tensor (32، 64) و (32، 128) التي مررناها كمدخلات نموذجية.
  ``s0`` هو رمز يمثل أن هذا البعد يمكن أن يكون
  نطاق من القيم.

* ``exported_program.range_constraints`` يصف نطاقات كل رمز
  يظهر في المخطط. في هذه الحالة، نرى أن ``s0`` له النطاق
  [2، inf]. لأسباب فنية يصعب شرحها هنا، يفترض أنها ليست 0 أو 1. هذا ليس خطأ، ولا يعني بالضرورة
  أن البرنامج المصدر لن يعمل للأبعاد 0 أو 1. راجع
  `The 0/1 Specialization Problem <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk>`_
  لمناقشة متعمقة حول هذا الموضوع.


يمكننا أيضًا تحديد علاقات أكثر تعبيرًا بين أشكال الإدخال، مثل
حيث قد يختلف زوج من الأشكال بواحد، أو قد يكون شكل ضعف
آخر، أو شكل زوجي. مثال:

::

    class M(torch.nn.Module):
        def forward(self, x, y):
            return x + y[1:]

    x, y = torch.randn(5), torch.randn(6)
    dimx = torch.export.Dim("dimx", min=3, max=6)
    dimy = dimx + 1

    exported_program = torch.export.export(
        M(), (x, y), dynamic_shapes={0: dimx, 1: dimy},
    )
    print(exported_program)

.. code-block::

    ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: "f32[s0]", arg1_1: "f32[s1]"):
            # code: return x + y[1:]
            slice_1: "f32[s0]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, 9223372036854775807);  arg1_1 = None
            add: "f32[s0]" = torch.ops.aten.add.Tensor(arg0_1, slice_1);  arg0_1 = slice_1 = None
            return (add,)

    Graph signature: ExportGraphSignature(
        input_specs=[
            InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg0_1'), target=None, persistent=None),
            InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg1_1'), target=None, persistent=None)
        ],
        output_specs=[
            OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='add'), target=None)]
    )
    Range constraints: {s0: ValueRanges(lower=3, upper=6, is_bool=False), s1: ValueRanges(lower=4, upper=7, is_bool=False)}

بعض الأمور التي يجب ملاحظتها:

* من خلال تحديد ``{0: dimx}`` للإدخال الأول، نرى أن الشكل الناتج للإدخال الأول أصبح الآن ديناميكيًا، وهو ``[s0]``. والآن، من خلال تحديد ``{1: dimy}`` للإدخال الثاني، نرى أن الشكل الناتج للإدخال الثاني ديناميكي أيضًا. ومع ذلك، لأننا عبرنا عن ``dimy = dimx + 1``، بدلاً من احتواء شكل ``arg1_1`` على رمز جديد، نرى أنه يتم تمثيله الآن باستخدام نفس الرمز المستخدم في ``arg0_1``، وهو ``s0``. يمكننا أن نرى أن العلاقة بين ``dimy = dimx + 1`` يتم عرضها من خلال ``s0 + 1``.

* عند النظر في قيود النطاق، نرى أن "s0" له النطاق [3، 6]، والذي تم تحديده مبدئيًا، ويمكننا أن نرى أن "s0 + 1" له النطاق المحسوب [4، 7].

.. _Serialization:

التخزين التسلسلي
^^^^^^^^^^^^

لحفظ ``ExportedProgram``، يمكن للمستخدمين استخدام واجهات برمجة التطبيقات :func:`torch.export.save` و :func:`torch.export.load`. ويتمثل التقليد في حفظ ``ExportedProgram`` باستخدام ملحق ملف ``.pt2``.

مثال:

::

    import torch
    import io

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + 10

    exported_program = torch.export.export(MyModule(), torch.randn(5))

    torch.export.save(exported_program, 'exported_program.pt2')
    saved_exported_program = torch.export.load('exported_program.pt2')

.. _Specializations:

التخصصات
^^^^^^^^^

مفهوم أساسي في فهم سلوك ``torch.export`` هو الفرق بين القيم *الثابتة* و*المتغيرة*.

القيمة *المتغيرة* هي قيمة يمكن أن تتغير من تشغيل إلى آخر. وتتصرف هذه القيم مثل الحجج العادية لدالة Python - يمكنك تمرير قيم مختلفة لحجة وتتوقع من دالتك أن تفعل الشيء الصحيح. وتعتبر بيانات *المصفوفة* قيمة متغيرة.

القيمة *الثابتة* هي قيمة ثابتة في وقت التصدير ولا يمكن أن تتغير بين عمليات تنفيذ البرنامج المصدر. عندما تتم مواجهة القيمة أثناء التعقب، سيعاملها المصدر على أنها ثابتة ويتم ترميزها في الرسم البياني.

عندما يتم تنفيذ عملية (على سبيل المثال، ``x + y``) وجميع المدخلات ثابتة، فستتم ترميز نتيجة العملية مباشرة في الرسم البياني، ولن تظهر العملية (أي سيتم طيها ثابتًا).

عندما يتم ترميز قيمة في الرسم البياني، نقول إن الرسم البياني تم *تخصصه* لتلك القيمة.

القيم التالية ثابتة:

أشكال المصفوفة المدخلة
~~~~~~~~~~~~~~~~~~~~~~

بشكل افتراضي، سيقوم ``torch.export`` بتعقب البرنامج المتخصص في أشكال المصفوفات المدخلة، ما لم يتم تحديد البعد على أنه ديناميكي عبر وسيط ``dynamic_shapes`` إلى ``torch.export``. وهذا يعني أنه إذا كان هناك تحكم في التدفق المعتمد على الشكل، فسوف يتخصص ``torch.export`` في الفرع الذي يتم اتخاذه مع إدخالات العينة المقدمة. على سبيل المثال:

::

    import torch
    from torch.export import export

    class Mod(torch.nn.Module):
        def forward(self, x):
            if x.shape[0] > 5:
                return x + 1
            else:
                return x - 1

    example_inputs = (torch.rand(10, 2),)
    exported_program = export(Mod(), example_inputs)
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[10, 2]):
                add: f32[10, 2] = torch.ops.aten.add.Tensor(arg0_1, 1);
                return (add,)

لا يظهر الشرط (``x.shape[0] > 5``) في ``ExportedProgram`` لأن إدخالات المثال لها الشكل الثابت (10، 2). نظرًا لأن ``torch.export`` يتخصص في الأشكال الثابتة للإدخالات، فلن يتم الوصول إلى فرع "else" (``x - 1``) مطلقًا. للحفاظ على سلوك التفرع الديناميكي المعتمد على شكل مصفوفة في الرسم البياني الذي يتم تعقبه، يجب استخدام :func:`torch.export.Dim` لتحديد بُعد مصفوفة الإدخال (``x.shape[0]``) على أنه ديناميكي، وسيتعين إعادة كتابة التعليمات البرمجية المصدر: ref: `<Data / Shape-Dependent Control Flow>`.

لاحظ أن المصفوفات التي تعد جزءًا من حالة الوحدة (مثل المعلمات والمخازن المؤقتة) لها دائمًا أشكال ثابتة.

البدائيات بايثون
~~~~~~~~~~~~

يتخصص ``torch.export`` أيضًا في بدائيات بايثون، مثل ``int`` و ``float`` و ``bool`` و ``str``. ومع ذلك، فإن لها متغيرات ديناميكية مثل ``SymInt`` و ``SymFloat`` و ``SymBool``.

على سبيل المثال:

::

    import torch
    from torch.export import export

    class Mod(torch.nn.Module):
        def forward(self, x: torch.Tensor, const: int, times: int):
            for i in range(times):
                x = x + const
            return x

    example_inputs = (torch.rand(2, 2), 1, 3)
    exported_program = export(Mod(), example_inputs)
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[2, 2], arg1_1, arg2_1):
                add: f32[2, 2] = torch.ops.aten.add.Tensor(arg0_1, 1);
                add_1: f32[2, 2] = torch.ops.aten.add.Tensor(add, 1);
                add_2: f32[2, 2] = torch.ops.aten.add.Tensor(add_1, 1);
                return (add_2,)

نظرًا لأن الأعداد الصحيحة متخصصة، يتم حساب عمليات ``torch.ops.aten.add.Tensor`` جميعها باستخدام الثابت 1، بدلاً من ``arg1_1``. إذا مرر المستخدم قيمة مختلفة لـ ``arg1_1`` في وقت التشغيل، مثل 2، بدلاً من 1 المستخدمة في وقت التصدير، فسيؤدي ذلك إلى حدوث خطأ. بالإضافة إلى ذلك، يتم "إدراج" متكرر "times" المستخدم في حلقة "for" في الرسم البياني من خلال 3 مكالمات متكررة لـ ``torch.ops.aten.add.Tensor``، ولا يتم أبدًا استخدام الإدخال "arg2_1".

الحاويات بايثون
~~~~~~~~~~~

تعتبر حاويات بايثون (``List`` و ``Dict`` و ``NamedTuple``، إلخ) ذات بنية ثابتة.

.. _Limitations of torch.export:

قيود التصدير الشعلة

انقطاعات الرسم البياني
^^^^^^^^^^^^^^^^^^

نظرًا لأن ``torch.export`` هي عملية لمرة واحدة لالتقاط رسم بياني للحساب من برنامج PyTorch، فقد ينتهي بها الأمر في النهاية إلى أجزاء من البرامج التي يتعذر تتبعها، حيث أنه من المستحيل تقريبًا دعم تتبع جميع ميزات PyTorch وPython. في حالة ``torch.compile``، ستتسبب العملية غير المدعومة في "انقطاع الرسم البياني" وسيتم تشغيل العملية غير المدعومة باستخدام التقييم الافتراضي لـ Python. على النقيض من ذلك، سيتطلب ``torch.export`` من المستخدمين توفير معلومات إضافية أو إعادة كتابة أجزاء من شفرتهم لجعلها قابلة للتتبع. نظرًا لأن التتبع يعتمد على TorchDynamo، والذي يقوم بالتقييم على مستوى بايت كود Python، ستكون هناك عمليات إعادة كتابة أقل بكثير مقارنة بأطر التتبع السابقة.

عند مواجهة انقطاع في الرسم البياني، يعد :ref:``<torch.export_db>`` ExportDB موردًا رائعًا لمعرفة أنواع البرامج المدعومة وغير المدعومة، إلى جانب طرق إعادة كتابة البرامج لجعلها قابلة للتتبع.

يتمثل أحد الخيارات للتعامل مع هذه الانقطاعات في الرسم البياني باستخدام :ref:``<Non-Strict Export>`` التصدير غير الصارم.

.. _Data/Shape-Dependent Control Flow:

تدفق التحكم المعتمد على البيانات/الشكل
^^^^^^^^^^^^^^^^^^^^^^^^^^

يمكن أيضًا مواجهة انقطاعات الرسم البياني في تدفق التحكم المعتمد على البيانات (``if x.shape[0] > 2``) عندما لا يتم تخصص الأشكال، حيث لا يمكن لمترجم التتبع التعامل معها دون توليد رمز لعدد متفجر من المسارات. في مثل هذه الحالات، سيتعين على المستخدمين إعادة كتابة شفرتهم باستخدام مشغلي تدفق التحكم الخاصين. حاليًا، ندعم :ref:``<cond>`` torch.cond للتعبير عن تدفق التحكم الشبيه بـ if-else (قادم قريبًا!).

النوى الوهمية/الافتراضية/المجردة المفقودة للمشغلين
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

عند التتبع، تكون نواة FakeTensor (المعروفة أيضًا باسم النواة المجردة، أو التنفيذ المجرد) مطلوبة لجميع المشغلين. ويستخدم هذا للتفكير في أشكال الإدخال/الإخراج لهذا المشغل.

يرجى الاطلاع على :func:``torch.library.register_fake`` لمزيد من التفاصيل.

في الحالة المؤسفة التي يستخدم فيها نموذجك مشغل ATen الذي لا يحتوي على تنفيذ نواة FakeTensor بعد، يرجى تقديم مشكلة.

اقرأ المزيد
---------

.. toctree::
   :caption: روابط إضافية لمستخدمي التصدير
   :maxdepth: 1

   export.ir_spec
   torch.compiler_transformations
   torch.compiler_ir
   generated/exportdb/index
   cond

.. toctree::
   :caption: الغوص العميق لمطوري PyTorch
   :maxdepth: 1

   torch.compiler_dynamo_overview
   torch.compiler_dynamo_deepdive
   torch.compiler_dynamic_shapes
   torch.compiler_fake_tensor

مرجع API
-------------

.. automodule:: torch.export
.. autofunction:: export
.. autofunction:: save
.. autofunction:: load
.. autofunction:: register_dataclass
.. autoclass:: torch.export.dynamic_shapes.DIM
.. autofunction:: torch.export.dynamic_shapes.Dim
.. autofunction:: dims
.. autoclass:: torch.export.dynamic_shapes.ShapesCollection

    .. automethod:: dynamic_shapes

.. autofunction:: torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes
.. autoclass:: Constraint
.. autoclass:: ExportedProgram

    .. automethod:: module
    .. automethod:: buffers
    .. automethod:: named_buffers
    .. automethod:: parameters
    .. automethod:: named_parameters
    .. automethod:: run_decompositions

.. autoclass:: ExportBackwardSignature
.. autoclass:: ExportGraphSignature
.. autoclass:: ModuleCallSignature
.. autoclass:: ModuleCallEntry


.. automodule:: torch.export.exported_program
.. automodule:: torch.export.graph_signature
.. autoclass:: InputKind
.. autoclass:: InputSpec
.. autoclass:: OutputKind
.. autoclass:: OutputSpec
.. autoclass:: ExportGraphSignature

    .. automethod:: replace_all_uses
    .. automethod:: get_replace_hook

.. autoclass:: torch.export.graph_signature.CustomObjArgument

.. py:module:: torch.export.dynamic_shapes

.. automodule:: torch.export.unflatten
    :members:

.. automodule:: torch.export.custom_obj

.. automodule:: torch.export.experimental
.. automodule:: torch.export.passes
.. autofunction:: torch.export.passes.move_to_device_pass