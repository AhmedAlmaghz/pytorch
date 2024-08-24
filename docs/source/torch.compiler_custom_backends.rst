الواجهات الخلفية المخصصة
========================

نظرة عامة
--------

تتيح ``torch.compile`` طريقة مباشرة للمستخدمين لتعريف الواجهات الخلفية المخصصة.

لدى الوظيفة الخلفية عقد ``(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable``.

يمكن استدعاء الوظائف الخلفية بواسطة TorchDynamo، وهو مكون تتبع الرسوم البيانية في ``torch.compile``، بعد تتبع رسم بياني لـ FX، ومن المتوقع أن تعيد وظيفة مجمعة مكافئة للرسم البياني FX الذي تم تتبعه. يجب أن يكون للدالة القابلة للاستدعاء التي تم إرجاعها نفس عقد دالة ``forward`` لـ ``torch.fx.GraphModule`` الأصلي الذي تم تمريره إلى الواجهة الخلفية: ``(*args: torch.Tensor) -> List[torch.Tensor]``.

لكي يستدعي TorchDynamo الواجهة الخلفية الخاصة بك، قم بتمرير وظيفة الواجهة الخلفية الخاصة بك كوسيط اسمه ``backend`` في ``torch.compile``. على سبيل المثال،

.. code-block:: python

    import torch

    def my_custom_backend(gm, example_inputs):
        return gm.forward

    def f(...):
        ...

    f_opt = torch.compile(f, backend=my_custom_backend)

    @torch.compile(backend=my_custom_backend)
    def g(...):
        ...

راجع أدناه للحصول على مزيد من الأمثلة.

تسجيل الواجهات الخلفية المخصصة
---------------------------

يمكنك تسجيل الواجهة الخلفية الخاصة بك باستخدام الديكور ``register_backend``، على سبيل المثال،

.. code-block:: python

    from torch._dynamo import register_backend

    @register_backend
    def my_compiler(gm, example_inputs):
        ...

بالإضافة إلى الديكور ``register_backend``، إذا كانت الواجهة الخلفية الخاصة بك في حزمة Python أخرى، فيمكنك أيضًا تسجيل الواجهة الخلفية الخاصة بك من خلال نقاط الدخول لحزمة Python، والتي توفر طريقة لحزمة لتسجيل إضافة لآخر.

.. hint::

    يمكنك معرفة المزيد حول ``entry_points`` في
    `وثائق تغليف Python <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`__.

لتسجيل الواجهة الخلفية الخاصة بك من خلال ``entry_points``، يمكنك إضافة وظيفة الواجهة الخلفية الخاصة بك إلى مجموعة نقاط الدخول ``torch_dynamo_backends`` في ملف ``setup.py`` لحزمتك مثل:

.. code-block:: python

    ...
    setup(
        ...
        'torch_dynamo_backends': [
            'my_compiler = your_module.submodule:my_compiler',
        ]
        ...
    )

يرجى استبدال ``my_compiler`` قبل ``=`` باسم الواجهة الخلفية الخاصة بك واستبدال الجزء بعد ``=`` باسم الوحدة النمطية ووظيفة وظيفة الواجهة الخلفية الخاصة بك. سيتم إضافة نقطة الدخول إلى بيئة Python الخاصة بك بعد تثبيت الحزمة.

عند استدعاء ``torch.compile(model, backend="my_compiler")``، سيبحث PyTorch أولاً عن الواجهة الخلفية التي تم تسميتها ``my_compiler`` والتي تم تسجيلها باستخدام ``register_backend``. إذا لم يتم العثور عليه، فسيستمر في البحث في جميع الواجهات الخلفية المسجلة عبر ``entry_points``.

يخدم التسجيل غرضين:

* يمكنك تمرير سلسلة تحتوي على اسم وظيفة الواجهة الخلفية الخاصة بك إلى ``torch.compile`` بدلاً من الوظيفة نفسها، على سبيل المثال، ``torch.compile(model، backend="my_compiler")``.
* مطلوب للاستخدام مع `minifier <https://pytorch.org/docs/main/torch.compiler_troubleshooting.html>`__. يجب أن يدعو أي كود تم إنشاؤه بواسطة minifier كودك الذي يسجل وظيفة الواجهة الخلفية الخاصة بك، عادةً من خلال عبارة ``import``.

الواجهات الخلفية المخصصة بعد AOTAutograd
---------------------------------

من الممكن تعريف الواجهات الخلفية المخصصة التي يستدعيها AOTAutograd بدلاً من TorchDynamo.
هذا مفيد لسببين رئيسيين:

* يمكن للمستخدمين تعريف الواجهات الخلفية التي تدعم تدريب النموذج، حيث يمكن لـ AOTAutograd إنشاء الرسم البياني الخلفي للتجميع.
* ينتج AOTAutograd رسومًا بيانية FX تتكون من `عمليات Aten الأساسية <https://pytorch.org/docs/main/torch.compiler_ir.html#core-aten-ir>`__. ونتيجة لذلك، تحتاج الواجهات الخلفية المخصصة فقط إلى دعم مجموعة التعليمات الأساسية Aten، والتي تعد مجموعة تعليمات أصغر بكثير من مجموعة تعليمات torch/Aten بأكملها.

قم بلف وظيفة الواجهة الخلفية الخاصة بك باستخدام ``torch._dynamo.backends.common.aot_autograd`` واستخدم ``torch.compile`` مع وسيط اسمه ``backend`` كما هو موضح سابقًا. يجب أن يكون للوظائف الخلفية الملفوفة بواسطة ``aot_autograd`` نفس العقد كما هو الحال من قبل.

يتم تمرير وظائف الواجهة الخلفية إلى ``aot_autograd`` من خلال وسيطات ``fw_compiler`` (مُجمِّع التقديم) أو ``bw_compiler`` (مُجمِّع الخلف). إذا لم يتم تحديد ``bw_compiler``، فإن وظيفة التجميع الخلفي الافتراضية هي نفس وظيفة التجميع الأمامي.

التحذير الوحيد هو أن AOTAutograd يتطلب أن تكون الوظائف المجمعة التي تعيدها الواجهات الخلفية "معلبة". يمكن القيام بذلك عن طريق لف الوظيفة المجمعة مع ``functorch.compile.make_boxed_func``.

على سبيل المثال،

.. code-block:: python

    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func

    def my_compiler(gm, example_inputs):
        return make_boxed_func(gm.forward)

    my_backend = aot_autograd(fw_compiler=my_compiler) # bw_compiler=my_compiler

    model_opt = torch.compile(model, backend=my_backend)

أمثلة
--------

واجهة خلفية التصحيح
^^^^^^^^^^^^^^^^^

إذا كنت تريد فهم ما يحدث بشكل أفضل أثناء التجميع، فيمكنك إنشاء مجمع مخصص، والذي يشار إليه باسم الواجهة الخلفية في هذا القسم، والذي سيطبع بشكل جميل طباعة وحدة نمطية ``GraphModule`` fx المستخرجة من تحليل بايتكود Dynamo
وإرجاع دالة ``forward()``.

على سبيل المثال:

.. code-block:: python

    from typing import List
    import torch
    def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("my_compiler() called with FX graph:")
        gm.graph.print_tabular()
        return gm.forward # return a python callable
    @torch.compile(backend=my_compiler)
    def fn(x, y):
        a = torch.cos(x)
        b = torch.sin(y)
        return a + b
    fn(torch.randn(10), torch.randn(10))

ينتج عن تشغيل المثال أعلاه الإخراج التالي:

::

    my_compiler() called with FX graph:
    opcode         name    target                                                  args        kwargs
    -------------  ------  ------------------------------------------------------  ----------  --------
    placeholder    x       x                                                       ()          {}
    placeholder    y       y                                                       ()          {}
    call_function  cos     <built-in method cos of type object at 0x7f1a894649a8>  (x,)        {}
    call_function  sin     <built-in method sin of type object at 0x7f1a894649a8>  (y,)        {}
    call_function  add     <built-in function add>                                 (cos, sin)  {}
    output         output  output                                                  ((add,),)   {}

هذا يعمل لـ ``torch.nn.Module`` أيضًا كما هو موضح أدناه:

.. code-block:: python

    from typing import List
    import torch
    def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("my_compiler() called with FX graph:")
        gm.graph.print_tabular()
        return gm.forward # return a python callable
    class MockModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()
        def forward(self, x):
            return self.relu(torch.cos(x))
    mod = MockModule()
    optimized_mod = torch.compile(mod, backend=my_compiler)
    optimized_mod(torch.randn(10))

دعونا نلقي نظرة على مثال آخر مع تدفق التحكم:

.. code-block:: python

    from typing import List
    import torch
    def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("my_compiler() called with FX graph:")
        gm.graph.print_tabular()
        return gm.forward # return a python callable
    @torch.compile(backend=my_compiler)
    def toy_example(a, b):
        x = a / (torch.abs(a) + 1)
        if b.sum() < 0:
            b = b * -1
        return x * b
    for _ in range(100):
        toy_example(torch.randn(10), torch.randn(10))

ينتج عن تشغيل هذا المثال الإخراج التالي:

::

    my_compiler() called with FX graph:
    opcode         name     target                                                  args              kwargs
    -------------  -------  ------------------------------------------------------  ----------------  --------
    placeholder    a        a                                                       ()                {}
    placeholder    b        b                                                       ()                {}
    call_function  abs_1    <built-in method abs of type object at 0x7f8d259298a0>  (a,)              {}
    call_function  add      <built-

واجهة خلفية سريعة
^^^^^^^^^^^^^^

من السهل أيضًا دمج الواجهة الخلفية المخصصة التي توفر أداءً متفوقًا، وسندمج واحدة حقيقية
مع `optimize_for_inference <https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html>`__:

.. code-block:: python

    def optimize_for_inference_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        scripted = torch.jit.script(gm)
        return torch.jit.optimize_for_inference(scripted)

بعد ذلك، يجب أن تتمكن من تحسين أي كود موجود باستخدام:

.. code-block:: python

    @torch.compile(backend=optimize_for_inference_compiler)
    def code_to_accelerate():
        ...

الواجهات الخلفية القابلة للتركيب
^^^^^^^^^^^^^^^^^^^

يتضمن TorchDynamo العديد من الواجهات الخلفية، والتي يمكن إدراجها باستخدام
``torch._dynamo.list_backends()``. يمكنك دمج هذه الواجهات الخلفية معًا باستخدام الكود التالي:

.. code-block:: python

    from torch._dynamo import lookup_backend
    def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        try:
            trt_compiled = lookup_backend("tensorrt")(gm, example_inputs)
            if trt_compiled is not None:
                return trt_compiled
        except Exception:
            pass
        # first backend failed, try something else...
        try:
            inductor_compiled = lookup_backend("inductor")(gm, example_inputs)
            if inductor_compiled is not None:
                return inductor_compiled
        except Exception:
            pass
        return gm.forward