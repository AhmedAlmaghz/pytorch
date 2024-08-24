.. _cond:

التحكم في التدفق - Cond
====================

`torch.cond` هو مشغل تدفق تحكم منظم. يمكن استخدامه لتحديد تدفق التحكم الشبيه بـ if-else
ويمكن أن يُنظر إليه منطقيًا على أنه مُنفذ على النحو التالي.

.. code-block:: python

    def cond(
        pred: Union[bool, torch.Tensor],
        true_fn: Callable,
        false_fn: Callable,
        operands: Tuple[torch.Tensor]
    ):
        if pred:
            return true_fn(*operands)
        else:
            return false_fn(*operands)

تكمن قوته الفريدة في قدرته على التعبير عن تدفق التحكم **الاعتماد على البيانات**: فهو يقلل من مشغل شرطي
(`torch.ops.higher_order.cond`)، والذي يحافظ على القيمة التنبؤية، والدالة الصحيحة والدالة الخاطئة.
هذا يفتح مرونة كبيرة في كتابة ونشر النماذج التي تغير بنية النموذج بناءً على
**القيمة** أو **الشكل** لمدخلات أو المخرجات الوسيطة لعمليات tensor.

.. warning::
    `torch.cond` هو ميزة نموذجية في PyTorch. فهو يدعم أنواع الإدخال والإخراج المحدودة
    ولا يدعم التدريب حاليًا. يرجى توقع تنفيذ أكثر استقرارًا في إصدار مستقبلي من PyTorch.
    اقرأ المزيد حول تصنيف الميزات في: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

أمثلة
~~~~

فيما يلي مثال يستخدم "cond" للتفرع بناءً على شكل الإدخال:

.. code-block:: python

    import torch

    def true_fn(x: torch.Tensor):
        return x.cos() + x.sin()

    def false_fn(x: torch.Tensor):
        return x.sin()

    class DynamicShapeCondPredicate(torch.nn.Module):
        """
        استخدام أساسي لـ "cond" بناءً على معيار الشكل الديناميكي.
        """

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            def true_fn(x: torch.Tensor):
                return x.cos()

            def false_fn(x: torch.Tensor):
                return x.sin()

            return torch.cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    dyn_shape_mod = DynamicShapeCondPredicate()

يمكننا تشغيل النموذج بحماس ونتوقع أن تختلف النتائج بناءً على شكل الإدخال:

.. code-block:: python

    inp = torch.randn(3)
    inp2 = torch.randn(5)
    assert torch.equal(dyn_shape_mod(inp), false_fn(inp))
    assert torch.equal(dyn_shape_mod(inp2), true_fn(inp2))

يمكننا تصدير النموذج لإجراء المزيد من التحويلات والنشر:

.. code-block:: python

    inp = torch.randn(4, 3)
    dim_batch = torch.export.Dim("batch", min=2)
    ep = torch.export.export(DynamicShapeCondPredicate(), (inp,), {}, dynamic_shapes={"x": {0: dim_batch}})
    print(ep)

هذا يعطينا برنامجًا مصدرًا كما هو موضح أدناه:

.. code-block::

    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sym_size: Sym(s0) = torch.ops.aten.sym_size.int(arg0_1, 0)
            gt: Sym(s0 > 4) = sym_size > 4;  sym_size = None
            true_graph_0 = self.true_graph_0
            false_graph_0 = self.false_graph_0
            conditional: f32[s0, 3] = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [arg0_1]);  gt = true_graph_0 = false_graph_0 = arg0_1 = None
            return (conditional,)

        class <lambda>(torch.nn.Module):
            def forward(self, arg0_1: f32[s0, 3]):
                cos: f32[s0, 3] = torch.ops.aten.cos.default(arg0_1)
                sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
                add: f32[s0, 3] = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
                return add

        class <lambda>(torch.nn.Module):
            def forward(self, arg0_1: f32[s0, 3]):
                sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
                return sin

لاحظ أن `torch.cond` تم تخفيضه إلى `torch.ops.higher_order.cond`، وأصبحت قيمته التنبؤية تعبيرًا رمزيًا على شكل الإدخال،
وتصبح دالات الفروع رسوميًا فرعيًا لنمط وحدة الرسوم البيانية ذات المستوى الأعلى.

هنا مثال آخر يوضح كيفية التعبير عن تدفق تحكم يعتمد على البيانات:

.. code-block:: python

    class DataDependentCondPredicate(torch.nn.Module):
        """
        استخدام أساسي لـ "cond" بناءً على معيار تنبؤي يعتمد على البيانات.
        """
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cond(x.sum() > 4.0, true_fn, false_fn, (x,))

البرنامج المصدر الذي نحصل عليه بعد التصدير:

.. code-block::

    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sum_1: f32[] = torch.ops.aten.sum.default(arg0_1)
            gt: b8[] = torch.ops.aten.gt.Scalar(sum_1, 4.0);  sum_1 = None

            true_graph_0 = self.true_graph_0
            false_graph_0 = self.false_graph_0
            conditional: f32[s0, 3] = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [arg0_1]);  gt = true_graph_0 = false_graph_0 = arg0_1 = None
            return (conditional,)

        class <lambda>(torch.nn.Module):
            def forward(self, arg0_1: f32[s0, 3]):
                cos: f32[s0, 3] = torch.ops.aten.cos.default(arg0_1)
                sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
                add: f32[s0, 3] = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
                return add

        class <lambda>(torch.nn.Module):
            def forward(self, arg0_1: f32[s0, 3]):
                sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
                return sin


ثوابت "torch.ops.higher_order.cond"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

هناك العديد من الثوابت المفيدة لـ `torch.ops.higher_order.cond`:

- بالنسبة للقيمة التنبؤية:
    - يتم الحفاظ على ديناميكية القيمة التنبؤية (على سبيل المثال، `gt` الموضحة في المثال أعلاه)
    - إذا كانت القيمة التنبؤية في برنامج المستخدم ثابتة (على سبيل المثال، ثابت منطقي Python)، فستكون `pred` للمشغل ثابتة.

- للفروع:
    - ستكون توقيعات الإدخال والإخراج عبارة عن مجموعة فرعية مسطحة.
    - إنها `torch.fx.GraphModule`.
    - تصبح الإغلاقات في الدالة الأصلية إدخالات صريحة. لا توجد إغلاقات.
    - لا يُسمح بإجراء أي طفرات على الإدخالات أو العموميات.

- للوسائط:
    - ستكون أيضًا مجموعة فرعية مسطحة.

- يؤدي تعشيش `torch.cond` في برنامج المستخدم إلى تعشيق وحدات الرسوم البيانية.


مرجع API
---------
.. autofunction:: torch._higher_order_ops.cond.cond