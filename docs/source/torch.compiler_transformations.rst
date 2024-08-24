.. _graph-writing-transformations-aten-ir:

كتابة تحويلات المخطط البياني على ATen IR
==================================

تعد القدرة على كتابة تحويلات المخطط البياني ميزة قوية في ATen IR. تتيح هذه الميزة للمستخدمين تخصيص وتعديل مخططاتهم البيانية بطرق متنوعة وقوية. في هذا القسم، سنستكشف أساسيات كتابة تحويلات المخطط البياني على ATen IR، بما في ذلك كيفية تحديد أنماط المخططات البيانية، وتطبيق التحويلات، وإنشاء تحويلات مخصصة.

تحديد أنماط المخطط البياني
---------------------

تعد الخطوة الأولى في كتابة تحويلات المخطط البياني هي القدرة على تحديد أنماط المخططات البيانية التي تريد استهدافها. في ATen IR، يمكنك استخدام التعبيرات المنتظمة للمخططات البيانية (Graph Regular Expressions) أو واجهة نمط المخطط البياني (Graph Pattern Interface) لتحديد أنماط المخططات البيانية.

- التعبيرات المنتظمة للمخططات البيانية: توفر طريقة مرنة وقوية لوصف أنماط المخططات البيانية باستخدام بناء جملة مشابه للتعبيرات المنتظمة. يمكنك تحديد أنماط بناءً على بنية المخطط البياني، وأنواع العقد والحواف، وحتى القيم المرتبطة بها.

- واجهة نمط المخطط البياني: تقدم طريقة أكثر تنظيماً وموجهة نحو الكائنات لتحديد أنماء المخططات البيانية. يمكنك إنشاء كائنات نمط المخطط البياني التي تصف البنية والخصائص المطلوبة، مما يجعل عملية تحديد الأنماط أكثر قابلية للقراءة والتنظيم.

تطبيق التحويلات
------------

بمجرد تحديد نمط المخطط البياني الذي تريد استهدافه، يمكنك تطبيق تحويلات لتعديل المخطط. يمكن أن تشمل هذه التحويلات إضافة أو إزالة العقد والحواف، أو تعديل خصائصها، أو حتى إعادة ترتيب بنية المخطط البياني. يوفر ATen IR مجموعة من الوظائف والواجهات لتطبيق هذه التحويلات بسهولة.

على سبيل المثال، يمكنك استخدام وظيفة ``aten::erase`` لإزالة عقد أو حواف محددة من المخطط البياني. وبالمثل، يمكنك استخدام ``aten::insert`` لإضافة عقد أو حواف جديدة. يمكنك أيضًا استخدام وظائف أكثر تخصصًا مثل ``aten::split`` لتقسيم عقدة إلى عدة عقد أو ``aten::fuse`` لدمج عدة عقد في واحدة.

إنشاء تحويلات مخصصة
---------------------

بالإضافة إلى التحويلات المضمنة التي يوفرها ATen IR، يمكنك أيضًا إنشاء تحويلات مخصصة خاصة بك. يمكن أن يكون هذا مفيدًا عندما يكون لديك متطلبات محددة لا يمكن تلبيتها باستخدام التحويلات القياسية.

لإنشاء تحويل مخصص، يمكنك تعريف وظيفة خاصة بك والتي تأخذ مخططًا بيانيًا كمدخل وتطبق التحويل المطلوب. يمكنك بعد ذلك تسجيل هذه الوظيفة كتحويل في ATen IR واستدعائها مثل أي تحويل آخر.

يوفر ATen IR واجهة مرنة تسمح لك بدمج تحويلاتك المخصصة بسلاسة مع النظام الأساسي، مما يتيح لك توسيع قدرات تحويل المخطط البياني وفقًا لاحتياجاتك المحددة.

أمثلة وتطبيقات
----------------

يمكن أن يكون لكتابة تحويلات المخطط البياني على ATen IR مجموعة واسعة من التطبيقات. فيما يلي بعض الأمثلة على كيفية استخدام هذه الميزة:

- **تحسين الأداء**: يمكنك تحليل مخطط بياني لشبكة عصبية وتحديد أنماط يمكن تحسينها. على سبيل المثال، يمكنك دمج عدة عمليات متتالية في عملية واحدة، أو إزالة العمليات غير الضرورية، مما يؤدي إلى تقليل وقت الحساب واستهلاك الذاكرة.

- **تعديل الشبكات العصبية**: يمكنك تعديل بنية شبكة عصبية موجودة عن طريق إضافة أو إزالة طبقات، أو تعديل اتصالاتها، أو تغيير خصائصها. يمكن أن يكون هذا مفيدًا في نقل التعلم، أو تكييف شبكة عصبية لمهام جديدة، أو حتى تصحيح الأخطاء.

- **إنشاء شبكات عصبية جديدة**: يمكنك أيضًا استخدام تحويلات المخطط البياني لإنشاء شبكات عصبية جديدة تمامًا. يمكنك البدء بمخطط بياني بسيط وتطبيق سلسلة من التحويلات لإنشاء بنية معقدة. يمكن أن يكون هذا مفيدًا في البحث عن بنى جديدة للشبكات العصبية وتصميمها.

في الختام، تعد القدرة على كتابة تحويلات المخطط البياني في ATen IR أداة قوية ومرنة يمكن أن تمكن المستخدمين من تخصيص وتعديل مخططاتهم البيانية بطرق متنوعة. سواء كان الأمر يتعلق بتحسين الأداء أو إنشاء تصاميم جديدة، فإن كتابة تحويلات المخطط البياني تفتح مجموعة واسعة من الإمكانيات للمطورين والباحثين في مجال التعلم العميق.

========================================

تمريرات
------

نظرًا لأن تمثيل ATen IR يقع على مستوى FX Graph/GraphModule، يمكن تطبيق التحويلات المكتوبة لـ FX Graphs بسهولة على تمثيل ATen IR. إذا كنت معتادًا على كتابة تحويلات مخطط FX، فسيكون الأمر نفسه.

أبسط طريقة لكتابة التحويلات هي عن طريق الحلقة عبر المخطط المعطى والتلاعب المباشر بالعقد داخل المخطط.

على سبيل المثال، لنقل أننا نريد استبدال استدعاءات "torch.ops.aten.add.Tensor()" باستدعاءات "torch.ops.aten.mul.Tensor()":

.. code:: python

   import torch

   def replace_add_with_mul(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
       for node in gm.graph.nodes:
           if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
               node.target = torch.ops.aten.mul.Tensor

يمكننا أيضًا حذف العقد وإلحاق عقد جديدة من خلال وظائف FX المساعدة التي يمكن العثور عليها في وثائق "Graph". على سبيل المثال، إذا أردنا إدراج "torch.ops.aten.relu.default()" بعد استدعاء "add":

.. code:: python

   import torch

   def insert_relu_after_add(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
       for node in gm.graph.nodes:
           if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:

               # تحديد نقطة الإدراج. سيتم إدراج أي عقد مضافة إلى المخطط ضمن
               # هذا النطاق بعد العقدة `node`
               with gm.graph.inserting_after(node):
                   # أضف عقدة `call_function` جديدة مع عملية `torch.ops.aten.relu.default`
                   new_relu_node = gm.graph.call_function(torch.ops.aten.relu.default, args=(node,))
                   # استبدل جميع الأماكن التي تستخدم `node` لاستخدام `new_relu_node`
                   node.replace_all_uses_with(new_relu_node)

بشكل عام، يمكن تصنيف التحويلات تقريبًا إلى بضع مجموعات:

المحور أ: 1. إنشاء رسم خريطة واحد إلى X (مثل التفكيك) 2. إنشاء رسم خريطة للكثير إلى واحد (مثل الانصهار)

المحور ب: 1. إجراء تكرار للأمام (مثل انتشار الشكل) 2. إجراء تكرار للخلف (مثل إزالة التعليمات البرمجية غير المستخدمة)

المحور ج: 1. يعتمد على معلومات العقدة المحلية (مثل التحويل الخارجي) 2. يعتمد على معلومات المخطط العالمي (مثل تخطيط الذاكرة)

توقعاتنا لوتيرة هذه الحالات الاستخدامية هي: 1. أ.1، ب.1، ج.1 2. أ.2 3. ب.2، ج.2

على الرغم من أنه يمكننا إجراء جميع تحويلات المخطط من خلال التلاعب المباشر بالمخطط، إلا أننا نوفر أيضًا بعض وظائف المساعدة لتسهيل الاستخدام لحالات الاستخدام من المستوى 1 و2.

المحول
~~~~~~~~~~~

لحالات الاستخدام من المستوى 1 (إنشاء رسم خريطة واحد إلى X، وإجراء تكرارات للأمام، والنظر في معلومات العقدة المحلية)، يمكننا استخدام فئة "المحول" لتنفيذ كل عقدة وإعادة إنشاء مخطط، باستثناء التحويلات المحددة.

تمريرة واحد إلى واحد
^^^^^^^^^^^^^^^

كمثال على رسم الخرائط واحد إلى واحد، إذا أردنا استبدال العملية أ بعملية أخرى ب، فيمكننا تشغيل GraphModule، وفي كل مرة نرى فيها العملية أ، نعيد العملية ب.

مثال على ذلك:

.. code:: python

   class ReplaceAddWithMul(torch.fx.Transformer):
       def call_function(self, target, args, kwargs):
           if target != torch.ops.aten.add.Tensor:
               return super().call_function(target, args, kwargs)
           return super().call_function(torch.ops.aten.mul.Tensor, args, kwargs)

   transformed_graph_module = ReplaceAddWithMul(graph_module).transform()

تؤدي مكالمة "super().call_function(target، args، kwargs، meta)" إلى إنشاء عقدة "call_function" FX، وإعادة نتيجة تشغيل المشغل باستخدام الحجج المعطاة.

تمريرة واحد إلى X
^^^^^^^^^^^^^

إذا أردنا إجراء رسم خرائط واحد إلى X، مثل استبدال العملية أ بعمليتين أخريين ب وج، فسنقوم بعد ذلك بإجراء مكالمتين إلى "super().call_function" لإنشاء عقدتين FX، واحدة مع العملية ب والأخرى مع العملية ج، وإعادة نتيجة تشغيل العملية ج.

على سبيل المثال:

.. code:: python

   class ReplaceAddWithMulSub(torch.fx.Transformer):
       """
       Original:
           def f(x, y):
               return x + y

       After pass:
           def f(x, y):
               z = x * y
               return z - y
       """
       def call_function(self, target, args, kwargs):
           if target != torch.ops.aten.add.Tensor:
               return super().call_function(target, args, kwargs)

           x, y = args

           mul_res = super().call_function(torch.ops.aten.mul.Tensor, args, {})
           return super().call_function(torch.ops.aten.sub.Tensor, (mul_res, y), {})

   transformed_graph_module = ReplaceAddWithMulSub(graph_module).transform()

تمريرة واحد إلى لا شيء
^^^^^^^^^^^^^^^^

إذا أردنا إزالة عملية، فيمكننا ببساطة إعادة القيمة التي تم تمريرها إلى الدالة:

.. code:: python

   class RemoveDetachPass(torch.fx.Transformer):
       def call_function(self, target, args, kwargs):
           if target not in (
               torch.ops.aten.detach.default,
               torch.ops.aten.detach_copy.default,
           ):
               return super().call_function(target, args, kwargs, meta)

           assert len(args) == 1
           return args[0]

   transformed_graph_partum = RemoveDetachPass(graph_module).transform()

الاستفادة من المعلومات المحلية
^^^^^^^^^^^^^^^^^^^^^^^^^^^

مثال على الاستفادة من معلومات العقدة المحلية هو، إذا أردنا تحويل جميع القيم القياسية داخل المخطط إلى tens، فيمكننا تشغيل "fx.GraphModule" المعطى، ولكل حجة تحتوي على قيمة قياسية، نقوم بتحويلها إلى tensor. قد يبدو الأمر كما يلي:

.. code:: python

   def args_map(target, fn, args, kwargs):
       assert isinstance(args, tuple)
       assert isinstance(kwargs, dict)
       args = list(args)
       kwargs = kwargs.copy()

       # تحديث الحجة بناءً على الدالة التي تم تمريرها
       def update(key، args، schema):
           args[key] = fn(args[key]، schema)

       # تحديث كل حجة في المخطط
       for i, schema in enumerate(target._schema.arguments):
           if schema.name in kwargs:
               update(schema.name, kwargs, schema)
           elif not schema.kwarg_only and i < len(args):
               update(i, args, schema)
       return tuple(args), kwargs

   class ScalarToTensorPass (torch.fx.Transformer):
       def call_function (self، target، args، kwargs):
           breakpoint ()

           def try_coerce (value، arg):
               return (
                   torch.tensor (value)
                   if isinstance (value، (float، int، bool))
                   and type (arg.type) == torch.TensorType
                   else value
               )

           args، kwargs = args_map (target، try_coerce، args، kwargs)
           return super().call_function (target، args، kwargs)

   transformed_graph_module = ScalarToTensorPass (graph_module).transform ()

معدل إعادة كتابة المخطط الفرعي
~~~~~~~~~~~~~~~~~~~~~~

لإنشاء رسم خرائط للكثير إلى واحد، يمكننا الاستفادة من "معدل إعادة كتابة المخطط الفرعي" في FX. بالنظر إلى "نمط"، فإنه يقوم بإنشاء مخطط فرعي من المشغلين المطابقين للنمط، ثم يستبدل كل مخطط فرعي متطابق بـ "الاستبدال".

ملاحظة:

::

   هذه عملية في المكان.

يجب أن تكون إدخالات "النمط" و"الاستبدال" دالات أو GraphModules قابلة للاستدعاء تحتوي على نفس المشغلين المستخدمين داخل المخطط (عمليات ATen) حتى يتمكن معدل إعادة كتابة المخطط الفرعي من العثور على النمط الصحيح في المخطط. سيتم التعامل مع الإدخالات إلى الدالات القابلة للاستدعاء في النمط/الاستبدال كحرف بري عند المطابقة.

مثال:

.. code:: python

   from torch.fx import subgraph_rewriter

   def replace_patterns(graph_module):
       def pattern(x, y):
           x = torch.ops.aten.add.Tensor(x, y)
           x = torch.ops.aten.mul.Tensor(x, y)
           return x

       def replacement(x, y):
           return torch.ops.aten.sub.Tensor(x, y)

   replaced_patterns = subgraph_rewriter.replace_pattern_with_filters(
       traced_module, pattern, replacement
   )

يعيد معدل إعادة كتابة المخطط الفرعي قائمة من "ReplacedPatterns":

.. code:: python

   @dataclass
   class ReplacedPatterns:
       # العقدة التي تم العثور على المطابقة منها
       anchor: Node
       # يقوم برسم خريطة للعقد في المخطط الفرعي للنمط إلى العقد في المخطط الأكبر
       nodes_map: Dict[Node, Node]
       # قائمة العقد التي تمت إضافتها إلى المخطط
       replacements: List[Node]

ملاحظة:

::

   لن تحتوي العقد التي تم إنشاؤها بواسطة معدل إعادة كتابة المخطط الفرعي على البيانات الوصفية التي
   يتم تعبئتها في العقد المتطابقة، ولكن يمكنك استخدام "ReplacedPatterns.nodes_map" للعثور على العقد في المخطط الأصلي
   التي تم مطابقتها، و "ReplacedPatterns.replacements" للعثور على العقد التي
   تم استبدالها في المخطط المحول.

مدير التمرير
------------

"مدير التمرير" هو فئة تستخدم لتشغيل تمريرات متعددة على Graph Module معين. عند تهيئة مثيل "مدير التمرير"، نقوم بتمرير قائمة من التمريرات التي نريد تشغيلها وتعيين بعض العلامات. لتشغيل مجموعة التمريرات على Graph Module، يمكننا تمرير Graph Module مباشرة إلى مثيل "مدير التمرير".

مثال:

.. code:: python

   from torch.fx.passes.infra.pass_manager import PassManager

   pm = PassManager(
       passes=[replace_add_with_div, replace_div_with_mul],
       run_checks_after_each_pass=True,
       suppress_check_failures=False,
   )
   graph_module_out = pm(graph_module)

لإضافة مجموعة شائعة من الفحوصات التي يتم تشغيلها بعد كل تمريرة، يمكننا استدعاء الدالة "set_checks(check: Callable)" التي تأخذ دالة قابلة للاستدعاء كإدخال. إذا تم تعيين العلامة "run_checks_after_each_pass"، فسيتم استدعاء "الفحص" بعد كل تمريرة يتم تشغيلها على Graph Module.

مثال:

.. code:: python

   pm = PassManager(passes=[replace_add_with_div, replace_div_with_mul])

   def check_div_target(graph_module):
       for node in graph_module.graph.nodes:
           if node.op == "call_function" and node.target != torch.div:
               raise ValueError("Target should be div!")

   pm.add_checks(check_div_target)

   pm(graph_module)    # raises ValueError after replace_div_with_mul pass

مُقسِّم
هناك بعض الأدوات الشائعة المعتمدة على مخطط FX والتي يمكننا استخدامها لتقسيم المخطط.

مطابقة المخطط الفرعي
~~~~~~~~~~~~~~~

لإيجاد المخططات الفرعية داخل مخطط ما والتي تتطابق مع نمط محدد، يمكننا استخدام
```SubgraphMatcher`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/utils/matcher_utils.py>`__
من FX.

خصائص الفئة:

-  ``pattern (Graph)``: نمط المطابقة المستهدف. ستُعامل العقد الاحتياطية
   في المخطط كحروف برية عند المطابقة.
-  ``match_output (bool)``: إذا كانت True، ستُعامل العقدة الإخراجية في
   مخطط النمط كجزء من النمط المستهدف. إذا كانت False، ستتم تجاهل العقدة
   الإخراجية أثناء المطابقة.
-  ``match_placeholder (bool)``: إذا كانت True، ستُعامل العقدة الاحتياطية
   في مخطط النمط كجزء من النمط المستهدف. إذا كانت False، ستُستخدم العقد
   الاحتياطية كحروف برية.
-  ``remove_overlapping_matches (bool)``: إذا كانت True، في حالة
   التطابقات المتداخلة، ستتم إعادة أول تطابق فقط.
-  ``ignore_literals (bool)``: إذا كانت True، لن يتم التحقق مما إذا كانت
   القيم الحرفية متساوية وسيتم معاملتها بدلاً من ذلك كحروف برية.

مثال:

.. code:: python

   from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

   class LargeModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self._weight = torch.nn.Parameter(torch.ones(3, 3))
           self._bias = torch.nn.Parameter(torch.ones(3, 3))

       def forward(self, x):
           return torch.ops.aten.addmm.default(self._bias, x, self._weight)

   large_model_graph = torch.export(LargeModel(), inputs).graph

   class PatternModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
           self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

       def forward(self, x):
           return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

   pattern_graph = torch.export(PatternModel(), inputs).graph

   subgraph_matcher = SubgraphMatcher(pattern_graph)
   match_result = subgraph_matcher.match(large_model_graph)

تعيد دالة ``match`` قائمة من ``InternalMatch``:

.. code:: python

   @dataclass
   class InternalMatch():
       # العقد التي تم العثور على المطابقة منها
       anchors: List[Node]
       # يقوم برسم خريطة للعقد في المخطط الفرعي للنمط إلى العقد في المخطط الأكبر
       nodes_map: Dict[Node, Node] = field(default_factory=dict)
       # العقد في المخطط المستهدف التي تتطابق مع العقدة الاحتياطية في النمط
       placeholder_nodes: List[Node] = field(default_factory=list)
       # العقد في المخطط الفرعي المتطابق التي تم إرجاعها بواسطة الإخراج
       returning_nodes: List[Node] = field(default_factory=list)

مُقسِّم قائم على القدرات
~~~~~~~~~~~~~~~~

لإيجاد أكبر المخططات الفرعية من العقد التي تدعم خاصية محددة، يمكننا استخدام
```CapabilityBasedPartitioner`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/infra/partitioner.py#L34>`__
من FX.

خصائص الفئة:

-  ``graph_module (torch.fx.GraphModule)``: وحدة المخطط النمطي التي
   نقوم بالتقسيم عليها.
-  ``operator_support (OperatorSupportBase)``: الكائن المستخدم لتحديد ما
   إذا كانت العقدة في المخطط مدعومة في التقسيم.
-  ``allows_single_node_partition (bool)``: إذا كانت True، تسمح بتكوين
   تقسيمات العقدة المفردة.
-  ``non_compute_ops (Optional[Sequence[str]])``: مجموعة من العمليات التي
   تعتبر "غير حاسوبية" (مثل ``torch.ops.aten.view`` و ``_operator.getitem``)،
   بحيث لا يقوم المُقسِّم بإنشاء مخططات تحتوي فقط على هذه العمليات غير
   الحاسوبية
-  ``allowed_single_node_partition_ops (Optional[Sequence[str]])``: مجموعة
   من العمليات المسموح بها في تقسيم العقدة المفردة.

تستخدم فئة
```OperatorSupportBase`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#LL28C1-L28C1>`__
بواسطة المُقسِّم لتحديد ما إذا كانت عقدة محددة في المخطط تنتمي إلى التقسيم.
ويتم ذلك عن طريق تجاوز دالة ``is_node_supported``. يمكنك توصيل عدة
فئات ``OperatorSupportBase`` باستخدام
```chain`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#L150>`__
(التي تعيد False إذا أعادت أي من فئات OperatorSupportBase القيمة False) و
```any_chain`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#L164>`__
(التي تعيد القيمة True إذا أعادت أي من فئات OperatorSupportBase القيمة
True).

مثال:

.. code:: python

   from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
   from torch.fx.passes.operator_support import any_chain, OperatorSupportBase

   class AddMulOperatorSupport(OperatorSupportBase):
       def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
           return node.op == "call_function" and node.target in [
               torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor,
           ]

   capability_partitioner = CapabilityBasedPartitioner(
       graph_module,
       op_support,
   )

   # إرجاع قائمة من التقسيمات (قائمة بالعقد التي تنتمي إلى كل تقسيم)
   partition_list = capability_partitioner.propose_partitions()
   # دمج التقسيمات في وحدات مخطط نمطي وإدراج عقد `call_module` في المخطط
   fused_graph_module = capability_partitioner.fuse_partitions(partition_list)