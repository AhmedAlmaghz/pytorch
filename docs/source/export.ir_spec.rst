.. _export.ir_spec:

مواصفات تنسيق IR في torch.export
=============================

Export IR هو تمثيل وسيط (IR) للمترجمين، والذي يشبه MLIR و TorchScript. وهو مصمم خصيصًا للتعبير عن دلالية برامج PyTorch. يمثل Export IR الحساب بشكل أساسي في قائمة مبسطة من العمليات، مع دعم محدود للديناميكية مثل تدفقات التحكم.

لإنشاء رسم بياني لـ Export IR، يمكن استخدام واجهة أمامية تقوم بالتقاط برنامج PyTorch بشكل صحيح عبر آلية التخصص في التتبع. بعد ذلك، يمكن تحسين IR الناتج وتنفيذه بواسطة backend. يمكن القيام بذلك اليوم من خلال: func: `torch.export.export`.

تشمل المفاهيم الرئيسية التي سيتم تغطيتها في هذه الوثيقة ما يلي:

- ExportedProgram: بنية البيانات التي تحتوي على برنامج Export IR
- Graph: الذي يتكون من قائمة من العقد.
- العقد: التي تمثل العمليات وتدفق التحكم والبيانات الوصفية المخزنة في هذه العقدة.
- القيم التي تنتجها العقد ويستهلكها.
- ترتبط الأنواع بالقيم والعقد.
- يتم أيضًا تحديد حجم القيم وتخطيط الذاكرة.

افتراضات
--------

يفترض هذا المستند أن القارئ على دراية كافية بـ PyTorch، وخاصة بـ: class: `torch.fx` وأدواته ذات الصلة. وبالتالي، فإنه سيتوقف عن وصف المحتويات الموجودة في: class: `torch.fx` الوثائق والورق.

ما هو Export IR
-----------------

Export IR هو تمثيل وسيط قائم على الرسم البياني لبرامج PyTorch. يتم تحقيق Export IR بناءً على: class: `torch.fx.Graph`. وبعبارة أخرى، **جميع رسومات Export IR صالحة أيضًا لرسومات FX**، وإذا تم تفسيرها باستخدام الدلالات القياسية لـ FX، فيمكن تفسير IR التصدير بشكل صحيح. أحد الآثار المترتبة على ذلك هو أنه يمكن تحويل الرسم البياني المصدر إلى برنامج Python صالح عبر توليد كود FX القياسي.

ستركز هذه الوثيقة بشكل أساسي على تسليط الضوء على المجالات التي يختلف فيها Export IR عن FX من حيث صرامته، مع تخطي الأجزاء التي تشترك فيها مع FX.

ExportedProgram
---------------

البناء الأعلى لـ Export IR هو فئة: class: `torch.export.ExportedProgram`. فهو يحزم الرسم البياني الحسابي لنموذج PyTorch (والذي يكون عادةً: class: `torch.nn.Module`) مع المعلمات أو الأوزان التي يستهلكها هذا النموذج.

بعض السمات البارزة لفئة: class: `torch.export.ExportedProgram` هي:

- ``graph_module`` (:class:`torch.fx.GraphModule`): بنية البيانات التي تحتوي على الرسم البياني المسطح للنموذج PyTorch. يمكن الوصول إلى الرسم البياني مباشرة من خلال `ExportedProgram.graph`.
- ``graph_signature`` (:class:`torch.export.ExportGraphSignature`): توقيع الرسم البياني، والذي يحدد أسماء المعلمات والمخازن المؤقتة المستخدمة والمعدلة داخل الرسم البياني. بدلاً من تخزين المعلمات والمخازن المؤقتة كسمات للرسم البياني، يتم رفعها كمدخلات للرسم البياني. يتم استخدام توقيع الرسم البياني لتتبع معلومات إضافية حول هذه المعلمات والمخازن المؤقتة.
- ``state_dict`` (``Dict [str، Union [torch.Tensor، torch.nn.Parameter]]``): بنية البيانات التي تحتوي على المعلمات والمخازن المؤقتة.
- ``range_constraints`` (``Dict [sympy.Symbol، RangeConstraint]``): بالنسبة للبرامج التي يتم تصديرها مع سلوك يعتمد على البيانات، ستتضمن البيانات الوصفية لكل عقدة أشكالًا رمزية (تبدو مثل ``s0``، ``i0``). تقوم هذه السمة بتعيين الأشكال الرمزية إلى نطاقاتها الدنيا/العليا.

Graph
-----

رسم بياني لـ Export IR هو برنامج PyTorch ممثَّل في شكل رسم بياني موجَّه غير دوري (DAG). تمثل كل عقدة في هذا الرسم البياني عملية حسابية أو عملية محددة، وتتكون حواف هذا الرسم البياني من مراجع بين العقد.

يمكننا عرض الرسم البياني الذي يحتوي على هذا المخطط:

.. code-block:: python

   class Graph:
     nodes: List[Node]

في الممارسة العملية، يتم تحقيق الرسم البياني لـ Export IR كفئة Python: class: `torch.fx.Graph`.

يحتوي الرسم البياني لـ Export IR على العقد التالية (سيتم وصف العقد بمزيد من التفاصيل في القسم التالي):

- 0 أو أكثر من العقد من النوع "placeholder"
- 0 أو أكثر من العقد من النوع "call_function"
- بالضبط 1 عقدة من النوع "output"

**نتيجة منطقية:** أصغر رسم بياني صالح سيكون من عقدة واحدة. أي أن العقد لا تكون فارغة أبدًا.

**التعريف:**
تمثل مجموعة من العقد "placeholder" في الرسم البياني **مدخلات** الرسم البياني لـ GraphModule. تمثل عقدة "output" في الرسم البياني **الإخراج** من الرسم البياني لـ GraphModule.

مثال::

   from torch import nn

   class MyModule(nn.Module):

       def forward(self، x، y):
         return x + y

   mod = torch.export.export(MyModule ())
   print(mod.graph)

ما سبق هو التمثيل النصي لرسم بياني، مع كل سطر يمثل عقدة.

Node
----

تمثل العقدة عملية حسابية أو عملية محددة ويتم تمثيلها في Python باستخدام فئة: class: `torch.fx.Node`. يتم تمثيل الحواف بين العقد كمراجع مباشرة إلى العقد الأخرى عبر خاصية "args" لفئة Node. باستخدام نفس آلية FX، يمكننا تمثيل العمليات التالية التي يحتاجها الرسم البياني الحسابي عادةً، مثل استدعاءات المشغل، والمواضع الاحتياطية (المعروفة باسم المدخلات)، والعبارات الشرطية، والحلقات.

لدى العقدة المخطط التالي:

.. code-block:: python

   class Node:
     name: str # name of node
     op_name: str  # type of operation

     # interpretation of the fields below depends on op_name
     target: [str|Callable]
     args: List[object]
     kwargs: Dict[str, object]
     meta: Dict[str, object]

**تنسيق نص FX**

كما هو موضح في المثال أعلاه، لاحظ أن لكل سطر هذا التنسيق::

   %<name>: [...] = <op_name> [target=<target>](args = (%arg1، %arg2، arg3، arg4، ...))، kwargs = {"keyword": arg5})

يستحوذ هذا التنسيق على كل ما هو موجود في فئة العقدة، باستثناء "meta"، بتنسيق مضغوط.

بشكل ملموس:

- **<name>** هو اسم العقدة كما سيظهر في ``node.name``.

- **<op_name>** هو حقل "node.op"، والذي يجب أن يكون واحدًا من هذه: `<call_function>`، `<placeholder>`،
  `<get_attr>`، أو `<output>`.

- **<target>** هو الهدف من العقدة كما هو موضح في ``node.target``. يعتمد معنى هذا
  الحقل على "op_name".

- **args1، ... args 4 ...** هي ما هو مدرج في قائمة "node.args". إذا كانت
  القيمة في القائمة عبارة عن: class: `torch.fx.Node`، فسيتم الإشارة إليه بشكل خاص برمز **%.**

على سبيل المثال، ستظهر مكالمة لدالة الإضافة على النحو التالي::

   %add1 = call_function [target = torch.op.aten.add.Tensor] (args = (%x، %y)، kwargs = {})

حيث ``%x``، ``%y`` هما عقدتان أخريان لهما الأسماء x و y. من الجدير بالذكر أن السلسلة "torch.op.aten.add.Tensor" تمثل الكائن القابل للاستدعاء الذي يتم تخزينه بالفعل في حقل الهدف، وليس مجرد اسمه كسلسلة.

السطر الأخير من هذا التنسيق النصي هو::

   return [add]

وهي عقدة مع ``op_name = output``، مما يشير إلى أننا نقوم بإرجاع هذا العنصر.

call_function
^^^^^^^^^^^^^

تمثل عقدة "call_function" استدعاء لمشغل.

**التعاريف**

- **وظيفي:** نقول إن الدالة القابلة للاستدعاء "وظيفية" إذا استوفت جميع المتطلبات التالية:

  - غير متحول: لا يقوم المشغل بتعديل قيمة مدخله (بالنسبة إلى التنسورات، يتضمن ذلك كلاً من البيانات الوصفية والبيانات).
  - لا توجد تأثيرات جانبية: لا يقوم المشغل بتعديل الحالات التي يمكن رؤيتها
    من الخارج، مثل تغيير قيم معلمات الوحدة النمطية.

- **المشغل:** هو دالة قابلة للاستدعاء وظيفية ذات مخطط محدد مسبقًا. تشمل أمثلة
  مثل هذه المشغلات مشغلات ATen الوظيفية.

**التمثيل في FX**

.. code-block::

    %name = call_function [target = operator] (args = (%x، %y، ...)، kwargs = {})


**الاختلافات عن call_function الفانيليا FX**

1. في الرسم البياني لـ FX، يمكن أن تشير "call_function" إلى أي دالة قابلة للاستدعاء، في Export IR، فنحن
   تقييد إلى مجموعة فرعية مختارة فقط من مشغلي ATen والمشغلين المخصصين ومشغلي تدفق التحكم.

2. في Export IR، ستتم تضمين الحجج الثابتة داخل الرسم البياني.

3. في الرسم البياني لـ FX، يمكن أن تمثل عقدة "get_attr" قراءة أي سمة مخزنة في
   :class:`torch.fx.GraphModule`. ومع ذلك، في Export IR، يقتصر هذا على قراءة فقط
   الوحدات الفرعية حيث يتم تمرير جميع المعلمات/المخازن المؤقتة كمدخلات إلى الوحدة النمطية للرسم البياني.

بيانات وصفية
~~~~~~~~

``Node.meta`` عبارة عن قاموس مرفق بكل عقدة FX. ومع ذلك، لا تحدد مواصفات FX البيانات الوصفية التي يمكن أو ستكون موجودة. يوفر Export IR عقدًا أقوى، وتحديدًا جميع العقد "call_function" التي ستضمن وجود الحقول التالية للبيانات الوصفية فقط:

- ``node.meta ["stack_trace"]`` عبارة عن سلسلة تحتوي على تتبع المكدس Python الذي يشير إلى كود المصدر Python الأصلي. يبدو مثال على تتبع المكدس كما يلي::

    File "my_module.py"، line 19، in forward
    return x + dummy_helper (y)
    File "helper_utility.py"، line 89، in dummy_helper
    return y + 1

- ``node.meta ["val"]`` يصف إخراج تشغيل العملية. يمكن أن يكون
  من النوع `<symint>`، `<FakeTensor>`،
  قائمة ``List [Union [FakeTensor، SymInt]]``، أو ``None``.

- ``node.meta ["nn_module_stack"]`` يصف "stacktrace" من: class: `torch.nn.Module` الذي جاءت منه العقدة، إذا كان من: class: `torch.nn.Module` مكالمة. على سبيل المثال، إذا احتوت عقدة تحتوي على عملية "addmm" على مكالمة من: class: `torch.nn.Linear` الوحدة النمطية داخل: class: `torch.nn.Sequential` الوحدة النمطية، فإن "nn_module_stack" ستبدو شيئًا مثل::

    {'self_linear': ('self.linear'، <class 'torch.nn.Linear'>)، 'self_sequential': ('self.sequential'، <class 'torch.nn.Sequential'>)}

- ``node.meta ["source_fn_stack"]`` يحتوي على الدالة ذات اللهب أو: class: `torch.nn.Module` الفئة التي تم استدعاء هذه العقدة منها قبل التفكيك. على سبيل المثال، ستتضمن العقدة التي تحتوي على عملية "addmm" من مكالمة: class: `torch.nn.Linear` الوحدة النمطية: class: `torch.nn.Linear` في "source_fn" الخاصة بهم، وستتضمن العقدة التي تحتوي على عملية "addmm" من مكالمة: class: `torch.nn.functional.Linear` الوحدة النمطية: class: `torch.nn.functional.Linear` في "source_fn" الخاصة بهم.

placeholder
^^^^^^^^^^^

يمثل الموضع الاحتياطي إدخالًا إلى الرسم البياني. دلالتها هي نفسها الموجودة في FX. يجب أن تكون عقد المواضع الاحتياطية هي العقد N الأولى في قائمة العقد في الرسم البياني. يمكن أن يكون N صفرًا.

**التمثيل في FX**

.. code-block:: python

   %name = placeholder [target = name] (args = ())

حقل الهدف هو سلسلة وهي اسم الإدخال.

إذا لم تكن فارغة، فيجب أن يكون "args" بحجم 1 لتمثيل القيمة الافتراضية لهذا الإدخال.

**بيانات وصفية**

تحتوي عقد المواضع الاحتياطية أيضًا على ``meta ['val']``، مثل عقد "call_function". في هذه الحالة، يمثل حقل "val" شكل/نوع البيانات الذي يتوقع الرسم البياني استلامه لهذا معلمة الإدخال.

output
^^^^^^

تمثل مكالمة الإخراج عبارة return في دالة؛ وبالتالي فإنه ينهي الرسم البياني الحالي. هناك عقدة إخراج واحدة فقط، وستكون دائمًا العقدة الأخيرة في الرسم البياني.

**التمثيل في FX**

.. code-block::

   output [] (args = (%something، ...))

لهذا نفس الدلالات الموجودة في: class: `torch.fx`. تمثل "args" العقدة
يتم إرجاعه.

**بيانات وصفية**

تحتوي عقدة الإخراج على نفس البيانات الوصفية مثل عقد "call_function".

get_attr
^^^^^^^^

تمثل عقد "get_attr" قراءة وحدة فرعية من: class: `torch.fx.GraphModule` المضمنة. على عكس الرسم البياني لـ FX العادي من: func: `torch.fx.symbolic_trace` الذي يتم فيه استخدام عقد "get_attr" لقراءة السمات مثل المعلمات والمخازن المؤقتة من: class: `torch.fx.GraphModule` أعلى مستوى، يتم تمرير المعلمات والمخازن المؤقتة كمدخلات إلى الوحدة النمطية للرسم البياني، ويتم تخزينها في: class: `torch.export.ExportedProgram` أعلى مستوى.

**التمثيل في FX**

.. code-block:: python

   %name = get_attr [target = name] (args = ())

**مثال**

ضع في اعتبارك النموذج التالي::

  من functorch.experimental.control_flow import cond

  def true_fn (x):
      return x.sin ()

  def false_fn (x):
      return x.cos ()

  def f (x، y):
      return cond (y، true_fn، false_fn، [x])

Graph::

  الرسم البياني ():
      %x_1: [num_users=1] = placeholder [target=x_1]
      %y_1: [num_users=1] = placeholder [target=y_1]
      %true_graph_0: [num_users=1] = get_attr [target=true_graph_0]
      %false_graph_0: [num_users=1] = get_attr [target=false_graph_0]
      %conditional: [num_users=1] = call_function [target=torch.ops.higher_order.cond] (args = (%y_1، %true_graph_0، %false_graph_0، [%x_1])، kwargs = {})
      return conditional

السطر، ``%true_graph_0: [num_users=1] = get_attr [target=true_graph_0]``،
يقرأ الوحدة الفرعية "true_graph_0" التي تحتوي على عملية "sin".

مراجع

SymInt
^^^^^^

SymInt هو كائن يمكن أن يكون إما عددًا صحيحًا حرفيًا أو رمزًا يمثل عددًا صحيحًا (يمثله في بايثون بواسطة فئة ``sympy.Symbol``). عندما يكون SymInt رمزًا، فهو يصف متغيرًا من النوع العدد الصحيح غير معروف للرسم البياني في وقت الترجمة، أي أن قيمته معروفة فقط في وقت التشغيل.

FakeTensor
^^^^^^^^^^

FakeTensor هو كائن يحتوي على البيانات الوصفية (الميتاداتا) لموتر (tensor). يمكن اعتباره على أنه يحتوي على البيانات الوصفية التالية:

.. code-block:: python

   class FakeTensor:
     size: List[SymInt]
     dtype: torch.dtype
     device: torch.device
     dim_order: List[int]  # هذا لا وجود له بعد

حقل الحجم (size) في FakeTensor هو قائمة من الأعداد الصحيحة أو SymInts. إذا كانت SymInts موجودة، فهذا يعني أن هذا الموتر له شكل ديناميكي. إذا كانت هناك أعداد صحيحة، فيفترض أن يكون للموتر نفس الشكل الثابت. رتبة TensorMeta غير ديناميكية أبدًا. يمثل حقل dtype نوع البيانات (dtype) للناتج من ذلك العقدة. لا توجد ترقيات نوع ضمنية في Edge IR. لا توجد خطوات في FakeTensor.

بعبارة أخرى:

- إذا كان المشغل في node.target يعيد موترًا، فإن ``node.meta['val']`` هو FakeTensor يصف ذلك الموتر.
- إذا كان المشغل في node.target يعيد مجموعة من الموترات، فإن ``node.meta['val']`` هو مجموعة من FakeTensors تصف كل موتر.
- إذا كان المشغل في node.target يعيد عددًا صحيحًا/فاصلة عائمة/مقياسًا معروفًا في وقت الترجمة، فإن ``node.meta['val']`` هو None.
- إذا كان المشغل في node.target يعيد عددًا صحيحًا/فاصلة عائمة/مقياسًا غير معروف في وقت الترجمة، فإن ``node.meta['val']`` هو من النوع SymInt.

على سبيل المثال:

- ``aten::add`` يعيد موترًا؛ لذلك سيكون مواصفاته FakeTensor مع dtype وحجم الموتر الذي يعيده هذا المشغل.
- ``aten::sym_size`` يعيد عددًا صحيحًا؛ لذلك سيكون قيمته SymInt لأن قيمته متاحة فقط في وقت التشغيل.
- ``max_pool2d_with_indexes`` يعيد مجموعة من (موتر، موتر)؛ لذلك فإن المواصفات ستكون أيضًا مجموعة من كائنين FakeTensor، يصف أول كائن TensorMeta العنصر الأول من القيمة المعادة، إلخ.

كود بايثون::

    def add_one(x):
      return torch.ops.aten(x, 1)

الرسم البياني::

    graph():
      %ph_0 : [#users=1] = placeholder[target=ph_0]
      %add_tensor : [#users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%ph_0, 1), kwargs = {})
      return [add_tensor]

FakeTensor::

    FakeTensor(dtype=torch.int, size=[2,], device=CPU)

أنواع Pytree-able
^^^^^^^^^^^^^^^^^

نحن نحدد نوعًا "Pytree-able"، إذا كان نوع ورقة أو نوع حاوية يحتوي على أنواع Pytree-able أخرى.

ملاحظة:

    مفهوم pytree هو نفسه الموثق
    `هنا <https://jax.readthedocs.io/en/latest/pytrees.html>`__ لـ JAX:

الأنواع التالية محددة على أنها **نوع ورقة**:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - النوع
     - التعريف
   * - موتر
     - :class:`torch.Tensor`
   * - قياسي
     - أي أنواع رقمية من بايثون، بما في ذلك الأنواع العددية الصحيحة، وأنواع الفاصلة العائمة، والموترات ذات البعد الصفري.
   * - int
     - بايثون int (مرتبط بـ int64_t في C++)
   * - float
     - بايثون float (مرتبط بـ double في C++)
   * - bool
     - بايثون bool
   * - str
     - سلسلة بايثون
   * - ScalarType
     - :class:`torch.dtype`
   * - Layout
     - :class:`torch.layout`
   * - MemoryFormat
     - :class:`torch.memory_format`
   * - جهاز
     - :class:`torch.device`

الأنواع التالية محددة على أنها **نوع حاوية**:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - النوع
     - التعريف
   * - مجموعة
     - مجموعة بايثون
   * - قائمة
     - قائمة بايثون
   * - قاموس
     - قاموس بايثون بمفاتيح قياسية
   * - NamedTuple
     - مجموعة بايثون مسماة
   * - Dataclass
     - يجب تسجيلها من خلال `register_dataclass <https://github.com/pytorch/pytorch/blob/901aa85b58e8f490631ce1db44e6555869a31893/torch/export/__init__.py#L693>`__
   * - فئة مخصصة
     - أي فئة مخصصة محددة باستخدام `_register_pytree_node <https://github.com/pytorch/pytorch/blob/901aa85b58e8f490631ce1db44e6555869a31893/torch/utils/_pytree.py#L72>`__