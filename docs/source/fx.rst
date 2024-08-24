.. currentmodule:: torch.fx

تورتش. إف.إكس Torch.fx
================

نظرة عامة
------------
.. automodule:: torch.fx

.. _كتابة التحولات:

كتابة التحولات
----------

هل يمكنك إخباري بالمزيد عن طبيعة التحولات التي تريد تطبيقها؟ هل هناك أي متطلبات أو قيود محددة؟ سيساعدني ذلك في تخصيص الإجابة لتلبية احتياجاتك بشكل أفضل.
ما هو تحويل FX؟ ببساطة، هو دالة تبدو على النحو التالي:

::

    import torch
    import torch.fx

    def transform(m: nn.Module,
                  tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
        # الخطوة 1: الحصول على رسم بياني يمثل الكود في 'm'

        # ملاحظة: torch.fx.symbolic_trace عبارة عن غلاف حول مكالمة إلى
        # fx.Tracer.trace وإنشاء GraphModule. سنقوم
        # بتقسيم ذلك في تحويلنا للسماح للمستدعي بتخصيص
        # سلوك التتبع.
        graph : torch.fx.Graph = tracer_class().trace(m)

        # الخطوة 2: تعديل هذا الرسم البياني أو إنشاء رسم بياني جديد
        graph = ...

        # الخطوة 3: إنشاء وحدة نمطية لإرجاعها
        return torch.fx.GraphModule(m, graph)

سيقوم تحويلك بأخذ :class:`torch.nn.Module`، والحصول على :class:`Graph`
منه، وإجراء بعض التعديلات، وإرجاع وحدة نمطية جديدة
:class:`torch.nn.Module`. يجب أن تفكر في وحدة :class:`torch.nn.Module` التي يعيدها تحويل FX على أنها مطابقة لوحدة :class:`torch.nn.Module` العادية - يمكنك تمريرها إلى تحويل FX آخر، أو يمكنك تمريرها إلى TorchScript، أو يمكنك
تشغيلها. إن ضمان أن تكون مدخلات ومخرجات تحويل FX الخاص بك عبارة عن
:class:`torch.nn.Module` ستسمح بإمكانية التكوين.

.. note::

    من الممكن أيضًا تعديل وحدة :class:`GraphModule` موجودة بدلاً من
    إنشاء وحدة جديدة، كما هو موضح أدناه::

        import torch
        import torch.fx

        def transform(m : nn.Module) -> nn.Module:
            gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

            # تعديل gm.graph
            # <...>

            # إعادة تجميع طريقة forward() ل 'gm' من Graph الخاص بها
            gm.recompile()

            return gm

    لاحظ أنه يجب عليك استدعاء :meth:`GraphModule.recompile` لمزامنة طريقة ``forward()`` المولدة
    في ``GraphModule`` مع :class:`Graph` المعدل.

نظرًا لأنك مررت بوحدة :class:`torch.nn.Module` التي تم تتبعها إلى
:class:`Graph`، فهناك الآن نهجان رئيسيان يمكنك اتباعهما لبناء :class:`Graph` جديد.

مقدمة سريعة حول الرسوم البيانية
^^^^^^^^^^^^^^^^^^^^^^^

يمكن العثور على المعالجة الكاملة لدلالات الرسوم البيانية في وثائق :class:`Graph`،
ولكننا سنغطي الأساسيات هنا. :class:`Graph`
هو هيكل بيانات يمثل طريقة في :class:`GraphModule`.
تتطلب المعلومات ما يلي:

- ما هي المدخلات للطريقة؟
- ما هي العمليات التي تعمل داخل الطريقة؟
- ما هي قيمة الإخراج (أي الإرجاع) من الطريقة؟

جميع هذه المفاهيم الثلاثة ممثلة بواسطة مثيلات :class:`Node`.
دعونا نرى ما نعنيه بذلك مع مثال قصير:

::

    import torch
    import torch.fx

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x):
            return torch.topk(torch.sum(
                self.linear(x + self.linear.weight).relu(), dim=-1), 3)

    m = MyModule()
    gm = torch.fx.symbolic_trace(m)

    gm.graph.print_tabular()

هنا نحدد وحدة ``MyModule`` لأغراض العرض التوضيحي، وننشئ مثيلًا لها،
ونتبعها رمزيًا، ثم نستدعي طريقة :meth:`Graph.print_tabular` لطباعة
جدول يعرض العقد في هذا الرسم البياني :class:`Graph`:

    +---------------+---------------+----------------------------+--------------------+-------------+
    | opcode        | name          | target                     | args               | kwargs      |
    +===============+===============+============================+====================+=============+
    | placeholder   | x             | x                          | ()                 | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | get_attr      | linear_weight | linear.weight              | ()                 | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_function | add_1         | <built-in function add>    | (x, linear_weight) | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_module   | linear_1      | linear                     | (add_1,)           | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_method   | relu_1        | relu                       | (linear_1,)        | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_function | sum_1         | <built-in method sum ...>  | (relu_1,)          | {'dim': -1} |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_function | topk_1        | <built-in method topk ...> | (sum_1, 3)         | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | output        | output        | output                     | (topk_1,)          | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+

يمكننا استخدام هذه المعلومات للإجابة على الأسئلة التي طرحناها أعلاه.

- ما هي المدخلات للطريقة؟ في FX، يتم تحديد مدخلات الطريقة
  عبر عقد ``placeholder`` الخاصة. في هذه الحالة، لدينا عقدة ``placeholder`` واحدة
  مع ``target`` من ``x``، مما يعني أن لدينا
  حجة واحدة (غير ذاتية) تسمى x.
- ما هي العمليات داخل الطريقة؟ تمثل عقد ``get_attr`` و
  ``call_function`` و ``call_module`` و ``call_method``
  العمليات في الطريقة. يمكن العثور على معالجة كاملة
  لدلالات جميع هذه العمليات في وثائق :class:`Node`.
- ما هي قيمة الإرجاع للطريقة؟ يتم تحديد قيمة الإرجاع في
  :class:`Graph` بواسطة عقدة ``output`` الخاصة.

نظرًا لأننا نعرف الآن أساسيات كيفية تمثيل الكود في
FX، يمكننا الآن استكشاف كيفية تحرير :class:`Graph`.

التلاعب بالرسم البياني
^^^^^^^^^^^^^^^^^^^^^^

التلاعب المباشر بالرسم البياني
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

هناك نهج لبناء هذا الرسم البياني الجديد وهو التلاعب المباشر بالرسم البياني القديم. للمساعدة في ذلك، يمكننا ببساطة أخذ الرسم البياني :class:`Graph` الذي نحصل عليه من التتبع الرمزي وتعديله. على سبيل المثال، لنفترض أننا نرغب في استبدال
مكالمات :func:`torch.add` بمكالمات :func:`torch.mul`.

::

    import torch
    import torch.fx

    # وحدة نمطية العينة
    class M(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    def transform(m: torch.nn.Module,
                  tracer_class : type = fx.Tracer) -> torch.nn.Module:
        graph : fx.Graph = tracer_class().trace(m)
        # يمثل FX الرسم البياني الخاص به على أنه قائمة مرتبة من
        # العقد، لذلك يمكننا التكرار خلالها.
        for node in graph.nodes:
            # يتحقق مما إذا كنا نستدعي دالة (أي:
            # torch.add)
            if node.op == 'call_function':
                # تحتوي السمة target على الدالة
                # التي يستدعيها call_function.
                if node.target == torch.add:
                    node.target = torch.mul

        graph.lint() # يقوم ببعض الفحوصات للتأكد من أن الرسم البياني
                     # له شكل جيد.

        return fx.GraphModule(m, graph)


يمكننا أيضًا إجراء عمليات إعادة كتابة أكثر تعقيدًا للرسم البياني، مثل
حذف العقد أو إضافتها. للمساعدة في هذه التحويلات،
يحتوي FX على وظائف مساعدة لتحويل الرسم البياني والتي
يمكن العثور عليها في وثائق :class:`Graph`. ويمكن
الاطلاع على مثال لاستخدام هذه الواجهات البرمجية لإضافة مكالمة :func:`torch.relu`
أدناه.

::

    # يحدد نقطة الإدراج. سيتم إدراج أي عقد مضافة إلى الرسم البياني
    # ضمن هذا النطاق بعد 'node'
    with traced.graph.inserting_after(node):
        # أضف عقدة 'call_function' جديدة تستدعي 'torch.relu'
        new_node = traced.graph.call_function(
            torch.relu, args=(node,))

        # نريد أن تستخدم جميع الأماكن التي استخدمت قيمة 'node'
        # الآن تلك القيمة بعد مكالمة 'relu' التي أضفناها.
        # نستخدم واجهة برمجة التطبيقات 'replace_all_uses_with' للقيام بذلك.
        node.replace_all_uses_with(new_node)

بالنسبة للتحويلات البسيطة التي تتكون فقط من الاستبدالات، يمكنك أيضًا
الاستفادة من "مُعيد كتابة الرسم البياني الفرعي". <https://github.com/pytorch/pytorch/blob/main/torch/fx/subgraph_rewriter.py>`__

إعادة كتابة الرسم البياني الفرعي باستخدام replace_pattern()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

يوفر FX أيضًا مستوى آخر من الأتمتة فوق التلاعب المباشر بالرسم البياني.
واجهة برمجة التطبيقات :func:`replace_pattern` هي في الأساس أداة "بحث/استبدال" لتحرير
الرسوم البيانية :class:`Graph`. يسمح لك بتحديد دالة ``pattern`` و ``replacement``، وسيتتبع هذه الدوال، ويجد حالات مجموعة العمليات
في الرسم البياني ``pattern``، ويستبدل تلك الحالات بنسخ من الرسم البياني ``replacement``. يمكن أن يساعد هذا في أتمتة رمز التلاعب بالرسم البياني المرهق، والذي يمكن أن يصبح غير عملي مع زيادة تعقيد التحويلات.

أمثلة على التلاعب بالرسم البياني
~~~~~~~~~~~~~~~~~~~~~~

-  `استبدال عملية واحدة
   <https://github.com/pytorch/examples/blob/master/fx/replace_op.py>`__
-  `دمج التحويل/الدفعة
   <https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50>`__
-  `replace_pattern: الاستخدام الأساسي <https://github.com/pytorch/examples/blob/master/fx/subgraph_rewriter_basic_use.py>`__
-  `التحويل <https://pytorch.org/docs/main/quantization.html#prototype-fx-graph-mode-quantization>`__
-  `تحويل العكس <https://github.com/pytorch/examples/blob/master/fx/invert.py>`__

الوكيل/إعادة التتبع
^^^^^^^^^^^^^^

هناك طريقة أخرى للتلاعب بالرسوم البيانية :class:`Graph` وهي إعادة استخدام آلية الوكيل
المستخدمة في التتبع الرمزي. على سبيل المثال، دعونا
نتخيل أننا أردنا كتابة تحويل يقوم بتفكيك وظائف PyTorch إلى عمليات أصغر. سيحول كل
مكالمة ``F.relu(x)`` إلى ``(x > 0) * x``. إحدى الإمكانيات هي
أداء إعادة كتابة الرسم البياني المطلوبة لإدراج المقارنة والضرب بعد
``F.relu``، ثم قم بتنظيف ``F.relu`` الأصلي. ومع ذلك، يمكننا أتمتة هذه العملية باستخدام كائنات :class:`Proxy`
لتسجيل العمليات تلقائيًا في الرسم البياني.

لاستخدام هذه الطريقة، نكتب العمليات التي نريد إدراجها ككود PyTorch عادي ونستدعي ذلك الكود باستخدام كائنات :class:`Proxy` كوسيط.
ستقوم كائنات :class:`Proxy` هذه بالتقاط العمليات التي يتم إجراؤها عليها وإضافتها إلى الرسم البياني.
بالتأكيد! إليك النص مترجمًا إلى اللغة العربية بتنسيق ReStructuredText:

::

    # لاحظ أن قاعدة التفكيك هذه يمكن قراءتها ككود Python عادي
    def relu_decomposition(x):
        return (x > 0) * x

    decomposition_rules = {}
    decomposition_rules[F.relu] = relu_decomposition

    def decompose(model: torch.nn.Module،
                  tracer_class : type = fx.Tracer) -> torch.nn.Module:
        """
        فك `model` إلى عمليات مكونة أصغر.
        حاليًا، يدعم هذا فقط تفكيك ReLU إلى تعريفه الرياضي: (x > 0) * x
        """
        graph : fx.Graph = tracer_class().trace(model)
        new_graph = fx.Graph()
        env = {}
        tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
        لكل عقدة في عقد الرسم البياني:
            إذا كانت node.op == 'call_function' وnode.target في decomposition_rules:
                # عن طريق لف الحجج بالوكلاء،
                # يمكننا التوزيع إلى قاعدة التفكيك المناسبة
                # وإضافته ضمنيًا إلى الرسم البياني عن طريق التتبع الرمزي له.
                proxy_args = [
                    fx.Proxy(env[x.name], tracer) إذا كان isinstance(x, fx.Node) else x لكل x في node.args]
                output_proxy = decomposition_rules[node.target](*proxy_args)

                # العمليات على `Proxy` تنتج دائمًا وكلاء جدد، و
                # قيمة الإرجاع لقاعدة التفكيك لدينا ليست استثناء.
                # نحن بحاجة إلى استخراج `Node` الأساسي من `Proxy`
                # لاستخدامه في التكرارات اللاحقة لهذا التحويل.
                new_node = output_proxy.node
                env[node.name] = new_node
            else:
                # الحالة الافتراضية: ليس لدينا قاعدة تفكيك لهذه العقدة،
                # لذا قم بنسخ العقدة إلى الرسم البياني الجديد.
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
        return fx.GraphModule(model, new_graph)

بالإضافة إلى تجنب التلاعب الصريح بالرسم البياني، فإن استخدام فئة "Proxy"
يسمح لك أيضًا بتحديد قواعد إعادة الكتابة ككود Python أصلي.
بالنسبة للتحويلات التي تتطلب عددًا كبيرًا من قواعد إعادة الكتابة
(مثل vmap أو grad)، يمكن أن يؤدي هذا غالبًا إلى تحسين قابلية القراءة
والصيانة للقواعد. لاحظ أنه عند استدعاء فئة "Proxy"، قمنا أيضًا
بتمرير tracer يشير إلى المتغير الأساسي "graph". يتم ذلك حتى في حالة
العمليات في الرسم البياني ثنائي (مثل إضافة عامل تشغيل ثنائي)،
لا يقوم استدعاء فئة "Proxy" بإنشاء مثيلات متعددة من tracer الرسم البياني
والذي يمكن أن يؤدي إلى أخطاء وقت التشغيل غير المتوقعة. نوصي بهذه الطريقة
من استخدام فئة "Proxy" خاصة عندما لا يمكن افتراض أن المشغلين الأساسيين
يكونون أحاديين بشكل آمن.

يمكن العثور على مثال عملي لاستخدام فئة "Proxy" للتلاعب بالرسم البياني
`هنا <https://github.com/pytorch/examples/blob/master/fx/proxy_based_graph_creation.py>`__.

نمط المترجم
^^^^^^^^^^^^

نمط مفيد لتنظيم التعليمات البرمجية في FX هو الحلقة عبر جميع الفئات "Node"
في فئة "Graph" وتنفيذها. يمكن استخدام هذا لعدة أشياء بما في ذلك
تحليل وقت التشغيل للقيم التي تتدفق عبر الرسم البياني أو تحويل التعليمات البرمجية
عن طريق إعادة التتبع باستخدام فئة "Proxy". على سبيل المثال، افترض أننا نريد
تشغيل فئة "GraphModule" وتسجيل خصائص الشكل ونوع العنصر لـ فئة "torch.Tensor"
على العقد كما نراها في وقت التشغيل. قد يبدو ذلك على النحو التالي:

::

    import torch
    import torch.fx
    from torch.fx.node import Node

    from typing import Dict

    class ShapeProp:
        """
        انتشار الشكل. تأخذ هذه الفئة `GraphModule`.
        بعد ذلك، تنفذ طريقة "propagate" الخاصة بها `GraphModule`
        عقدة تلو الأخرى مع الحجج المعطاة. مع تنفيذ كل عملية،
        تخزن فئة ShapeProp بعيدًا الشكل و
        أنواع العناصر لقيم الإخراج لكل عملية على
        سمات "shape" و"dtype" لعملية "Node".
        """
        def __init__(self, mod):
            self.mod = mod
            self.graph = mod.graph
            self.modules = dict(self.mod.named_modules())

        def propagate(self, *args):
            args_iter = iter(args)
            env : Dict[str, Node] = {}

            def load_arg(a):
                return torch.fx.graph.map_arg(a, lambda n: env[n.name])

            def fetch_attr(target : str):
                target_atoms = target.split('.')
                attr_itr = self.mod
                لكل i، atom في enumerate(target_atoms):
                    إذا لم يكن hasattr(attr_itr, atom):
                        raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                    attr_itr = getattr(attr_itr, atom)
                return attr_itr

            لكل عقدة في عقد الرسم البياني:
                إذا كانت node.op == 'placeholder':
                    النتيجة = next(args_iter)
                elif node.op == 'get_attr':
                    النتيجة = fetch_attr(node.target)
                elif node.op == 'call_function':
                    النتيجة = node.target(*load_arg(node.args), **load_arg(node.kwargs))
                elif node.op == 'call_method':
                    self_obj, *args = load_arg(node.args)
                    kwargs = load_arg(node.kwargs)
                    النتيجة = getattr(self_obj, node.target)(*args, **kwargs)
                elif node.op == 'call_module':
                    النتيجة = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

                # هذا هو الكود الوحيد المحدد لانتشار الشكل.
                # يمكنك حذف فرع "if" هذا ويصبح هذا
                # مترجم "GraphModule" عام.
                إذا كان isinstance(result, torch.Tensor):
                    node.shape = result.shape
                    node.dtype = result.dtype

                env[node.name] = result

            return load_arg(self.graph.result)

كما ترون، فإن مترجم FX الكامل ليس معقدًا
ولكن يمكن أن يكون مفيدًا جدًا. لتسهيل استخدام هذا النمط، نقدم
فئة "Interpreter" التي تشمل المنطق أعلاه
بطريقة يمكن بها تجاوز جوانب معينة من تنفيذ المترجم
عبر تجاوزات الأساليب.

بالإضافة إلى تنفيذ العمليات، يمكننا أيضًا إنشاء رسم بياني جديد
من خلال تمرير قيم "Proxy" عبر مترجم.
وبالمثل، نقدم فئة "Transformer" لتشمل
هذا النمط. تتصرف فئة "Transformer" بشكل مشابه لفئة "Interpreter"،
ولكن بدلاً من استدعاء طريقة "run" للحصول على قيمة مخرجات ملموسة من الوحدة النمطية،
ستقوم باستدعاء طريقة "transform" للعودة إلى فئة "GraphModule" جديدة
تخضع لأي قواعد تحويل قمت بتثبيتها كتجاوزات للأساليب.

أمثلة على نمط المترجم
~~~~~~~~~~~~~~

-  `انتشار الشكل <https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py>`__
-  `مُوصِّف الأداء <https://github.com/pytorch/tutorials/pull/1319>`__


تصحيح الأخطاء
.. _introduction:

مقدمة
^^^^^^^^

غالباً ما لا يكون الكود الخاص بنا صحيحًا تمامًا أثناء تأليف التحولات. في هذه الحالة، قد نحتاج إلى إجراء بعض التصحيح. المفتاح هو العمل
عكسيا: أولاً، تحقق من نتائج استدعاء الوحدة النمطية المولدة لإثبات أو
نفي الصحة. ثم، تفحص الكود المولد وقم بتصحيحه. بعد ذلك، قم بتصحيح عملية التحولات التي أدت إلى الكود المولد.

إذا لم تكن على دراية بأدوات التصحيح، فيرجى الاطلاع على القسم المساعد
:ref:`أدوات التصحيح المتاحة`.

.. _المزالق الشائعة في تأليف التحولات:

المزالق الشائعة في تأليف التحولات
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ترتيب التكرار غير المحدد لمجموعة "set". في بايثون، نوع البيانات "set" غير مرتب. استخدام "set" لاحتواء مجموعات من الكائنات مثل العقد،
على سبيل المثال، يمكن أن يسبب عدم تحديد غير متوقع. أحد الأمثلة هو التكرار
عبر مجموعة من العقد لإدراجها في رسم بياني. نظرًا لأن نوع البيانات "set" غير مرتب، فإن ترتيب العمليات في برنامج الإخراج
سيكون غير محدد ويمكن أن يتغير عبر استدعاءات البرنامج. البديل الموصى به هو استخدام نوع البيانات "dict"، والذي يكون
`مرتب حسب الإدراج <https://mail.python.org/pipermail/python-dev/2017-December/151283.html>`_
اعتبارًا من بايثون 3.7 (واعتبارًا من cPython 3.6). يمكن استخدام "dict" بشكل مماثل
لمجموعة عن طريق تخزين القيم المراد إزالة التكرارات منها في مفاتيح "dict".

.. _التحقق من صحة الوحدات النمطية:

التحقق من صحة الوحدات النمطية
^^^^^^^^^^^^^^^^^^^^^^^^

نظرًا لأن إخراج معظم وحدات التعلم العميق يتكون من مثيلات النقطة العائمة :class:`torch.Tensor`، فإن التحقق من التكافؤ بين
نتائج وحدتين من النوع :class:`torch.nn.Module` ليس بسيطًا
مثل إجراء فحص المساواة. لتوضيح ذلك، دعنا نستخدم مثالًا:

::

    import torch
    import torch.fx
    import torchvision.models as models

    def transform(m : torch.nn.Module) -> torch.nn.Module:
        gm = torch.fx.symbolic_trace(m)

        # تخيل أننا نقوم ببعض التحولات هنا
        # <...>

        gm.recompile()

        return gm

    resnet18 = models.resnet18()
    transformed_resnet18 = transform(resnet18)

    input_image = torch.randn(5, 3, 224, 224)

    assert resnet18(input_image) == transformed_resnet18(input_image)
    """
    RuntimeError: قيمة منطقية للنسخة غير واضحة
    """

هنا، حاولنا التحقق من مساواة قيم نموذجين للتعلم العميق باستخدام عامل التشغيل "==". ومع ذلك، هذا غير محدد جيدًا
بسبب مشكلة أن المشغل يعيد نسخة وليس قيمة منطقية، ولكن أيضًا لأن مقارنة قيم النقطة العائمة
يجب أن تستخدم هامش خطأ (أو "epsilon") لمراعاة
عدم التبادلية لعمليات النقطة العائمة (راجع
`هنا <https://floating-point-gui.de/errors/comparison/>`__ لمزيد من التفاصيل). يمكننا استخدام :func:`torch.allclose` بدلاً من ذلك، والذي سيعطينا
مقارنة تقريبية تأخذ في الاعتبار عتبة تسامح نسبي ومطلق:

::

    assert torch.allclose(resnet18(input_image), transformed_resnet18(input_image))

هذه هي الأداة الأولى في صندوق أدواتنا للتحقق مما إذا كانت الوحدات النمطية المحولة تتصرف كما هو متوقع مقارنة بالتنفيذ المرجعي.

.. _تصحيح الكود المولد:

تصحيح الكود المولد
^^^^^^^^^^^^^^^^

نظرًا لأن FX يقوم بتوليد دالة "forward()" في وحدات :class:`GraphModule`، فإن استخدام تقنيات التصحيح التقليدية مثل عبارات "print" أو "pdb"
ليس مباشرًا. لحسن الحظ، هناك عدة تقنيات يمكننا استخدامها
لتصحيح الكود المولد.

.. _استخدام pdb:

استخدام "pdb"
~~~~~~~~~~~~~
استدعاء "pdb" للخطوة إلى البرنامج قيد التشغيل. على الرغم من أن الكود الذي
يمثل الرسم البياني غير موجود في أي ملف مصدر، إلا أنه يمكننا مع ذلك الدخول إليه يدويًا باستخدام "pdb" عندما يتم استدعاء التمرير الأمامي.

::

    import torch
    import torch.fx
    import torchvision.models as models

    def my_pass(inp: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
        graph = tracer_class().trace(inp)
        # منطق التحول هنا
        # <...>

        # إعادة الوحدة النمطية الجديدة
        return fx.GraphModule(inp, graph)

    my_module = models.resnet18()
    my_module_transformed = my_pass(my_module)

    input_value = torch.randn(5, 3, 224, 224)

    # عند تنفيذ هذا السطر في وقت التشغيل، سننتقل إلى موجه "pdb" تفاعلي. يمكننا استخدام الأمر "step" أو "s"
    # للخطوة إلى تنفيذ السطر التالي
    import pdb; pdb.set_trace()

    my_module_transformed(input_value)

.. _طباعة الكود المولد:

طباعة الكود المولد
~~~~~~~~~~~~~
إذا كنت ترغب في تشغيل نفس الكود عدة مرات، فقد يكون الأمر
مرهقًا بعض الشيء للانتقال إلى الكود الصحيح باستخدام "pdb". في هذه الحالة، أحد
النهج هو ببساطة نسخ ولصق تمرير "forward" المولد في كودك وفحصه من هناك.

::

    # نفترض أن "traced" هي وحدة نمطية من النوع "GraphModule" خضعت لبعض
    # التحولات

    # نسخ هذا الكود للاستخدام لاحقًا
    print(traced)
    # طباعة الكود المولد من التتبع الرمزي. هذا الإخراج:
    """
    def forward(self, y):
        x = self.x
        add_1 = x + y;  x = y = None
        return add_1
    """

    # إنشاء فئة فرعية من الوحدة النمطية الأصلية
    class SubclassM(M):
        def __init__(self):
            super().__init__()

        # قم بلصق دالة "forward" المولدة (التي قمنا بطباعتها ونسخها أعلاه) هنا
        def forward(self, y):
            x = self.x
            add_1 = x + y;  x = y = None
            return add_1

    # إنشاء مثيل للوحدة النمطية الأصلية غير المتبعة. ثم، قم بإنشاء مثيل للوحدة النمطية باستخدام دالة "forward" المنسوخة. يمكننا
الآن مقارنة الإخراج لكل من الإصدار الأصلي والإصدار المتبع.
    pre_trace = M()
    post_trace = SubclassM()

.. _استخدام الدالة to_folder من GraphModule:

استخدام الدالة "to_folder" من "GraphModule"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:meth:`GraphModule.to_folder` هي دالة في "GraphModule" تسمح لك بتفريغ كود FX المولد إلى مجلد. على الرغم من أن نسخ تمرير "forward" إلى الكود غالبًا ما يكون كافيًا كما هو موضح في :ref:`طباعة الكود المولد`،
قد يكون من الأسهل فحص الوحدات والمؤشرات باستخدام "to_folder".

::

    m = symbolic_trace(M())
    m.to_folder("foo", "Bar")
    from foo import Bar
    y = Bar()

بعد تشغيل المثال أعلاه، يمكننا بعد ذلك النظر في الكود داخل
"foo/module.py" وتعديله حسب الرغبة (على سبيل المثال، إضافة عبارات "print"
أو استخدام "pdb") لتصحيح الكود المولد.

.. _تصحيح التحول:

تصحيح التحول
^^^^^^^^^^^^^

الآن بعد أن حددنا أن أحد التحولات يقوم بإنشاء كود غير صحيح، فقد حان الوقت لتصحيح التحول نفسه. أولاً، سنتحقق من
قسم :ref:`محدودية التتبع الرمزي` في الوثائق.
بمجرد التحقق من أن التتبع يعمل كما هو متوقع، يصبح الهدف
هو معرفة ما حدث خطأ أثناء تحويل "GraphModule". قد يكون هناك إجابة سريعة في
:ref:`كتابة التحولات`، ولكن إذا لم يكن الأمر كذلك، فهناك عدة طرق
لدراسة الوحدة النمطية المتبعة لدينا:

::

    # وحدة نمطية نموذجية
    class M(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    # إنشاء مثيل للوحدة النمطية "M"
    m = M()

    # تتبع رمزي لمثيل الوحدة النمطية "M" (يعيد وحدة نمطية من النوع "GraphModule"). في
    # هذا المثال، سنناقش فقط كيفية فحص وحدة نمطية من النوع "GraphModule"، لذلك لا نعرض أي تحولات نموذجية للإيجاز.
    traced = symbolic_trace(m)

    # طباعة الكود الذي ينتجه تتبع الوحدة النمطية.
    print(traced)
    # دالة "forward" المولدة هي:
    """
    def forward(self, x, y):
        add = x + y;  x = y = None
        return add
    """

    # طباعة الرسم البياني الداخلي.
    print(traced.graph)
    # يعيد هذا الإخراج ما يلي:
    """
    graph():
        %x : [num_users=1] = placeholder[target=x]
        %y : [num_users=1] = placeholder[target=y]
        %add : [num_users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
        return add
    """

    # طباعة تمثيل جدولي للرسم البياني الداخلي.
    traced.graph.print_tabular()
    # يعطينا هذا ما يلي:
    """
    opcode         name    target                   args    kwargs
    -------------  ------  -----------------------  ------  --------
    placeholder    x       x                        ()      {}
    placeholder    y       y                        ()      {}
    call_function  add     <built-in function add>  (x, y)  {}
    output         output  output                   (add,)  {}
    """

باستخدام وظائف المساعدة أعلاه، يمكننا مقارنة الوحدة النمطية المتبعة لدينا
قبل وبعد تطبيق تحولاتنا. في بعض الأحيان، تكون المقارنة المرئية البسيطة كافية لاكتشاف خطأ. إذا لم يكن من الواضح ما هو الخطأ، فقد يكون أداة تصحيح مثل "pdb" خطوة جيدة التالية.

بناءً على المثال أعلاه، ضع في اعتبارك الكود التالي:

::

    # دالة محددة من قبل المستخدم
    def transform_graph(module: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
        # الحصول على الرسم البياني من الوحدة النمطية المتبعة لدينا
        g = tracer_class().trace(module)

        """
        التحولات على "g" توضع هنا
        """

        return fx.GraphModule(module, g)

    # تحويل الرسم البياني
    transformed = transform_graph(traced)

    # طباعة الكود الجديد بعد تحولاتنا. تحقق مما إذا كان كما هو متوقع
    print(transformed)

باستخدام المثال أعلاه، دعنا نقول أن استدعاء "print(traced)"
أظهر لنا أنه كان هناك خطأ في تحولاتنا. نريد معرفة ما الخطأ باستخدام أداة تصحيح. نبدأ جلسة "pdb". يمكننا معرفة ما يحدث أثناء التحول عن طريق كسر
"transform_graph(traced)"، ثم الضغط على "s" للانتقال إلى استدعاء
"transform_graph(traced)".

قد يكون لدينا حظ جيد أيضًا عن طريق تحرير الدالة "print_tabular" لطباعة
سمات مختلفة لعقد الرسم البياني. (على سبيل المثال، قد نرغب في رؤية
"input_nodes" و"users" للعقدة.)

.. _أدوات التصحيح المتاحة:

أدوات التصحيح المتاحة
^^^^^^^^^^^^^^^^

أداة تصحيح بايثون الأكثر شيوعًا هي
`pdb <https://docs.python.org/3/library/pdb.html>`__. يمكنك بدء
برنامجك في وضع "التصحيح" باستخدام "pdb" عن طريق كتابة
``python -m pdb FILENAME.py`` في سطر الأوامر، حيث "FILENAME"
هو اسم الملف الذي تريد تصحيحه. بعد ذلك، يمكنك استخدام أوامر "pdb"
`أوامر التصحيح
<https://docs.python.org/3/library/pdb.html#debugger-commands>`__
للتنقل خلال برنامجك قيد التشغيل خطوة بخطوة. من الشائع تعيين نقطة توقف (``b LINE-NUMBER``) عند بدء "pdb"، ثم استدعاء "c" لتشغيل البرنامج حتى تلك النقطة. يمنعك هذا من الاضطرار إلى الانتقال عبر كل سطر من التنفيذ (باستخدام "s" أو "n") للوصول إلى الجزء من الكود الذي تريد فحصه. بدلاً من ذلك، يمكنك كتابة
``import pdb; pdb.set_trace()`` قبل السطر الذي تريد التوقف عنده.
إذا أضفت "pdb.set_trace()"، فسيبدأ برنامجك تلقائيًا في وضع "التصحيح" عند تشغيله. (بمعنى آخر، يمكنك ببساطة كتابة
``python FILENAME.py`` في سطر الأوامر بدلاً من
``python -m pdb FILENAME.py``.) بمجرد تشغيل ملفك في
وضع "التصحيح"، يمكنك التنقل خلال الكود وفحص الحالة الداخلية لبرنامجك باستخدام أوامر معينة. هناك العديد من البرامج التعليمية الممتازة حول "pdb" عبر الإنترنت، بما في ذلك البرنامج التعليمي لـ RealPython
`“Python Debugging With Pdb” <https://realpython.com/python-debugging-pdb/>`__.

تأتي بيئات التطوير المتكاملة مثل PyCharm أو VSCode عادةً مع أداة تصحيح مضمنة. في بيئة التطوير المتكاملة الخاصة بك، يمكنك اختيار إما) استخدام "pdb" عن طريق فتح نافذة طرفية في بيئة التطوير المتكاملة الخاصة بك (مثل View → Terminal في VSCode)، أو ب) استخدام أداة التصحيح المدمجة (عادةً ما تكون غلافًا رسوميًا حول "pdb").

.. _محدودية التتبع الرمزي:

محدودية التتبع الرمزي
FX يستخدم نظام **التتبع الرمزي** (المعروف أيضًا باسم `التنفيذ الرمزي <https://en.wikipedia.org/wiki/Symbolic_execution>`__) لالتقاط دلالة البرامج في شكل قابل للتحويل/التحليل.

النظام هو **تتبع** لأنه ينفذ البرنامج (في الواقع :class:`torch.nn.Module` أو دالة) لتسجيل العمليات. وهو رمزي لأنه بدلاً من تدفق البيانات الحقيقية خلال هذا التنفيذ، يتم استخدام الرموز (:class:`Proxy` في لغة FX).

على الرغم من أن التتبع الرمزي يعمل لمعظم أكواد الشبكات العصبية، إلا أن لديه بعض القيود.

التحكم الديناميكي في التدفق
^^^^^^^^^^^^^^^^^^^^

القيود الرئيسية للتتبع الرمزي هي أنه لا يدعم حاليًا التحكم الديناميكي في التدفق. أي الحلقات أو عبارات "if" حيث قد يعتمد الشرط على قيم الإدخال للبرنامج.

على سبيل المثال، دعنا نفحص البرنامج التالي:

::

    def func_to_trace(x):
        if x.sum() > 0:
            return torch.relu(x)
        else:
            return torch.neg(x)

    traced = torch.fx.symbolic_trace(func_to_trace)
    """
      <...>
      File "dyn.py", line 6, in func_to_trace
        if x.sum() > 0:
      File "pytorch/torch/fx/proxy.py", line 155, in __bool__
        return self.tracer.to_bool(self)
      File "pytorch/torch/fx/proxy.py", line 85, in to_bool
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
    torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
    """

يعتمد شرط عبارة "if" على قيمة ``x.sum()``، والتي تعتمد على قيمة ``x``، وهو إدخال الدالة. نظرًا لأنه يمكن تغيير ``x`` (أي عند تمرير تنسيق إدخال جديد إلى الدالة المتبعة)، فإن هذا يمثل تحكمًا ديناميكيًا في التدفق. ويتم تتبع المسار للوصول إلى المكان الذي يحدث فيه هذا الموقف.

التحكم الثابت في التدفق
~~~~~~~~~~~~~~~~

من ناحية أخرى، ما يسمى بالتحكم الثابت في التدفق مدعوم. التحكم الثابت في التدفق هو الحلقات أو عبارات "if" التي لا يمكن أن تتغير قيمتها عبر الاستدعاءات. عادةً، في برامج PyTorch، ينشأ التحكم في التدفق هذا عن كود يتخذ قرارات بشأن بنية النموذج بناءً على فرط المعلمات. كمثال ملموس:

::

    import torch
    import torch.fx

    class MyModule(torch.nn.Module):
        def __init__(self, do_activation : bool = False):
            super().__init__()
            self.do_activation = do_activation
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            x = self.linear(x)
            # This if-statement is so-called static control flow.
            # Its condition does not depend on any input values
            if self.do_activation:
                x = torch.relu(x)
            return x

    without_activation = MyModule(do_activation=False)
    with_activation = MyModule(do_activation=True)

    traced_without_activation = torch.fx.symbolic_trace(without_activation)
    print(traced_without_activation.code)
    """
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        return linear_1
    """

    traced_with_activation = torch.fx.symbolic_trace(with_activation)
    print(traced_with_activation.code)
    """
    import torch
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        relu_1 = torch.relu(linear_1);  linear_1 = None
        return relu_1
    """

عبارة "if" ``if self.do_activation`` لا تعتمد على أي إدخالات للدالة، وبالتالي فهي ثابتة. يمكن اعتبار ``do_activation`` فرط معلمة، وتختلف آثار مثيلات مختلفة من ``MyModule`` مع قيم مختلفة لهذا المعلمة في الكود. هذا النمط صالح ومدعوم من قبل التتبع الرمزي.

تعد العديد من حالات التحكم الديناميكي في التدفق تحكمًا ثابتًا في التدفق من الناحية الدلالية. يمكن جعل هذه الحالات تدعم التتبع الرمزي عن طريق إزالة تبعيات البيانات على قيم الإدخال، على سبيل المثال عن طريق نقل القيم إلى سمات ``Module`` أو عن طريق ربط قيم ملموسة بالحجج أثناء التتبع الرمزي:

::

        def f(x, flag):
            if flag: return x
            else: return x*2

        fx.symbolic_trace(f) # Fails!

        fx.symbolic_trace(f, concrete_args={'flag': True})

في حالة التحكم الديناميكي حقًا في التدفق، يمكن تتبع أجزاء البرنامج التي تحتوي على هذا الكود كمكالمات للأسلوب (راجع :ref: `Customizing Tracing`) أو الدالة (راجع :func: `wrap`) بدلاً من التتبع خلالها.

وظائف غير "التورتش"
^^^^^^^^^^^^^^^^^^^^^

يستخدم FX ``__torch_function__`` كآلية لاعتراض المكالمات (راجع `نظرة عامة فنية <https://github.com/pytorch/pytorch/blob/master/torch/fx/OVERVIEW.md#technical-details>`__ للحصول على مزيد من المعلومات حول هذا الموضوع). بعض الوظائف، مثل وظائف Python المدمجة أو تلك الموجودة في وحدة "الرياضيات"، لا يغطيها ``__torch_function__``، ولكننا نريد تسجيلها في التتبع الرمزي. على سبيل المثال:

::

    import torch
    import torch.fx
    from math import sqrt

    def normalize(x):
        """
        Normalize `x` by the size of the batch dimension
        """
        return x / sqrt(len(x))

    # It's valid Python code
    normalize(torch.rand(3, 4))

    traced = torch.fx.symbolic_trace(normalize)
    """
      <...>
      File "sqrt.py", line 9, in normalize
        return x / sqrt(len(x))
      File "pytorch/torch/fx/proxy.py", line 161, in __len__
        raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
    RuntimeError: 'len' is not supported in symbolic tracing by default. If you want this call to be recorded, please call torch.fx.wrap('len') at module scope
    """

يخبرنا الخطأ بأن دالة Python المدمجة ``len`` غير مدعومة. يمكننا جعل الدوال مثل هذه مسجلة في الأثر كاستدعاءات مباشرة باستخدام واجهة برمجة التطبيقات :func: `wrap`:

::

    torch.fx.wrap('len')
    torch.fx.wrap('sqrt')

    traced = torch.fx.symbolic_trace(normalize)

    print(traced.code)
    """
    import math
    def forward(self, x):
        len_1 = len(x)
        sqrt_1 = math.sqrt(len_1);  len_1 = None
        truediv = x / sqrt_1;  x = sqrt_1 = None
        return truediv
    """

.. _Customizing Tracing:

تخصيص التتبع باستخدام فئة "Tracer"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

فئة :class: `Tracer` هي الفئة التي تقوم عليها تنفيذ ``symbolic_trace``. يمكن تخصيص سلوك التتبع عن طريق إنشاء فئة فرعية من Tracer، مثل:

::

    class MyCustomTracer(torch.fx.Tracer):
        # Inside here you can override various methods
        # to customize tracing. See the `Tracer` API
        # reference
        pass


    # Let's use this custom tracer to trace through this module
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) + torch.ones(3, 4)

    mod = MyModule()

    traced_graph = MyCustomTracer().trace(mod)
    # trace() returns a Graph. Let's wrap it up in a
    # GraphModule to make it runnable
    traced = torch.fx.GraphModule(mod, traced_graph)

الوحدات النمطية الورقية
~~~~~~~~~~~~~~~~

الوحدات النمطية الورقية هي الوحدات النمطية التي تظهر كمكالمات في الأثر الرمزي بدلاً من التتبع خلالها. مجموعة الوحدات النمطية الورقية الافتراضية هي مجموعة الوحدات النمطية القياسية لـ ``torch.nn``. على سبيل المثال:

::

    class MySpecialSubmodule(torch.nn.Module):
        def forward(self, x):
            return torch.neg(x)

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)
            self.submod = MySpecialSubmodule()

        def forward(self, x):
            return self.submod(self.linear(x))

    traced = torch.fx.symbolic_trace(MyModule())
    print(traced.code)
    # `linear` is preserved as a call, yet `submod` is traced though.
    # This is because the default set of "Leaf Modules" includes all
    # standard `torch.nn` modules.
    """
    import torch
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        neg_1 = torch.neg(linear_1);  linear_1 = None
        return neg_1
    """

يمكن تخصيص مجموعة الوحدات النمطية الورقية عن طريق تجاوز :meth: `Tracer.is_leaf_module`.

متفرقات
^^^^^^^^

-  لا يمكن تتبع منشئي المصفوفة (على سبيل المثال ``torch.zeros``، ``torch.ones``، ``torch.rand``، ``torch.randn``، ``torch.sparse_coo_tensor``) حاليًا.

   -  يمكن استخدام منشئي المصفوفة المحددين (``zeros``، ``ones``) والقيمة التي ينتجونها ستكون مضمنة في الأثر كقيمة ثابتة. هذا لا يمثل مشكلة إلا إذا كانت الحجج لهؤلاء المنشئين تشير إلى أحجام الإدخال الديناميكية. في هذه الحالة، قد يكون ``ones_like`` أو ``zeros_like`` بديلاً صالحًا.
   -  سيتم تضمين قيمة عشوائية واحدة في الأثر لمنشئي المصفوفة غير المحددين (``rand``، ``randn``). من غير المرجح أن يكون هذا هو السلوك المقصود. تتمثل إحدى طرق العمل حول ذلك في لف ``torch.randn`` في دالة ``torch.fx.wrap`` واستدعاء تلك الدالة بدلاً من ذلك.

    ::

        @torch.fx.wrap
        def torch_randn(x, shape):
            return torch.randn(shape)

        def f(x):
            return x + torch_randn(x, 5)
        fx.symbolic_trace(f)

   -  قد يتم إصلاح هذا السلوك في إصدار مستقبلي.

-  التعليقات التوضيحية للنوع

   -  التعليقات التوضيحية للنوع على النمط Python 3 (على سبيل المثال
      ``func(x : torch.Tensor، y : int) -> torch.Tensor``) مدعومة وسيتم الحفاظ عليها من خلال التتبع الرمزي.
   -  التعليقات التوضيحية للنوع على النمط Python 2 ``# type: (torch.Tensor، int) -> torch.Tensor`` غير مدعومة حاليًا.
   -  التعليقات التوضيحية للنوع على الأسماء المحلية داخل دالة غير مدعومة حاليًا.


-  مشكلة حول علم "التدريب" والوحدات النمطية الفرعية

   -  عند استخدام الدوال الوظيفية مثل ``torch.nn.functional.dropout``، سيكون من الشائع تمرير علم "التدريب" كـ ``self.training``. أثناء التتبع في FX، من المحتمل أن يتم تضمين هذه القيمة كقيمة ثابتة.

    ::

        import torch
        import torch.fx

        class DropoutRepro(torch.nn.Module):
          def forward(self, x):
            return torch.nn.functional.dropout(x, training=self.training)


        traced = torch.fx.symbolic_trace(DropoutRepro())
        print(traced.code)
        """
        def forward(self, x):
          dropout = torch.nn.functional.dropout(x, p = 0.5, training = True, inplace = False);  x = None
          return dropout
        """

        traced.eval()

        x = torch.randn(5, 3)
        torch.testing.assert_close(traced(x), x)
        """
        AssertionError: Tensor-likes are not close!

        Mismatched elements: 15 / 15 (100.0%)
        Greatest absolute difference: 1.6207983493804932 at index (0, 2) (up to 1e-05 allowed)
        Greatest relative difference: 1.0 at index (0, 0) (up to 0.0001 allowed)
        """

   - ومع ذلك، عندما يتم استخدام الوحدة النمطية الفرعية القياسية ``nn.Dropout()``، يتم تضمين علم "التدريب" ويمكن تغييره - بسبب الحفاظ على نموذج كائن ``nn.Module``.

    ::

        class DropoutRepro2(torch.nn.Module):
          def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout()

          def forward(self, x):
            return self.drop(x)

        traced = torch.fx.symbolic_trace(DropoutRepro2())
        print(traced.code)
        """
        def forward(self, x):
          drop = self.drop(x);  x = None
          return drop
        """

        traced.eval()

        x = torch.randn(5, 3)
        torch.testing.assert_close(traced(x), x)

  - بسبب هذا الاختلاف، ضع في اعتبارك وضع علامة على الوحدات النمطية التي تتفاعل مع علم "التدريب" ديناميكيًا كوحدات نمطية ورقية.


مرجع واجهة برمجة التطبيقات
.. autofunction:: torch.fx.symbolic_trace

.. autofunction:: torch.fx.wrap

.. autoclass:: torch.fx.GraphModule
  :members:

  .. automethod:: __init__

.. autoclass:: torch.fx.Graph بالعربية
  :members:

  .. automethod:: __init__

.. autoclass:: torch.fx.Node بالعربية
  :members:

.. autoclass:: torch.fx.Tracer بالعربية
  :members:
  :inherited-members:

.. autoclass:: torch.fx.Proxy بالعربية

.. autoclass:: torch.fx.Interpreter بالعربية
  :members:

.. autoclass:: torch.fx.Transformer بالعربية
  :members:

.. autofunction:: torch.fx.replace_pattern بالعربية


.. The experimental and passes submodules are missing docs.
.. Adding it here for coverage but this doesn't add anything to the
.. rendered doc.
.. py:module:: torch.fx.passes
.. py:module:: torch.fx.passes.infra
.. py:module:: torch.fx.passes.backends
.. py:module:: torch.fx.passes.utils
.. py:module:: torch.fx.passes.tests
.. py:module:: torch.fx.experimental
.. py:module:: torch.fx.experimental.unification
.. py:module:: torch.fx.experimental.unification.multipledispatch
.. py:module:: torch.fx.experimental.migrate_gradual_types
.. py:module:: torch.fx.passes.dialect
.. py:module:: torch.fx.passes.dialect.common
.. py:module:: torch.fx.annotate
.. py:module:: torch.fx.config
.. py:module:: torch.fx.experimental.accelerator_partitioner
.. py:module:: torch.fx.experimental.const_fold
.. py:module:: torch.fx.experimental.debug
.. py:module:: torch.fx.experimental.graph_gradual_typechecker
.. py:module:: torch.fx.experimental.merge_matmul
.. py:module:: torch.fx.experimental.meta_tracer
.. py:module:: torch.fx.experimental.migrate_gradual_types.constraint
.. py:module:: torch.fx.experimental.migrate_gradual_types.constraint_generator
.. py:module:: torch.fx.experimental.migrate_gradual_types.constraint_transformation
.. py:module:: torch.fx.experimental.migrate_gradual_types.operation
.. py:module:: torch.fx.experimental.migrate_gradual_types.transform_to_z3
.. py:module:: torch.fx.experimental.migrate_gradual_types.util
.. py:module:: torch.fx.experimental.migrate_gradual_types.z3_types
.. py:module:: torch.fx.experimental.normalize
.. py:module:: torch.fx.experimental.optimization
.. py:module:: torch.fx.experimental.partitioner_utils
.. py:module:: torch.fx.experimental.recording
.. py:module:: torch.fx.experimental.refinement_types
.. py:module:: torch.fx.experimental.rewriter
.. py:module:: torch.fx.experimental.schema_type_annotation
.. py:module:: torch.fx.experimental.sym_node
.. py:module:: torch.fx.experimental.unification.core
.. py:module:: torch.fx.experimental.unification.dispatch
.. py:module:: torch.fx.experimental.unification.match
.. py:module:: torch.fx.experimental.unification.more
.. py:module:: torch.fx.experimental.unification.multipledispatch.conflict
.. py:module:: torch.fx.experimental.unification.multipledispatch.core
.. py:module:: torch.fx.experimental.unification.multipledispatch.dispatcher
.. py:module:: torch.fx.experimental.unification.multipledispatch.utils
.. py:module:: torch.fx.experimental.unification.multipledispatch.variadic
.. py:module:: torch.fx.experimental.unification.unification_tools
.. py:module:: torch.fx.experimental.unification.utils
.. py:module:: torch.fx.experimental.unification.variable
.. py:module:: torch.fx.experimental.unify_refinements
.. py:module:: torch.fx.experimental.validator
.. py:module:: torch.fx.graph
.. py:module:: torch.fx.graph_module
.. py:module:: torch.fx.immutable_collections
.. py:module:: torch.fx.interpreter
.. py:module:: torch.fx.node
.. py:module:: torch.fx.operator_schemas
.. py:module:: torch.fx.passes.annotate_getitem_nodes
.. py:module:: torch.fx.passes.backends.cudagraphs
.. py:module:: torch.fx.passes.dialect.common.cse_pass
.. py:module:: torch.fx.passes.fake_tensor_prop
.. py:module:: torch.fx.passes.graph_drawer
.. py:module:: torch.fx.passes.graph_manipulation
.. py:module:: torchMultiplier
.. py:module:: torch.fx.passes.graph_transform_observer
.. py:module:: torch.fx.passes.infra.partitioner
.. py:module:: torch.fx.passes.infra.pass_base
.. py:module:: torch.fx.passes.infra.pass_manager
.. py:module:: torch.fx.passes.net_min_base
.. py:module:: torch.fx.passes.operator_support
.. py:module:: torch.fx.passes.param_fetch
.. py:module:: torch.fx.passes.pass_manager
.. py:module:: torch.fx.passes.reinplace
.. py:module:: torch.fx.passes.runtime_assert
.. py:module:: torch.fx.passes.shape_prop
.. py:module:: torch.fx.passes.split_module
.. py:module:: torch.fx.passes.split_utils
.. py:module:: torch.fx.passes.splitter_base
.. py:module:: torch.fx.passes.tests.test_pass_manager
.. py:module:: torch.fx.passes.tools_common
.. py:module:: torch.fx.passes.utils.common
.. py:module:: torch.fx.passes.utils.fuser_utils
.. py:module:: torch.fx.passes.utils.matcher_utils
.. py:module:: torch.fx.passes.utils.matcher_with_name_node_map_utils
.. py:module:: torch.fx.passes.utils.source_matcher_utils
.. py:module:: torch.fx.proxy
.. py:module:: torch.fx.subgraph_rewriter
.. py:module:: torch.fx.tensor_type
.. py:module:: torch.fx.traceback