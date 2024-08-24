.. contents::
    :local:
    :depth: 2

.. _torchScript-language-reference:

مرجع لغة تورتشسكريبت TorchScript Language Reference
===============================================

TorchScript هي مجموعة فرعية ثابتة من Python يمكن كتابتها مباشرة (باستخدام
مُزيّن :func:`@torch.jit.script <torch.jit.script>`) أو إنشاؤها تلقائيًا من كود Python عبر
التتبع. عند استخدام التتبع، يتم تحويل الكود تلقائيًا إلى هذه المجموعة الفرعية من
Python عن طريق تسجيل مشغلي التنسور الفعليين فقط وببساطة تنفيذ وحذف
كود Python المحيط الآخر.

عند كتابة TorchScript مباشرة باستخدام مُزيّن ``@torch.jit.script``، يجب
على المبرمج أن يستخدم فقط المجموعة الفرعية من Python المدعومة في TorchScript. يوثق هذا القسم
ما هو مدعوم في TorchScript كما لو كان مرجعًا للغة لِلُغة قائمة بذاتها. أي ميزات لـ Python غير مذكورة في هذا المرجع ليست
جزءًا من TorchScript. راجع `الوظائف المضمنة` للحصول على مرجع كامل لأساليب التنسور PyTorch المتاحة، والوحدات، والوظائف.

باعتبارها مجموعة فرعية من Python، فإن أي دالة TorchScript صالحة هي أيضًا دالة Python
صالحة. هذا يجعل من الممكن `تعطيل TorchScript` والتنقيح
الدالة باستخدام أدوات Python القياسية مثل ``pdb``. العكس غير صحيح: هناك
العديد من برامج Python الصحيحة التي ليست برامج TorchScript صحيحة.
بدلاً من ذلك، يركز TorchScript بشكل محدد على ميزات Python اللازمة
لتمثيل نماذج الشبكات العصبية في PyTorch.

.. _types:
.. _supported type:

الأنواع
~~~~~

أكبر اختلاف بين TorchScript ولغة Python الكاملة هو أن
TorchScript يدعم فقط مجموعة صغيرة من الأنواع اللازمة للتعبير عن الشبكات العصبية
النماذج. على وجه الخصوص، يدعم TorchScript ما يلي:

.. csv-table::
   :header: "النوع", "الوصف"

   "``Tensor``", "تنسور PyTorch بأي نوع بيانات، أو بُعد، أو خلفية"
   "``Tuple[T0, T1, ..., TN]``", "مجموعة تحتوي على الأنواع الفرعية ``T0``، ``T1``، إلخ (على سبيل المثال ``Tuple[Tensor، Tensor]``)"
   "``bool``", "قيمة منطقية"
   "``int``", "عدد صحيح قياسي"
   "``float``", "رقم نقطة عائمة قياسي"
   "``str``", "سلسلة"
   "``List[T]``", "قائمة يكون جميع أعضائها من النوع ``T``"
   "``Optional[T]``", "قيمة تكون إما None أو من النوع ``T``"
   "``Dict[K، V]``"، "قاموس بنوع مفتاح ``K`` ونوع قيمة ``V``. يُسمح فقط بـ ``str``، ``int``، و ``float`` كأنواع مفاتيح."
   "``T``"، " `فئة TorchScript`_"
   "``E``"، " `مُعدَّد TorchScript`_"
   "``NamedTuple[T0, T1, ...]``"، "نوع مجموعة :func:`collections.namedtuple <collections.namedtuple>`"
   "``Union[T0, T1, ...]``"، "واحد من الأنواع الفرعية ``T0``، ``T1``، إلخ."

على عكس Python، يجب أن يكون لكل متغير في دالة TorchScript نوع ثابت واحد.
هذا يجعل من السهل تحسين وظائف TorchScript.

مثال (عدم تطابق النوع)

.. testcode::

    import torch

    @torch.jit.script
    def an_error(x):
        if x:
            r = torch.rand(1)
        else:
            r = 4
        return r


.. testoutput::

     Traceback (most recent call last):
       ...
     RuntimeError: ...

     Type mismatch: r is set to type Tensor in the true branch and type int in the false branch:
     @torch.jit.script
     def an_error(x):
         if x:
         ~~~~~
             r = torch.rand(1)
             ~~~~~~~~~~~~~~~~~
         else:
         ~~~~~
             r = 4
             ~~~~~ <--- HERE
         return r
     and was used here:
         else:
             r = 4
         return r
                ~ <--- HERE...

بنيات الأنواع غير المدعومة
^^^^^^^^^^^^^^^^^^^
لا يدعم TorchScript جميع الميزات والأنواع في وحدة :mod:`typing`. بعض هذه
الأشياء الأساسية التي من غير المرجح أن تضاف في المستقبل في حين أن البعض الآخر
قد يتم إضافتها إذا كان هناك طلب كافٍ من المستخدمين لجعلها أولوية.

هذه الأنواع والميزات من وحدة :mod:`typing` غير متوفرة في TorchScript.

.. csv-table::
   :header: "البند", "الوصف"

   ":any:`typing.Any`", ":any:`typing.Any` قيد التطوير حاليًا ولكن لم يتم إصداره بعد"
   ":any:`typing.NoReturn`", "غير منفذ"
   ":any:`typing.Sequence`", "غير منفذ"
   ":any:`typing.Callable`", "غير منفذ"
   ":any:`typing.Literal`", "غير منفذ"
   ":any:`typing.ClassVar`", "غير منفذ"
   ":any:`typing.Final`", "هذا مدعوم لـ :any:`module attributes <Module Attributes>` ملاحظات النوع فئة ولكن ليس للوظائف"
   ":any:`typing.AnyStr`", "لا يدعم TorchScript :any:`bytes` لذلك لا يستخدم هذا النوع"
   ":any:`typing.overload`", ":any:`typing.overload` قيد التطوير حاليًا ولكن لم يتم إصداره بعد"
   "أسماء الأنواع"، "غير منفذ"
   "التفرع الاسمي مقابل التفرع الهيكلي"، "التفرع الاسمي قيد التطوير، ولكن التفرع الهيكلي غير مدعوم"
   "NewType"، "من غير المرجح أن يتم تنفيذه"
   "Generics"، "من غير المرجح أن يتم تنفيذه"

أي وظائف أخرى من وحدة :any:`typing` غير مدرجة صراحة في هذه الوثيقة غير مدعومة.

الأنواع الافتراضية
^^^^^^^^^^^^^

بشكل افتراضي، يفترض أن جميع المعلمات لدالة TorchScript هي Tensor.
لتحديد أن حجة لدالة TorchScript هي من نوع آخر، من الممكن استخدام
ملاحظات النوع على طريقة MyPy باستخدام الأنواع المذكورة أعلاه.

.. testcode::

    import torch

    @torch.jit.script
    def foo(x، tup):
        # type: (int, Tuple[Tensor, Tensor]) -> Tensor
        t0، t1 = tup
        return t0 + t1 + x

    print(foo(3, (torch.rand(3)، torch.rand(3))))

.. testoutput::
    :hide:

    ...

.. note::
  من الممكن أيضًا استخدام ملاحظات النوع باستخدام تلميحات النوع Python 3 من
  وحدة ``typing``.

  .. testcode::

    import torch
    from typing import Tuple

    @torch.jit.script
    def foo(x: int, tup: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t0, t1 = tup
        return t0 + t1 + x

    print(foo(3, (torch.rand(3), torch.rand(3))))

  .. testoutput::
    :hide:

    ...


يفترض أن تكون القائمة الفارغة ``List[Tensor]`` والقواميس الفارغة
``Dict[str, Tensor]``. لإنشاء قائمة أو قاموس فارغ من الأنواع الأخرى،
استخدم `تلميحات النوع Python 3`.

مثال (ملاحظات النوع لـ Python 3):

.. testcode::

    import torch
    import torch.nn as nn
    from typing import Dict, List, Tuple

    class EmptyDataStructures(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[int, float]], Dict[str, int]]:
            # This annotates the list to be a `List[Tuple[int, float]]`
            my_list: List[Tuple[int, float]] = []
            for i in range(10):
                my_list.append((i, x.item()))

            my_dict: Dict[str, int] = {}
            return my_list, my_dict

    x = torch.jit.script(EmptyDataStructures())




تحسين النوع الاختياري
^^^^^^^^^^^^^^^^^

سيحسن TorchScript نوع متغير من النوع ``Optional[T]`` عند
إجراء مقارنة مع ``None`` داخل الشرط الشرطي لـ if-statement أو تم التحقق منه في ``assert``.
يمكن لمترجم البرنامج أن يستنتج عمليات فحص "None" المتعددة التي يتم دمجها مع
``and``، ``or``، و ``not``. سيحدث التحسين أيضًا بالنسبة إلى كتل else لبيانات if-statements
التي لم يتم كتابتها بشكل صريح.

يجب أن يكون فحص "None" داخل شرط if-statement؛ تعيين
فحص "None" إلى متغير واستخدامه في شرط if-statement لن
تحسين أنواع المتغيرات في الفحص.
سيتم تحسين المتغيرات المحلية فقط، ولن يتم تحسين سمة مثل ``self.x`` ويجب تعيينها إلى
متغير محلي ليتم تحسينه.


مثال (تحسين الأنواع على المعلمات والمتغيرات المحلية):

.. testcode::

    import torch
    import torch.nn as nn
    from typing import Optional

    class M(nn.Module):
        z: Optional[int]

        def __init__(self, z):
            super().__init__()
            # If `z` is None، its type cannot be inferred، so it must
            # be specified (above)
            self.z = z

        def forward(self, x, y, z):
            # type: (Optional[int], Optional[int], Optional[int]) -> int
            if x is None:
                x = 1
                x = x + 1

            # Refinement for an attribute by assigning it to a local
            z = self.z
            if y is not None and z is not None:
                x = y + z

            # Refinement via an `assert`
            assert z is not None
            x += z
            return x

    module = torch.jit.script(M(2))
    module = torch.jit.script(M(None))


.. _TorchScript Class:
.. _TorchScript Classes:
.. _torchscript-classes:

فئات TorchScript
^^^^^^^^^^^^^^^^^^^

.. warning::

    دعم فئة TorchScript تجريبي. حاليًا، فهو مناسب بشكل أفضل
    لأنواع السجلات البسيطة (تخيل ``NamedTuple`` مع أساليب
    مرفقة).

يمكن استخدام الفئات Python في TorchScript إذا تم وضع علامة عليها باستخدام :func:`@torch.jit.script <torch.jit.script>`،
مشابه لكيفية إعلان دالة TorchScript:

.. testcode::
    :skipif: True  # TODO: fix the source file resolving so this can be tested

    @torch.jit.script
    class Foo:
      def __init__(self, x, y):
        self.x = x

      def aug_add_x(self, inc):
        self.x += inc


هذه المجموعة الفرعية مقيدة:

* يجب أن تكون جميع الوظائف وظائف TorchScript صالحة (بما في ذلك ``__init__()``).
* يجب أن تكون الفئات فئات جديدة، حيث نستخدم ``__new__()`` لبنائها باستخدام pybind11.
* فئات TorchScript ثابتة النوع. يمكن الإعلان عن الأعضاء فقط عن طريق تعيين
  self في طريقة ``__init__()``.

    على سبيل المثال، تعيين ``self`` خارج طريقة ``__init__()``: ::

        @torch.jit.script
        class Foo:
          def assign_x(self):
            self.x = torch.rand(2, 3)

    ستؤدي إلى: ::

        RuntimeError:
        Tried to set nonexistent attribute: x. Did you forget to initialize it in __init__()?:
        def assign_x(self):
          self.x = torch.rand(2, 3)
          ~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

* لا يُسمح بأي تعبيرات باستثناء تعريفات الأساليب في جسم الفئة.
* لا يوجد دعم للوراثة أو أي استراتيجية تعدد الأشكال الأخرى، باستثناء الوراثة
  من ``object`` لتحديد فئة جديدة.

بعد تعريف الفئة، يمكن استخدامها في كل من TorchScript وPython بشكل متناوب
مثل أي نوع TorchScript آخر:

::

    # Declare a TorchScript class
    @torch.jit.script
    class Pair:
      def __init__(self, first, second):
        self.first = first
        self.second = second

    @torch.jit.script
    def sum_pair(p):
      # type: (Pair) -> Tensor
      return p.first + p.second

    p = Pair(torch.rand(2, 3), torch.rand(2, 3))
    print(sum_pair(p))


.. _TorchScript Enum:
.. _TorchScript Enums:
.. _torchscript-enums:

مُعدَّدات TorchScript
^^^^^^^^^^^^^^^^^^^

يمكن استخدام المُعدَّدات Python في TorchScript دون أي ملاحظات أو رموز إضافية:
::

    from enum import Enum

    class Color(Enum):
        RED = 1
        GREEN = 2

    @torch.jit.script
    def enum_fn(x: Color, y: Color) -> bool:
        if x == Color.RED:
            return True
        return x == y

بعد تعريف Enum، يمكن استخدامه في كل من TorchScript وPython بشكل متبادل
مثل أي نوع آخر من TorchScript. يجب أن يكون نوع قيم Enum إما "int"
أو "float"، أو "str". يجب أن تكون جميع القيم من نفس النوع؛ لا تُدعم الأنواع المتغايرة للقيم
Enum.

Named Tuples
^^^^^^^^^^^^
يمكن استخدام الأنواع التي ينتجها :func:`collections.namedtuple <collections.namedtuple>` في TorchScript.

.. testcode::

    import torch
    import collections

    Point = collections.namedtuple('Point', ['x', 'y'])

    @torch.jit.script
    def total(point):
        # type: (Point) -> Tensor
        return point.x + point.y

    p = Point(x=torch.rand(3), y=torch.rand(3))
    print(total(p))

.. testoutput::
    :hide:

    ...

.. _jit_iterables:

Iterables
^^^^^^^^^

يمكن لبعض الوظائف (على سبيل المثال، :any:`zip` و:any:`enumerate`) أن تعمل فقط على الأنواع القابلة للتحديد.
تشمل الأنواع القابلة للتحديد في TorchScript كلًا من "Tensor"s، والقوائم، والمجاميع، والقواميس، والسلاسل،
:any:`torch.nn.ModuleList` و:any:`torch.nn.ModuleDict`.

Expressions
~~~~~~~~~~~

يتم دعم تعبيرات Python التالية.

Literals
^^^^^^^^
::

    True
    False
    None
    'string literals'
    "string literals"
    3  # interpreted as int
    3.4  # interpreted as a float

List Construction
"""""""""""""""""
يُفترض أن تكون القائمة الفارغة من النوع ``List[Tensor]``.
تُستمد أنواع قوائم الأحرف الأخرى من نوع الأعضاء.
راجع `Default Types`_ لمزيد من التفاصيل.

::

    [3, 4]
    []
    [torch.rand(3), torch.rand(4)]

Tuple Construction
""""""""""""""""""
::

    (3, 4)
    (3,)

Dict Construction
"""""""""""""""""
يُفترض أن يكون القاموس الفارغ من النوع ``Dict[str, Tensor]``.
تُستمد أنواع القواميس الأخرى من نوع الأعضاء.
راجع `Default Types`_ لمزيد من التفاصيل.

::

    {'hello': 3}
    {}
    {'a': torch.rand(3), 'b': torch.rand(4)}

Variables
^^^^^^^^^
راجع `Variable Resolution`_ لمعرفة كيفية حل المتغيرات.

::

    my_variable_name

Arithmetic Operators
^^^^^^^^^^^^^^^^^^^^
::

    a + b
    a - b
    a * b
    a / b
    a ^ b
    a @ b

Comparison Operators
^^^^^^^^^^^^^^^^^^^^
::

    a == b
    a != b
    a < b
    a > b
    a <= b
    a >= b

Logical Operators
^^^^^^^^^^^^^^^^^
::

    a and b
    a or b
    not b

Subscripts and Slicing
^^^^^^^^^^^^^^^^^^^^^^
::

    t[0]
    t[-1]
    t[0:2]
    t[1:]
    t[:1]
    t[:]
    t[0, 1]
    t[0, 1:2]
    t[0, :1]
    t[-1, 1:, 0]
    t[1:, -1, 0]
    t[i:j, i]

Function Calls
^^^^^^^^^^^^^^
مكالمات `builtin functions`

::

    torch.rand(3, dtype=torch.int)

مكالمات لوظائف النص البرمجي الأخرى:

.. testcode::

    import torch

    @torch.jit.script
    def foo(x):
        return x + 1

    @torch.jit.script
    def bar(x):
        return foo(x)

Method Calls
^^^^^^^^^^^^
مكالمات لطرق الأنواع المضمنة مثل tensor: ``x.mm(y)``

بالنسبة إلى الوحدات النمطية، يجب تجميع الطرق قبل إمكانية استدعائها. يقوم مترجم TorchScript بتجميع الطرق بشكل متكرر
يراها عند تجميع طرق أخرى. بشكل افتراضي، يبدأ التجميع في طريقة "forward". يتم تجميع أي طرق تستدعيها "forward"
وسيتم تجميعها، وأي طرق تستدعيها تلك الطرق، وهكذا. لبدء التجميع في طريقة أخرى غير "forward"، استخدم الديكور
:func:`@torch.jit.export <torch.jit.export>` (يتم وضع علامة "forward" بشكل ضمني على أنها ``@torch.jit.export``).

إن استدعاء وحدة فرعية مباشرة (على سبيل المثال، ``self.resnet(input)``) يعادل
استدعاء طريقة "forward" الخاصة بها (على سبيل المثال، ``self.resnet.forward(input)``).

.. testcode::
    :skipif: torchvision is None

    import torch
    import torch.nn as nn
    import torchvision

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            means = torch.tensor([103.939, 116.779, 123.68])
            self.means = torch.nn.Parameter(means.resize_(1, 3, 1, 1))
            resnet = torchvision.models.resnet18()
            self.resnet = torch.jit.trace(resnet, torch.rand(1, 3, 224, 224))

        def helper(self, input):
            return self.resnet(input - self.means)

        def forward(self, input):
            return self.helper(input)

        # Since nothing in the model calls `top_level_method`, the compiler
        # must be explicitly told to compile this method
        @torch.jit.export
        def top_level_method(self, input):
            return self.other_helper(input)

        def other_helper(self, input):
            return input + 10

    # `my_script_module` will have the compiled methods `forward`, `helper`,
    # `top_level_method`, and `other_helper`
    my_script_module = torch.jit.script(MyModule())

Ternary Expressions
^^^^^^^^^^^^^^^^^^^
::

    x if x > y else y

Casts
^^^^^
::

    float(ten)
    int(3.5)
    bool(ten)
    str(2)``

Accessing Module Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    self.my_parameter
    self.my_submodule.my_parameter

Statements
~~~~~~~~~~

يدعم TorchScript أنواع البيانات التالية:

Simple Assignments
^^^^^^^^^^^^^^^^^^
::

    a = b
    a += b # short-hand for a = a + b, does not operate in-place on a
    a -= b

Pattern Matching Assignments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    a, b = tuple_or_list
    a, b, *c = a_tuple

Multiple Assignments
::

    a = b, c = tup

Print Statements
^^^^^^^^^^^^^^^^
::

    print("the result of an add:", a + b)

If Statements
^^^^^^^^^^^^^
::

    if a < 4:
        r = -a
    elif a < 3:
        r = a + a
    else:
        r = 3 * a

بالإضافة إلى bools، يمكن استخدام floats وints وTensors في شرط
وسيتم تحويلها ضمنيًا إلى قيمة منطقية.

While Loops
^^^^^^^^^^^
::

    a = 0
    while a < 4:
        print(a)
        a += 1

For loops with range
^^^^^^^^^^^^^^^^^^^^
::

    x = 0
    for i in range(10):
        x *= i

For loops over tuples
^^^^^^^^^^^^^^^^^^^^^
يتم فك حلقات التكرار هذه، وتوليد جسم لكل عضو في المصفوفة. يجب أن يتحقق الجسم من النوع بشكل صحيح لكل عضو.

::

    tup = (3, torch.rand(4))
    for x in tup:
        print(x)

For loops over constant nn.ModuleList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

لاستخدام ``nn.ModuleList`` داخل طريقة مجمعة، يجب وضع علامة عليه
ثابت عن طريق إضافة اسم السمة إلى قائمة ``__constants__``
لنوع البيانات. يتم فك حلقات التكرار على ``nn.ModuleList`` ثابتة في وقت التجميع، مع كل عضو في
قائمة الوحدات النمطية الثابتة.

.. testcode::

    class SubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2))

        def forward(self, input):
            return self.weight + input

    class MyModule(torch.nn.Module):
        __constants__ = ['mods']

        def __init__(self):
            super().__init__()
            self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])

        def forward(self, v):
            for module in self.mods:
                v = module(v)
            return v


    m = torch.jit.script(MyModule())

Break and Continue
^^^^^^^^^^^^^^^^^^
::

    for i in range(5):
        if i == 1:
            continue
        if i == 3:
            break
        print(i)

Return
^^^^^^
::

    return a, b

Variable Resolution
~~~~~~~~~~~~~~~~~~~

يدعم TorchScript مجموعة فرعية من قواعد Python لحل المتغيرات
(أي النطاق). تتصرف المتغيرات المحلية بنفس الطريقة التي تعمل بها في Python، باستثناء القيد
يجب أن يكون للمتغير نفس النوع على جميع المسارات عبر الدالة.
إذا كان للمتغير نوع مختلف على فروع مختلفة من عبارة if، فمن الخطأ استخدامه بعد نهاية
إذا كان البيان.

وبالمثل، لا يُسمح باستخدام متغير إذا كان محددًا فقط على طول بعض المسارات عبر الدالة.

مثال:

.. testcode::

    @torch.jit.script
    def foo(x):
        if x < 0:
            y = 4
        print(y)

.. testoutput::

     Traceback (most recent call last):
       ...
     RuntimeError: ...

     y is not defined in the false branch...
     @torch.jit.script...
     def foo(x):
         if x < 0:
         ~~~~~~~~~
             y = 4
             ~~~~~ <--- HERE
         print(y)
     and was used here:
         if x < 0:
             y = 4
         print(y)
               ~ <--- HERE...

يتم حل المتغيرات غير المحلية إلى قيم Python في وقت التجميع عند
يتم تعريف الدالة. يتم بعد ذلك تحويل هذه القيم إلى قيم TorchScript باستخدام
القواعد الموضحة في `Use of Python Values`_.

Use of Python Values
~~~~~~~~~~~~~~~~~~~~

لجعل كتابة TorchScript أكثر ملاءمة، نسمح لرمز النص البرمجي بالإشارة
إلى قيم Python في النطاق المحيط. على سبيل المثال، في أي وقت يكون هناك مرجع لـ ``torch``،
يقوم مترجم TorchScript فعليًا بحلها إلى الوحدة النمطية لـ Python ``torch`` عندما يتم الإعلان عن الدالة. هذه القيم
Python ليست جزءًا من الدرجة الأولى في TorchScript. بدلاً من ذلك، يتم إلغاء السكر في وقت التجميع
إلى الأنواع الأولية التي يدعمها TorchScript. يعتمد هذا على النوع الديناميكي لقيمة Python المشار إليها عند حدوث التجميع.
يوضح هذا القسم القواعد المستخدمة عند الوصول إلى قيم Python في TorchScript.

Functions
^^^^^^^^^

يمكن لـ TorchScript استدعاء وظائف Python. هذه الوظيفة مفيدة جدًا عند
تحويل نموذج بشكل متزايد إلى TorchScript. يمكن نقل النموذج من وظيفة إلى وظيفة
إلى TorchScript، مع ترك استدعاءات لوظائف Python في مكانها. بهذه الطريقة يمكنك بشكل متزايد
التحقق من صحة النموذج أثناء التنقل.


.. autofunction:: torch.jit.is_scripting

.. autofunction:: torch.jit.is_tracing


Attribute Lookup On Python Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
يمكن لـ TorchScript البحث عن السمات في الوحدات النمطية. يتم الوصول إلى `Builtin functions` مثل ``torch.add``
بهذه الطريقة. يسمح هذا لـ TorchScript باستدعاء الوظائف المحددة في
الوحدات النمطية الأخرى.

.. _constant:

Python-defined Constants
^^^^^^^^^^^^^^^^^^^^^^^^
يوفر TorchScript أيضًا طريقة لاستخدام الثوابت المحددة في Python.
يمكن استخدام هذه الثوابت لترميز المعلمات الأساسية للنموذج، أو لتحديد الثوابت العالمية. هناك طريقتان لتحديد أن قيمة Python
يجب التعامل معه كقيمة ثابتة.

1. يتم افتراض أن تكون القيم التي يتم البحث عنها كسمات للوحدة النمطية ثابتة:

.. testcode::

    import math
    import torch

    @torch.jit.script
    def fn():
        return math.pi

2. يمكن وضع علامة على سمات ScriptModule على أنها ثابتة عن طريق وضع علامة عليها ``Final[T]``
::

    import torch
    import torch.nn as nn

    class Foo(nn.Module):
        # يمكن أيضًا استخدام `Final` من وحدة `typing_extensions`
        a: torch.jit.Final[int]

        def __init__(self):
            super().__init__()
            self.a = 1 + 4

        def forward(self, input):
            return self.a + input

    f = torch.jit.script(Foo())

أنواع Python الثابتة المدعومة هي:

* ``int``
* ``float``
* ``bool``
* ``torch.device``
* ``torch.layout``
* ``torch.dtype``
* tuples تحتوي على أنواع مدعومة
* ``torch.nn.ModuleList`` والتي يمكن استخدامها في حلقات TorchScript

.. _module attributes:

خصائص الوحدة النمطية
^^^^^^^^^^^^^^^^^

يمكن استخدام غلاف ``torch.nn.Parameter`` و ``register_buffer`` لتخصيص
tensors إلى وحدة نمطية. يتم إضافة القيم الأخرى المخصصة لوحدة نمطية مجمعة
إلى الوحدة النمطية المجمّعة إذا أمكن استنتاج أنواعها. يمكن استخدام جميع `الأنواع`_
المتوفرة في TorchScript كخصائص للوحدة النمطية. الخصائص tensor مماثلة
للأجهزة من الناحية الدلالية. لا يمكن استنتاج نوع القوائم والقواميس الفارغة
وقيم ``None`` ويجب تحديدها عبر
`ملاحظات PEP 526-style <https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations>`_ class.
إذا لم يمكن استنتاج نوع ولم يتم تحديده بشكل صريح، فلن يتم إضافته كخاصية
إلى :class: `ScriptModule` الناتج.

مثال:

.. testcode::

    from typing import List, Dict

    class Foo(nn.Module):
        # يتم تهيئة `words` كقائمة فارغة، لذلك يجب تحديد نوعها
        words: List[str]

        # يمكن استنتاج النوع إذا لم تكن `a_dict` (أدناه) فارغة، ولكن هذا التعليق
        # يضمن أن يتم تحويل `some_dict` إلى النوع الصحيح
        some_dict: Dict[str, int]

        def __init__(self, a_dict):
            super().__init__()
            self.words = []
            self.some_dict = a_dict

            # يمكن استنتاج الأنواع `int`
            self.my_int = 10

        def forward(self, input):
            # type: (str) -> int
            self.words.append(input)
            return self.some_dict[input] + self.my_int

    f = torch.jit.script(Foo({'hi': 2}))