.. _language-refrence-TorchScript:

مرجع لغة TorchScript
==============================

يوضح دليل المرجع هذا بناء الجملة والدلالات الأساسية الخاصة بلغة TorchScript.
TorchScript هي مجموعة فرعية ثابتة النوع من لغة بايثون. توضح هذه الوثيقة ميزات بايثون المدعومة في TorchScript وكذلك كيف تختلف اللغة عن بايثون العادية. أي ميزات بايثون غير المذكورة في دليل المرجع هذا ليست جزءًا من TorchScript. يركز TorchScript بشكل خاص على ميزات بايثون اللازمة لتمثيل نماذج الشبكات العصبية في PyTorch.

.. contents::
    :local:
    :depth: 1

.. _type_system:

المصطلحات
~~~~~~~~~

تستخدم هذه الوثيقة المصطلحات التالية:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - النمط
     - ملاحظات
   * - " "::="
     - يشير إلى أن الرمز المعطى معرف على أنه.
   * - " "
     - يمثل الكلمات الرئيسية والفاصلات الحقيقية التي تعد جزءًا من بناء الجملة.
   * - "A | B"
     - يشير إلى إما A أو B.
   * - "( ) "
     - يشير إلى التجميع.
   * - "[ ] "
     - يشير إلى اختياري.
   * - "A+"
     - يشير إلى تعبير عادي حيث يتكرر المصطلح A مرة واحدة على الأقل.
   * - "A*"
     - يشير إلى تعبير عادي حيث يتكرر المصطلح A صفر مرة أو أكثر.

نظام النوع
~~~~~~~~~
TorchScript هي مجموعة فرعية ثابتة النوع من بايثون. أكبر اختلاف بين TorchScript ولغة بايثون الكاملة هو أن TorchScript تدعم فقط مجموعة صغيرة من الأنواع اللازمة للتعبير عن نماذج الشبكات العصبية.

أنواع TorchScript
^^^^^^^^^^^^^^^^^

يتكون نظام نوع TorchScript من "TSType" و"TSModuleType" كما هو محدد أدناه.

::

    TSAllType ::= TSType | TSModuleType
    TSType    ::= TSMetaType | TSPrimitiveType | TSStructuralType | TSNominalType

يمثل "TSType" معظم أنواع TorchScript التي يمكن تركيبها والتي يمكن استخدامها في إشارات نوع TorchScript.
يشير "TSType" إلى أي مما يلي:

* الأنواع الفوقية، على سبيل المثال، "أي"
* الأنواع الأولية، على سبيل المثال، "int" و"float" و"str"
* الأنواع الهيكلية، على سبيل المثال، "Optional[int]" أو "List[MyClass]"
* الأنواع الاسمية (فئات بايثون)، على سبيل المثال، "MyClass" (محددة من قبل المستخدم)، "torch.tensor" (مدمجة)

يمثل "TSModuleType" فئة "torch.nn.Module" والفئات الفرعية الخاصة بها. يتم التعامل معه بشكل مختلف عن "TSType" لأن مخطط النوع الخاص به يتم استنتاجه جزئيًا من مثيل الكائن وجزئيًا من تعريف الفئة.
وبالتالي، قد لا تتبع مثيلات "TSModuleType" نفس مخطط النوع الثابت. لا يمكن استخدام "TSModuleType" كإشارة نوع TorchScript أو تركيبه مع "TSType" لاعتبارات الأمان من النوع.

الأنواع الفوقية
^^^^^^^^^^

الأنواع الفوقية مجردة جدًا لدرجة أنها تشبه قيود النوع أكثر من الأنواع الفعلية.
يعرّف TorchScript حاليًا نوعًا فوقيًا واحدًا، وهو "أي"، الذي يمثل أي نوع TorchScript.

نوع "Any"
""""""""""

يمثل نوع "أي" أي نوع TorchScript. لا يحدد "أي" أي قيود على النوع، وبالتالي لا يوجد تحقق من النوع على "أي".
وبالتالي، يمكن ربطه بأي نوع من أنواع بيانات بايثون أو TorchScript (على سبيل المثال، "int" أو "tuple" الخاص بـ TorchScript أو فئة بايثون تعسفية غير مكتوبة).

::

    TSMetaType ::= "Any"

حيث:

* "أي" هو اسم فئة بايثون من وحدة "الطباعة". لذلك، لاستخدام نوع "أي"، يجب استيراده من "الطباعة" (على سبيل المثال، "من الطباعة استيراد أي").
* نظرًا لأن "أي" يمكن أن يمثل أي نوع من أنواع TorchScript، فإن مجموعة المشغلين المسموح بها للعمل على قيم هذا النوع على "أي" محدودة.

المشغلون المدعومون لنوع "Any"
""""""""""""""""""""""""""

* تعيين البيانات من نوع "أي".
* ربط معلمة أو إرجاع من نوع "أي".
* "x is"، "x is not" حيث "x" من نوع "أي".
* "isinstance(x، Type)" حيث "x" من نوع "أي".
* يمكن طباعة بيانات من نوع "أي".
* قد تكون البيانات من نوع "List[Any]" قابلة للفرز إذا كانت البيانات قائمة بقيم من نفس النوع "T" وأن "T" يدعم مشغلي المقارنة.

**مقارنة مع بايثون**

"أي" هو أقل أنواع النظام النوعي في TorchScript تقييدًا. بهذا المعنى، فهو مشابه جدًا لفئة "Object" في بايثون. ومع ذلك، يدعم "أي" فقط مجموعة فرعية من المشغلين والطرق التي تدعمها "Object".

ملاحظات التصميم
""""""""""""

عند كتابة نموذج PyTorch، قد نواجه بيانات غير مشاركة في تنفيذ البرنامج النصي. ومع ذلك، يجب وصفه
بواسطة مخطط النوع. ليس من الصعب وصف الأنواع الثابتة للبيانات غير المستخدمة فقط (في سياق البرنامج النصي)، ولكن قد يؤدي أيضًا إلى فشل البرمجة غير الضروري. تم تقديم "أي" لوصف نوع البيانات حيث لا تكون الأنواع الثابتة الدقيقة ضرورية للترجمة.

**المثال 1**

يوضح هذا المثال كيف يمكن استخدام "أي" للسماح للعنصر الثاني من زوج القيمة أن يكون من أي نوع. هذا ممكن
لأن "x[1]" غير مشارك في أي حساب يتطلب معرفة نوعه الدقيق.

.. testcode::

    import torch

    from typing import Tuple
    from typing import Any

    @torch.jit.export
    def inc_first_element(x: Tuple[int, Any]):
        return (x[0]+1, x[1])

    m = torch.jit.script(inc_first_element)
    print(m((1,2.0)))
    print(m((1,(100,200))))

ينتج المثال أعلاه الإخراج التالي:

.. testoutput::

    (2, 2.0)
    (2, (100, 200))

العنصر الثاني من الزوج هو من نوع "أي"، وبالتالي يمكن ربطه بأنواع متعددة.
على سبيل المثال، تربط "(1، 2.0)" نوع float بـ "أي" كما هو الحال في "Tuple[int، Any]"،
في حين تربط "(1، (100، 200))" زوجًا بـ "أي" في الاستدعاء الثاني.


**المثال 2**

يوضح هذا المثال كيف يمكننا استخدام "isinstance" للتحقق ديناميكيًا من نوع البيانات التي تم وضع علامة عليها كنوع "أي":

.. testcode::

    import torch
    from typing import Any

    def f(a:Any):
        print(a)
        return (isinstance(a, torch.Tensor))

    ones = torch.ones([2])
    m = torch.jit.script(f)
    print(m(ones))

ينتج المثال أعلاه الإخراج التالي:

.. testoutput::

     1
     1
    [ CPUFloatType{2} ]
    True

الأنواع الأولية
^^^^^^^^^^

أنواع TorchScript الأولية هي الأنواع التي تمثل نوعًا واحدًا من القيم وتأتي مع اسم نوع محدد مسبقًا.

::

    TSPrimitiveType ::= "int" | "float" | "double" | "complex" | "bool" | "str" | "None"

الأنواع الهيكلية
^^^^^^^^^^^

الأنواع الهيكلية هي أنواع يتم تعريفها هيكليًا بدون اسم محدد من قبل المستخدم (على عكس الأنواع الاسمية)،
مثل "Future[int]". الأنواع الهيكلية قابلة للتركيب مع أي "TSType".

::

    TSStructuralType ::=  TSTuple | TSNamedTuple | TSList | TSDict |
                        TSOptional | TSUnion | TSFuture | TSRRef | TSAwait

    TSTuple          ::= "Tuple" "[" (TSType ",")* TSType "]"
    TSNamedTuple     ::= "namedtuple" "(" (TSType ",")* TSType ")"
    TSList           ::= "List" "[" TSType "]"
    TSOptional       ::= "Optional" "[" TSType "]"
    TSUnion          ::= "Union" "[" (TSType ",")* TSType "]"
    TSFuture         ::= "Future" "[" TSType "]"
    TSRRef           ::= "RRef" "[" TSType "]"
    TSAwait          ::= "Await" "[" TSType "]"
    TSDict           ::= "Dict" "[" KeyType "," TSType "]"
    KeyType          ::= "str" | "int" | "float" | "bool" | TensorType | "Any"

حيث:

* "Tuple" و"List" و"Optional" و"Union" و"Future" و"Dict" تمثل أسماء فئات بايثون المحددة في الوحدة النمطية "الطباعة". لاستخدام أسماء الأنواع هذه، يجب استيرادها من "الطباعة" (على سبيل المثال، "من الطباعة استيراد الزوج").
* "namedtuple" يمثل فئة بايثون "collections.namedtuple" أو "typing.NamedTuple".
* "Future" و"RRef" يمثلان فئات بايثون "torch.futures" و"torch.distributed.rpc".
* "Await" يمثل فئة بايثون "torch._awaits._Await"

**مقارنة مع بايثون**

بصرف النظر عن إمكانية تركيبها مع أنواع TorchScript، غالبًا ما تدعم هذه الأنواع الهيكلية لـ TorchScript مجموعة فرعية مشتركة من المشغلين والطرق الخاصة بنظيراتها في بايثون.

**المثال 1**

يستخدم هذا المثال بناء جملة "typing.NamedTuple" لتحديد زوج:

.. testcode::

    import torch
    from typing import NamedTuple
    from typing import Tuple

    class MyTuple(NamedTuple):
        first: int
        second: int

    def inc(x: MyTuple) -> Tuple[int, int]:
        return (x.first+1, x.second+1)

    t = MyTuple(first=1, second=2)
    scripted_inc = torch.jit.script(inc)
    print("TorchScript:", scripted_inc(t))

ينتج المثال أعلاه الإخراج التالي:

.. testoutput::

    TorchScript: (2, 3)

**المثال 2**

يستخدم هذا المثال بناء جملة "collections.namedtuple" لتحديد زوج:

.. testcode::

    import torch
    from typing import NamedTuple
    from typing import Tuple
    from collections import namedtuple

    _AnnotatedNamedTuple = NamedTuple('_NamedTupleAnnotated', [('first', int), ('second', int)])
    _UnannotatedNamedTuple = namedtuple('_NamedTupleAnnotated', ['first', 'second'])

    def inc(x: _AnnotatedNamedTuple) -> Tuple[int, int]:
        return (x.first+1, x.second+1)

    m = torch.jit.script(inc)
    print(inc(_UnannotatedNamedTuple(1,2)))

ينتج المثال أعلاه الإخراج التالي:

.. testoutput::

    (2, 3)

**المثال 3**

يوضح هذا المثال خطأ شائعًا في وضع علامات على الأنواع الهيكلية، أي عدم استيراد فئات الأنواع المركبة من الوحدة النمطية "الطباعة":

::

    import torch

    # ERROR: لا يتم التعرف على الزوج لأنه غير مستورد من الطباعة
    @torch.jit.export
    def inc(x: Tuple[int, int]):
        return (x[0]+1, x[1]+1)

    m = torch.jit.script(inc)
    print(m((1,2)))

ينتج عن تشغيل الكود أعلاه خطأ البرمجة التالي:

::

    File "test-tuple.py", line 5, in <module>
        def inc(x: Tuple[int, int]):
    NameError: name 'Tuple' is not defined

العلاج هو إضافة السطر "من الطباعة استيراد الزوج" في بداية الكود.

الأنواع الاسمية
^^^^^^^^^

أنواع TorchScript الاسمية هي فئات بايثون. تسمى هذه الأنواع بالاسمية لأنها معلنة باسم مخصص ويتم مقارنتها باستخدام أسماء الفئات. يتم تصنيف الفئات الاسمية بشكل أكبر إلى الفئات التالية:

::

    TSNominalType ::= TSBuiltinClasses | TSCustomClass | TSEnum

من بينها، يجب أن تكون "TSCustomClass" و"TSEnum" قابلة للترجمة إلى تمثيل TorchScript الوسيط (IR). يتم تطبيق ذلك بواسطة مدقق النوع.

الفئة المدمجة
^^^^^^^^^^^

أنواع الاسم المدمجة هي فئات بايثون التي تكون دلالتها مضمنة في نظام TorchScript (مثل أنواع المنسوجات).
يحدد TorchScript دلالة هذه الأنواع المدمجة، ويدعم غالبًا مجموعة فرعية فقط من الطرق أو
سمات تعريف الفئة الخاصة بها في بايثون.

::

    TSBuiltinClass ::= TSTensor | "torch.device" | "torch.Stream" | "torch.dtype" |
                       "torch.nn.ModuleList" | "torch.nn.ModuleDict" | ...
    TSTensor       ::= "torch.Tensor" | "common.SubTensor" | "common.SubWithTorchFunction" |
                       "torch.nn.parameter.Parameter" | والفئات الفرعية لـ torch.Tensor


ملاحظة خاصة حول "torch.nn.ModuleList" و"torch.nn.ModuleDict"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

على الرغم من تعريف "torch.nn.ModuleList" و"torch.nn.ModuleDict" كقائمة وقاموس في بايثون،
إلا أنها تتصرف بشكل أكثر مثل الأزواج في TorchScript:

* في TorchScript، تكون مثيلات "torch.nn.ModuleList" أو "torch.nn.ModuleDict" ثابتة.
* يتم فك تشفير الكود الذي يقوم بالتعيين على "torch.nn.ModuleList" أو "torch.nn.ModuleDict" تمامًا بحيث يمكن أن تكون عناصر "torch.nn.ModuleList" أو مفاتيح "torch.nn.ModuleDict" من الفئات الفرعية المختلفة لـ "torch.nn.Module".

**مثال**

يسلط المثال التالي الضوء على استخدام بعض فئات Torchscript المدمجة (torch.*):

::

    import torch

    @torch.jit.script
    class A:
        def __init__(self):
            self.x = torch.rand(3)

        def f(self, y: torch.device):
            return self.x.to(device=y)

    def g():
        a = A()
        return a.f(torch.device("cpu"))

    script_g = torch.jit.script(g)
    print(script_g.graph)

الفئة المخصصة
^^^^^^^^^^^^

على عكس الفئات المدمجة، تكون دلالة الفئات المخصصة محددة من قبل المستخدم ويجب أن يكون تعريف الفئة بأكمله قابلًا للترجمة إلى تمثيل IR الخاص بـ TorchScript ويخضع لقواعد التحقق من نوع TorchScript.
بالتأكيد! فيما يلي ترجمة لنص ReStructuredText إلى اللغة العربية:

::

    TSClassDef ::= [ "@torch.jit.script" ]
                     "class" ClassName [ "(object)" ]  ":"
                        MethodDefinition |
                    [ "@torch.jit.ignore" ] | [ "@torch.jit.unused" ]
                        MethodDefinition

حيث:

* يجب أن تكون الفئات من النوع الجديد. يدعم Python 3 فقط الفئات من النوع الجديد. في Python 2.x، يتم تحديد فئة جديدة عن طريق الوراثة من الكائن.
* يتم كتابة أنواع بيانات الخصائص بشكل ثابت، ويجب الإعلان عن خصائص الكائنات عن طريق التخصيص داخل طريقة ``__init__()``.
* لا يتم دعم التحميل الزائد للطرق (أي لا يمكنك امتلاك طرق متعددة بنفس اسم الطريقة).
* يجب أن تكون ``MethodDefinition`` قابلة للترجمة إلى TorchScript IR وتلتزم بقواعد فحص أنواع TorchScript، (أي يجب أن تكون جميع الطرق وظائف TorchScript صالحة ويجب أن تكون تعريفات خصائص الفئة عبارة عن عبارات TorchScript صالحة).
* يمكن استخدام ``torch.jit.ignore`` و ``torch.jit.unused`` لتجاهل الطريقة أو الوظيفة التي لا تدعم Torchscript بشكل كامل أو التي يجب تجاهلها بواسطة المترجم.

**المقارنة مع Python**

فئات TorchScript المخصصة محدودة للغاية مقارنة بنظيراتها في Python. ففئات Torchscript المخصصة:

* لا تدعم خصائص الفئة.
* لا تدعم الوراثة باستثناء الوراثة من نوع واجهة أو كائن.
* لا تدعم التحميل الزائد للطرق.
* يجب أن تقوم بتحديد جميع خصائص الكائنات الخاصة بها في ``__init__()``؛ لأن TorchScript يقوم ببناء مخطط ثابت للفئة عن طريق استنتاج أنواع الخصائص في ``__init__()``.
* يجب أن تحتوي فقط على طرق تلبي قواعد فحص أنواع TorchScript ويمكن ترجمتها إلى TorchScript IRs.

**المثال 1**

يمكن استخدام الفئات في Python في TorchScript إذا تم وضع علامة عليها باستخدام ``@torch.jit.script``، على غرار كيفية الإعلان عن وظيفة TorchScript:

::

    @torch.jit.script
    class MyClass:
        def __init__(self, x: int):
            self.x = x

        def inc(self, val: int):
            self.x += val


**المثال 2**

يجب أن "تعلن" فئة TorchScript المخصصة جميع خصائص كائناتها عن طريق التخصيص في ``__init__()``. إذا لم يتم تحديد خاصية كائن في ``__init__()`` ولكن تم الوصول إليها في طرق أخرى للفئة، فلن يتم تجميع الفئة كفئة TorchScript، كما هو موضح في المثال التالي:

::

    import torch

    @torch.jit.script
    class foo:
        def __init__(self):
            self.y = 1

    # ERROR: self.x غير معرف في __init__
    def assign_x(self):
        self.x = torch.rand(2, 3)

ستفشل الفئة في التجميع وستصدر الخطأ التالي:

::

    RuntimeError:
    حاولت تعيين سمة غير موجودة: x. هل نسيت تحديدها في __init__()?:
    def assign_x(self):
        self.x = torch.rand(2, 3)
        ~~~~~~~~~~~~~~~~~~~~~~~~ <--- هنا

**المثال 3**

في هذا المثال، تقوم فئة TorchScript المخصصة بتعريف متغير فئة، وهو ما لا يُسمح به:

::

    import torch

    @torch.jit.script
    class MyClass(object):
        name = "MyClass"
        def __init__(self, x: int):
            self.x = x

    def fn(a: MyClass):
        return a.name

يؤدي ذلك إلى خطأ وقت التجميع التالي:

::

    RuntimeError:
    '__torch__.MyClass' ليس لديه سمة أو طريقة باسم 'name'. هل نسيت تحديد سمة في __init__()?:
        File "test-class2.py", line 10
    def fn(a: MyClass):
        return a.name
            ~~~~~~ <--- هنا

نوع Enum
^^^^^^^^^

مثل الفئات المخصصة، فإن دلالة نوع enum معرفة من قبل المستخدم ويجب أن يكون تعريف الفئة بأكملها قابل للترجمة إلى TorchScript IR ويلتزم بقواعد فحص أنواع TorchScript.

::

    TSEnumDef ::= "class" Identifier "(enum.Enum | TSEnumType)" ":"
                   ( MemberIdentifier "=" Value )+
                   ( MethodDefinition )*

حيث:

* يجب أن تكون القيمة عبارة عن ثابتة TorchScript من النوع ``int`` أو ``float`` أو ``str``، ويجب أن تكون من نفس نوع TorchScript.
* ``TSEnumType`` هو اسم نوع مدرج في TorchScript. وعلى غرار enum في Python، يسمح TorchScript بالوراثة المقيدة من ``Enum``، أي أن الوراثة من نوع مدرج مسموح بها فقط إذا لم يتم تحديد أي أعضاء.

**المقارنة مع Python**

* يدعم TorchScript فقط ``enum.Enum``. ولا يدعم الاختلافات الأخرى مثل ``enum.IntEnum`` و ``enum.Flag`` و ``enum.IntFlag`` و ``enum.auto``.
* يجب أن تكون قيم أعضاء TorchScript من نفس النوع ويمكن أن تكون فقط من الأنواع ``int`` أو ``float`` أو ``str``، في حين يمكن أن تكون قيم أعضاء enum في Python من أي نوع.
* يتم تجاهل الأنواع التي تحتوي على طرق في TorchScript.

**المثال 1**

يحدد المثال التالي الفئة ``Color`` كنوع ``Enum``:

::

    import torch
    from enum import Enum

    class Color(Enum):
        RED = 1
        GREEN = 2

    def enum_fn(x: Color, y: Color) -> bool:
        if x == Color.RED:
            return True
        return x == y

    m = torch.jit.script(enum_fn)

    print("Eager: ", enum_fn(Color.RED, Color.GREEN))
    print("TorchScript: ", m(Color.RED, Color.GREEN))

**المثال 2**

يوضح المثال التالي حالة الوراثة المقيدة من enum، حيث لا يقوم ``BaseColor`` بتحديد أي عضو، وبالتالي يمكن أن يرث منه ``Color``:

::

    import torch
    from enum import Enum

    class BaseColor(Enum):
        def foo(self):
            pass

    class Color(BaseColor):
        RED = 1
        GREEN = 2

    def enum_fn(x: Color, y: Color) -> bool:
        if x == Color.RED:
            return True
        return x == y

    m = torch.jit.script(enum_fn)

    print("TorchScript: ", m(Color.RED, Color.GREEN))
    print("Eager: ", enum_fn(Color.RED, Color.GREEN))

فئة وحدة نمطية TorchScript
^^^^^^^^^^^^^^^^^^^^^^^^

``TSModuleType`` هو نوع فئة خاص يتم استنتاجه من مثيلات الكائنات التي يتم إنشاؤها خارج TorchScript. يتم تسمية ``TSModuleType`` باسم فئة Python لكائن المثيل. لا تعتبر طريقة ``__init__()`` للفئة في Python طريقة TorchScript، لذلك لا يتعين عليها الالتزام بقواعد فحص أنواع TorchScript.

يتم بناء مخطط النوع لمثيل الفئة مباشرة من كائن المثيل (الذي تم إنشاؤه خارج نطاق TorchScript) بدلاً من استنتاجه من ``__init__()`` مثل الفئات المخصصة. من الممكن أن يتبع كائنان من نفس نوع مثيل الفئة مخططي نوع مختلفين.

وبهذا المعنى، فإن ``TSModuleType`` ليس نوعًا ثابتًا حقًا. لذلك، لأسباب تتعلق باعتبارات الأمان، لا يمكن استخدام ``TSModuleType`` في تعليمة نوع TorchScript أو تركيبه مع ``TSType``.

مثيل الفئة
^^^^^^^^^^^^^^^^^^^^^

يمثل نوع الوحدة النمطية TorchScript مخطط نوع مثيل وحدة نمطية PyTorch المعرفة من قبل المستخدم. عند كتابة وحدة نمطية PyTorch، يتم دائمًا إنشاء كائن الوحدة النمطية خارج نطاق TorchScript (أي يتم تمريره كمعلمة إلى ``forward``). تتم معاملة فئة الوحدة النمطية في Python كفئة مثيل وحدة نمطية، لذلك لا تخضع طريقة ``__init__()`` لفئة الوحدة النمطية لقواعد فحص أنواع TorchScript.

::

    TSModuleType ::= "class" Identifier "(torch.nn.Module)" ":"
                        ClassBodyDefinition

حيث:

* يجب أن تكون طريقة ``forward()`` والطرق الأخرى المزينة بـ ``@torch.jit.export`` قابلة للترجمة إلى TorchScript IR وتخضع لقواعد فحص أنواع TorchScript.

على عكس الفئات المخصصة، لا يلزم سوى أن تكون طريقة ``forward`` والطرق الأخرى المزينة بـ ``@torch.jit.export`` من نوع الوحدة النمطية قابلة للترجمة. والأهم من ذلك، أن طريقة ``__init__()`` لا تعتبر طريقة TorchScript. وبالتالي، لا يمكن استدعاء منشئي نوع الوحدة النمطية ضمن نطاق TorchScript. بدلاً من ذلك، يتم دائمًا إنشاء كائنات الوحدة النمطية في TorchScript من الخارج وتمريرها إلى ``torch.jit.script(ModuleObj)``.

**المثال 1**

يوضح هذا المثال بعض ميزات أنواع الوحدات النمطية:

* يتم إنشاء مثيل ``TestModule`` خارج نطاق TorchScript (أي قبل استدعاء ``torch.jit.script``).
* لا تعتبر طريقة ``__init__()`` طريقة TorchScript، لذلك لا يلزم وضع علامة عليها ويمكن أن تحتوي على أي رمز Python. بالإضافة إلى ذلك، لا يمكن استدعاء طريقة ``__init__()`` لفئة مثيل في رمز TorchScript. نظرًا لأن مثيلات ``TestModule`` يتم إنشاؤها في Python، في هذا المثال، يقوم ``TestModule(2.0)`` و ``TestModule(2)`` بإنشاء مثيلين لهما نوعان مختلفان لخصائص بياناتهما. ``self.x`` من النوع ``float`` لـ ``TestModule(2.0)``، في حين أن ``self.y`` من النوع ``int`` لـ ``TestModule(2.0)``.
* يقوم TorchScript تلقائيًا بتجميع الطرق الأخرى (مثل ``mul()``) التي تستدعيها الطرق الموضحة باستخدام ``@torch.jit.export`` أو طرق ``forward()``.
* نقاط الدخول إلى برنامج TorchScript هي إما ``forward()`` لنوع الوحدة النمطية، أو وظائف موضحة باستخدام ``torch.jit.script``، أو طرق موضحة باستخدام ``torch.jit.export``.

.. testcode::

    import torch

    class TestModule(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.x = v

        def forward(self, inc: int):
            return self.x + inc

    m = torch.jit.script(TestModule(1))
    print(f"First instance: {m(3)}")

    m = torch.jit.script(TestModule(torch.ones([5])))
    print(f"Second instance: {m(3)}")

ينتج المثال أعلاه الإخراج التالي:

.. testoutput::

    First instance: 4
    Second instance: tensor([4., 4., 4., 4., 4.])

**المثال 2**

يوضح المثال التالي استخدامًا غير صحيح لنوع الوحدة النمطية. على وجه التحديد، يستدعي هذا المثال منشئ ``TestModule`` داخل نطاق TorchScript:

.. testcode::

    import torch

    class TestModule(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.x = v

        def forward(self, x: int):
            return self.x + x

    class MyModel:
        def __init__(self, v: int):
            self.val = v

        @torch.jit.export
        def doSomething(self, val: int) -> int:
            # error: لا ينبغي استدعاء منشئ نوع الوحدة النمطية
            myModel = TestModule(self.val)
            return myModel(val)

    # m = torch.jit.script(MyModel(2)) # يؤدي إلى خطأ RuntimeError التالي
    # RuntimeError: Could not get name of python class object

.. _type_annotation:

تعليمة نوع
~~~~~~~~~~~
نظرًا لأن TorchScript له أنواع ثابتة، يحتاج المبرمجون إلى وضع علامات على الأنواع في *النقاط الاستراتيجية* لرمز TorchScript بحيث يكون لكل متغير محلي أو سمة بيانات كائن نوع ثابت، ولكل وظيفة وطريقة توقيع بنوع ثابت.

متى يتم وضع علامات على الأنواع
^^^^^^^^^^^^^^^^^^^^^^
بشكل عام، تكون تعليقات الأنواع مطلوبة فقط في الأماكن التي لا يمكن فيها استنتاج الأنواع الثابتة تلقائيًا (على سبيل المثال، المعلمات أو في بعض الأحيان أنواع الإرجاع للطرق أو الوظائف). غالبًا ما يتم استنتاج أنواع المتغيرات المحلية وخصائص البيانات من عبارات التعيين الخاصة بها. في بعض الأحيان، قد يكون النوع المستنتج مقيدًا للغاية، على سبيل المثال، يتم استنتاج ``x`` على أنه ``NoneType`` من خلال التعيين ``x = None``، في حين أن ``x`` هو في الواقع ``Optional``. في مثل هذه الحالات، قد تكون تعليقات الأنواع مطلوبة لتجاوز الاستنتاج التلقائي، على سبيل المثال، ``x: Optional[int] = None``. لاحظ أنه من الآمن دائمًا وضع علامة على نوع متغير محلي أو سمة بيانات حتى إذا كان من الممكن استنتاج النوع تلقائيًا. يجب أن يكون النوع المعلم متوافقًا مع فحص نوع TorchScript.

عندما لا يتم وضع علامة على معلمة أو متغير محلي أو سمة بيانات ولا يمكن استنتاج النوع تلقائيًا، يفترض TorchScript أنها من النوع الافتراضي ``TensorType`` أو ``List[TensorType]`` أو ``Dict[str, TensorType]``.

وضع علامات على توقيع الدالة
^^^^^^^^^^^^^^^^^^^^^^^^^^^
نظرًا لأنه قد لا يتم استنتاج معلمة من جسم الدالة (بما في ذلك كل من الوظائف والطرق)، فيجب وضع علامات على الأنواع. وإلا، فإنها تفترض النوع الافتراضي ``TensorType``.

يدعم TorchScript أسلوبين لوضع علامات على توقيعات الطرق والوظائف:

* **Python3-style** يعلق الأنواع مباشرة على التوقيع. وبالتالي، فإنه يسمح بترك معلمات فردية بدون علامة نوع (والتي سيكون نوعها الافتراضي ``TensorType``)، أو يسمح بترك نوع الإرجاع بدون علامة (والذي سيتم استنتاجه تلقائيًا).
::

    Python3Annotation ::= "def" Identifier [ "(" ParamAnnot* ")" ] [ReturnAnnot] ":"
                                FuncOrMethodBody
    ParamAnnot        ::= Identifier [ ":" TSType ] ","
    ReturnAnnot       ::= "->" TSType

يرجى ملاحظة أنه عند استخدام نمط Python3، يتم استنتاج نوع "self" تلقائيًا ولا يجب إضافته في التعليق.

* **نمط Mypy** يضيف الأنواع كتعليق مباشرة أسفل تعريف الدالة/الطريقة. وبما أن أسماء المعاملات لا تظهر في التعليق، فيجب إضافة أنواع لجميع المعاملات.


::

    MyPyAnnotation ::= "# type:" "(" ParamAnnot* ")" [ ReturnAnnot ]
    ParamAnnot     ::= TSType ","
    ReturnAnnot    ::= "->" TSType

**المثال 1**

في هذا المثال:

* لا يتم إضافة نوع إلى "a" ويتم افتراض النوع الافتراضي "TensorType".
* يتم إضافة نوع "int" إلى المعامل "b".
* لا يتم إضافة نوع القيمة المرجعة ويتم استنتاجه تلقائيًا كنوع "TensorType" (بناءً على نوع القيمة التي يتم إرجاعها).

::

    import torch

    def f(a, b: int):
        return a+b

    m = torch.jit.script(f)
    print("TorchScript:", m(torch.ones([6]), 100))

**المثال 2**

يستخدم المثال التالي نمط Mypy للإضافة. يرجى ملاحظة أنه يجب إضافة أنواع إلى المعاملات أو قيم الإرجاع حتى إذا كان بعضها يستخدم النوع الافتراضي.

::

    import torch

    def f(a, b):
        # type: (torch.Tensor, int) → torch.Tensor
        return a+b

    m = torch.jit.script(f)
    print("TorchScript:", m(torch.ones([6]), 100))


إضافة أنواع إلى المتغيرات وسمات البيانات
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
بشكل عام، يمكن استنتاج أنواع سمات البيانات (بما في ذلك سمات البيانات الخاصة بالصفوف والنماذج) والمتغيرات المحلية تلقائيًا من جمل التعيين. ومع ذلك، إذا ارتبط متغير أو سمة بقيم ذات أنواع مختلفة (على سبيل المثال، كقيمة "None" أو "TensorType")، فقد تحتاج إلى إضافة نوع صريح لها كنوع "أوسع"، مثل "Optional[int]" أو "Any".

المتغيرات المحلية
"""""""""""""
يمكن إضافة أنواع إلى المتغيرات المحلية وفقًا لقواعد إضافة الأنواع في وحدة "typing" في Python3، أي:

::

    LocalVarAnnotation ::= Identifier [":" TSType] "=" Expr

بشكل عام، يمكن استنتاج أنواع المتغيرات المحلية تلقائيًا. ومع ذلك، في بعض الحالات، قد تحتاج إلى إضافة نوع متعدد إلى متغيرات محلية قد ترتبط بأنواع ملموسة مختلفة. وتشمل الأنواع المتعددة النموذجية "Optional[T]" و"Any".

**مثال**

::

    import torch

    def f(a, setVal: bool):
        value: Optional[torch.Tensor] = None
        if setVal:
            value = a
        return value

    ones = torch.ones([6])
    m = torch.jit.script(f)
    print("TorchScript:", m(ones, True), m(ones, False))

سمات بيانات النماذج
"""""""""""""""
بالنسبة لصفوف "ModuleType"، يمكن إضافة أنواع إلى سمات بيانات النماذج وفقًا لقواعد إضافة الأنواع في وحدة "typing" في Python3. يمكن إضافة أنواع إلى سمات بيانات النماذج (اختياريًا) كسمات "نهائية" باستخدام "Final".

::

    "class" ClassIdentifier "(torch.nn.Module):"
    InstanceAttrIdentifier ":" ["Final("] TSType [")"]
    ...

حيث:

* "InstanceAttrIdentifier" هو اسم سمة النموذج.
* "Final" تشير إلى أنه لا يمكن إعادة تعيين السمة خارج الدالة "__init__" أو تجاوزها في الصفوف الفرعية.

**مثال**

::

    import torch

    class MyModule(torch.nn.Module):
        offset_: int

    def __init__(self, offset):
        self.offset_ = offset

    ...



واجهات برمجة التطبيقات الخاصة بإضافة الأنواع
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``torch.jit.annotate(T, expr)``
"""""""""""""""""""""""""""""""
تقوم واجهة برمجة التطبيقات هذه بإضافة النوع "T" إلى التعبير "expr". ويتم استخدامها غالبًا عندما لا يكون النوع الافتراضي للتعبير هو النوع المقصود من قبل المبرمج.
على سبيل المثال، تحتوي القائمة الفارغة (أو القاموس الفارغ) على النوع الافتراضي "List[TensorType]" (أو "Dict[TensorType, TensorType]")، ولكن في بعض الأحيان قد يتم استخدامها لتهيئة قائمة من أنواع أخرى. وهناك حالة استخدام شائعة أخرى تتمثل في إضافة نوع القيمة المرجعة من الدالة "tensor.tolist()". ومع ذلك، لا يمكن استخدامها لإضافة نوع سمة النموذج في الدالة "__init__"؛ ويجب استخدام "torch.jit.Attribute" بدلاً من ذلك.

**مثال**

في هذا المثال، يتم الإعلان عن "[]" كقائمة من الأعداد الصحيحة باستخدام "torch.jit.annotate" (بدلاً من افتراض أن "[]" هي من النوع الافتراضي "List[TensorType]"):

::

    import torch
    from typing import List

    def g(l: List[int], val: int):
        l.append(val)
        return l

    def f(val: int):
        l = g(torch.jit.annotate(List[int], []), val)
        return l

    m = torch.jit.script(f)
    print("Eager:", f(3))
    print("TorchScript:", m(3))


راجع :meth:`torch.jit.annotate` لمزيد من المعلومات.


ملحق إضافة الأنواع
^^^^^^^^^^^^^^

تعريف نظام الأنواع في TorchScript
"""""""""""""""""""""""""""""

::

    TSAllType       ::= TSType | TSModuleType
    TSType          ::= TSMetaType | TSPrimitiveType | TSStructuralType | TSNominalType

    TSMetaType      ::= "Any"
    TSPrimitiveType ::= "int" | "float" | "double" | "complex" | "bool" | "str" | "None"

    TSStructuralType ::= TSTuple | TSNamedTuple | TSList | TSDict | TSOptional |
                         TSUnion | TSFuture | TSRRef | TSAwait
    TSTuple         ::= "Tuple" "[" (TSType ",")* TSType "]"
    TSNamedTuple    ::= "namedtuple" "(" (TSType ",")* TSType ")"
    TSList          ::= "List" "[" TSType "]"
    TSOptional      ::= "Optional" "[" TSType "]"
    TSUnion         ::= "Union" "[" (TSType ",")* TSType "]"
    TSFuture        ::= "Future" "[" TSType "]"
    TSRRef          ::= "RRef" "[" TSType "]"
    TSAwait         ::= "Await" "[" TS太阳公" "]"
    TSDict          ::= "Dict" "[" KeyType "," TSType "]"
    KeyType         ::= "str" | "int" | "float" | "bool" | TensorType | "Any"

    TSNominalType   ::= TSBuiltinClasses | TSCustomClass | TSEnum
    TSBuiltinClass  ::= TSTensor | "torch.device" | "torch.stream"|
                        "torch.dtype" | "torch.nn.ModuleList" |
                        "torch.nn.ModuleDict" | ...
    TSTensor        ::= "torch.tensor" and subclasses

بنيات إضافة الأنواع غير المدعومة
"""""""""""""""""""""""
لا يدعم TorchScript جميع ميزات وأنواع وحدة "typing" في Python3.
أي وظيفة من وحدة "typing" غير محددة صراحة في هذه الوثيقة غير مدعومة. يلخص الجدول التالي بنيات "typing" التي إما غير مدعومة أو مدعومة مع قيود في TorchScript.

=============================  ================
 البند                           الوصف
-----------------------------  ----------------
``typing.Any``                  قيد التطوير
``typing.NoReturn``             غير مدعوم
``typing.Callable``             غير مدعوم
``typing.Literal``              غير مدعوم
``typing.ClassVar``             غير مدعوم
``typing.Final``                مدعوم لسمات النماذج، وسمات الصفوف، والإضافات، ولكن ليس للدوال.
``typing.AnyStr``               غير مدعوم
``typing.overload``             قيد التطوير
أسماء الأنواع البديلة           غير مدعوم
إضافة الأنواع الاسمية           قيد التطوير
إضافة الأنواع التركيبية        غير مدعوم
NewType                         غير مدعوم
Generics                        غير مدعوم
=============================  ================


.. _expressions:


التعبيرات
~~~~~~~

يصف القسم التالي قواعد بناء الجملة الخاصة بالتعبيرات المدعومة في TorchScript.
وهي مبنية على "فصل التعبيرات في مرجع لغة Python <https://docs.python.org/3/reference/expressions.html>`_".

التحويلات الحسابية
^^^^^^^^^^^^
هناك عدد من التحويلات الضمنية للأنواع التي يتم تنفيذها في TorchScript:


* يمكن تحويل "Tensor" ذات نوع بيانات "float" أو "int" بشكل ضمني إلى مثيل من "FloatType" أو "IntType" بشرط أن يكون حجمها 0، وألا يكون لديها "require_grad" مضبوطًا على "True"، وألا تحتاج إلى تضييق.
* يمكن تحويل مثيلات "StringType" بشكل ضمني إلى "DeviceType".
* يمكن تطبيق قواعد التحويل الضمني من نقطتي القائمة السابقتين على مثيلات "TupleType" لإنتاج مثيلات "ListType" ذات النوع المحتوى المناسب.


يمكن استدعاء التحويلات الصريحة باستخدام الدوال المضمنة "float"، و"int"، و"bool"، و"str"
التي تقبل أنواع البيانات الأولية كمعاملات ويمكنها قبول الأنواع المحددة من قبل المستخدم إذا كانت تنفذ
``__bool__``، ``__str__``، إلخ.


الذرات
^^^^^
الذرات هي العناصر الأساسية للتعبيرات.

::

    atom      ::=  identifier | literal | enclosure
    enclosure ::=  parenth_form | list_display | dict_display

المعرفات
"""""""""""
القواعد التي تحدد ما هو معرف قانوني في TorchScript هي نفسها
كما هو الحال في `نظائرها في Python <https://docs.python.org/3/reference/lexical_analysis.html#identifiers>`_.

الأحرف الحرفية
""""""""""

::

    literal ::=  stringliteral | integer | floatnumber

يقوم تقييم الحرفي بإرجاع كائن من النوع المناسب مع القيمة المحددة
(مع تطبيق التقريبات حسب الحاجة للأعداد العائمة). الأحرف الحرفية ثابتة، وقد يحصل التقييمات المتعددة
لأحرف حرفية متطابقة على نفس الكائن أو كائنات مختلفة بنفس القيمة.
`stringliteral <https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals>`_،
`integer <https://docs.python.org/3/reference/lexical_analysis.html#integer-literals>`_، و
`floatnumber <https://docs.python.org/3/reference/lexical_analysis.html#floating-point-literals>`_
محددة بنفس الطريقة كما هي في Python.

الأشكال المحاطة بأقواس
"""""""""""""""""""

::

    parenth_form ::=  '(' [expression_list] ')'

يقوم التعبير المحاط بأقواس بإرجاع ما تقوم قائمة التعبيرات بإرجاعه. إذا احتوت القائمة على فاصلة واحدة على الأقل، فإنها تقوم بإرجاع "Tuple"؛ وإلا، فإنها تقوم بإرجاع التعبير الفردي الموجود داخل قائمة التعبيرات. ويقوم زوج الأقواس الفارغ بإرجاع كائن "Tuple" فارغ (``Tuple[]``).

عرض القوائم والقواميس
""""""""""""""""""

::

    list_comprehension ::=  expression comp_for
    comp_for           ::=  'for' target_list 'in' or_expr
    list_display       ::=  '[' [expression_list | list_comprehension] ']'
    dict_display       ::=  '{' [key_datum_list | dict_comprehension] '}'
    key_datum_list     ::=  key_datum (',' key_datum)*
    key_datum          ::=  expression ':' expression
    dict_comprehension ::=  key_datum comp_for

يمكن إنشاء القوائم والقواميس عن طريق إدراج محتويات الحاوية بشكل صريح أو عن طريق توفير
تعليمات حول كيفية حسابها من خلال مجموعة من تعليمات التكرار (أي "التفهم"). التفهم
هو مكافئ دلاليًا لاستخدام حلقة "for" وإضافة العناصر إلى قائمة مستمرة.
تنشئ التفهمات بشكل ضمني نطاقها الخاص للتأكد من أن عناصر قائمة الأهداف لا تتسرب إلى النطاق المحيط. في حالة إدراج عناصر الحاوية بشكل صريح، يتم تقييم التعبيرات في قائمة التعبيرات
من اليسار إلى اليمين. إذا تم تكرار مفتاح في "dict_display" يحتوي على "key_datum_list"، فإن
القاموس الناتج يستخدم القيمة من آخر عنصر في القائمة يستخدم المفتاح المتكرر.

الأساسيات
^^^^^^^^^

::

    primary ::=  atom | attributeref | subscription | slicing | call


مراجع السمات
""""""""""""

::

    attributeref ::=  primary '.' identifier


يجب أن يقوم "primary" بتقييم كائن من نوع يدعم مراجع السمات التي تحتوي على سمة باسم
``identifier``.

الاشتراكات
""""""""
::

    subscription ::=  primary '[' expression_list ']'

يجب أن يتم تقييم ``primary`` إلى كائن يدعم الاشتراك.

* إذا كان الأساسي هو ``List`` أو ``Tuple`` أو ``str``، يجب أن يتم تقييم قائمة التعبيرات إلى عدد صحيح أو شريحة.
* إذا كان الأساسي هو ``Dict``، يجب أن يتم تقييم قائمة التعبيرات إلى كائن من نفس نوع مفتاح ``Dict``.
* إذا كان الأساسي هو ``ModuleList``، يجب أن يكون التعبير عبارة عن حرفي ``integer``.
* إذا كان الأساسي هو ``ModuleDict``، يجب أن يكون التعبير عبارة عن ``stringliteral``.

Slicings
""""""""
يحدد الشرائح نطاقًا من العناصر في ``str`` أو ``Tuple`` أو ``List`` أو ``Tensor``. يمكن استخدام الشرائح كتعبيرات أو أهداف في تعليمات التعيين أو ``del``.

::

    slicing      ::=  primary '[' slice_list ']'
    slice_list   ::=  slice_item (',' slice_item)* [',']
    slice_item   ::=  expression | proper_slice
    proper_slice ::=  [expression] ':' [expression] [':' [expression] ]

يمكن استخدام الشرائح التي تحتوي على أكثر من عنصر شريحة واحد في قوائم الشرائح الخاصة بها فقط مع الأساسيات التي يتم تقييمها إلى كائن من النوع ``Tensor``.

Calls
"""""

::

    call          ::=  primary '(' argument_list ')'
    argument_list ::=  args [',' kwargs] | kwargs
    args          ::=  [arg (',' arg)*]
    kwargs        ::=  [kwarg (',' kwarg)*]
    kwarg         ::=  arg '=' expression
    arg           ::=  identifier

يجب أن يقوم ``primary`` بإلغاء السكر أو تقييمه إلى كائن قابل للاستدعاء. يتم تقييم جميع تعبيرات الحجج قبل محاولة إجراء المكالمة.

Power Operator
^^^^^^^^^^^^^^

::

    power ::=  primary ['**' u_expr]

لدى مشغل الطاقة نفس الدلالات مثل دالة pow المدمجة (غير المدعومة)؛ فهو يحسب قيمة حجته اليسرى مرفوعة إلى قوة حجته اليمنى. يرتبط بشكل أكثر إحكامًا من المشغلين أحاديين على اليسار، ولكنه أقل إحكامًا من المشغلين أحاديين على اليمين؛ أي ``-2 ** -3 == -(2 ** (-3))``. يمكن أن يكون المعاملان الأيسر والأيمن ``int`` أو ``float`` أو ``Tensor``. يتم بث المقياسين في حالة عمليات الأس الأسكالي/الأسكالي-الشعاعي الشعاعي، ويتم إجراء الأس الأس-شعاعي دون أي بث.

Unary and Arithmetic Bitwise Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    u_expr ::=  power | '-' power | '~' power

يعطي المشغل أحادي ``-`` نتيجة طرح حجته. يعطي المشغل أحادي ``~`` نتيجة عكس البت لحجته. يمكن استخدام ``-`` مع ``int`` و ``float`` و ``Tensor`` من ``int`` و ``float``. يمكن فقط استخدام ``~`` مع ``int`` و ``Tensor`` من ``int``.

Binary Arithmetic Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    m_expr ::=  u_expr | m_expr '*' u_expr | m_expr '@' m_expr | m_expr '//' u_expr | m_expr '/' u_expr | m_expr '%' u_expr
    a_expr ::=  m_expr | a_expr '+' m_expr | a_expr '-' m_expr

يمكن لمشغلي الحساب الثنائي العمل على ``Tensor`` و ``int`` و ``float``. بالنسبة لعمليات الشعاع-الشعاع، يجب أن يكون لكل من الحجج نفس الشكل. بالنسبة لعمليات الشعاعي-الشعاعي أو الشعاعي-الشعاعي، يتم عادةً بث المقياس إلى حجم الشعاع. لا يمكن لعمليات القسمة أن تقبل سوى المقياس كحجة الجانب الأيمن، ولا تدعم البث. يعمل مشغل ``@`` على الضرب المصفوفي ويقبل فقط حجج ``Tensor``. يمكن استخدام مشغل الضرب (``*``) مع قائمة وعدد صحيح للحصول على نتيجة تتكون من القائمة الأصلية المتكررة عددًا معينًا من المرات.

Shifting Operations
^^^^^^^^^^^^^^^^^^^

::

    shift_expr ::=  a_expr | shift_expr ( '<<' | '>>' ) a_expr

تقبل هذه المشغلات وسيطين من نوع ``int``، أو وسيطين من نوع ``Tensor``، أو وسيطًا من نوع ``Tensor`` ووسيطًا من نوع ``int`` أو ``float``. في جميع الحالات، يتم تعريف الإزاحة اليمنى بواسطة n كقسمة صحيحة على ``pow(2, n)``، ويتم تعريف الإزاحة اليسرى بواسطة n كضرب في ``pow(2, n)``. عندما يكون كلا الوسيطين من نوع ``Tensors``، يجب أن يكون لهما نفس الشكل. عندما يكون أحدهما مقياسًا والآخر ``Tensor``، يتم بث المقياس منطقيًا لمطابقة حجم ``Tensor``.

Binary Bitwise Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    and_expr ::=  shift_expr | and_expr '&' shift_expr
    xor_expr ::=  and_expr | xor_expr '^' and_expr
    or_expr  ::=  xor_expr | or_expr '|' xor_expr

يقوم المشغل ``&`` بحساب البت AND من وسيطيه، ويقوم المشغل ``^`` بحساب البت XOR، ويقوم المشغل ``|`` بحساب البت OR. يجب أن يكون كلا الوسيطين من نوع ``int`` أو ``Tensor``، أو يجب أن يكون الوسيط الأيسر من نوع ``Tensor`` والوسيط الأيمن من نوع ``int``. عندما يكون كلا الوسيطين من نوع ``Tensor``، يجب أن يكون لهما نفس الشكل. عندما يكون الوسيط الأيمن من نوع ``int``، ويكون الوسيط الأيسر من نوع ``Tensor``، يتم بث الوسيط الأيمن منطقيًا لمطابقة شكل ``Tensor``.

Comparisons
^^^^^^^^^^^

::

    comparison    ::=  or_expr (comp_operator or_expr)*
    comp_operator ::=  '<' | '>' | '==' | '>=' | '<=' | '!=' | 'is' ['not'] | ['not'] 'in'

تعطي المقارنة قيمة منطقية (``True`` أو ``False``)، أو إذا كان أحد الوسيطين من نوع ``Tensor``، فإنها تعطي ``Tensor`` منطقي. يمكن تسلسل المقارنات بشكل تعسفي طالما أنها لا تنتج قيم منطقية ``Tensor`` تحتوي على أكثر من عنصر واحد. ``a op1 b op2 c ...`` مكافئ لـ ``a op1 b and b op2 c and ...``.

Value Comparisons
"""""""""""""""""
تقارن المشغلات ``<`` و ``>`` و ``==`` و ``>=`` و ``<=`` و ``!=`` قيمتين لاثنين من الكائنات. بشكل عام، يجب أن يكون للكائنين نفس النوع، ما لم يكن هناك تحويل نوع ضمني متاح بين الكائنات. يمكن مقارنة الأنواع المحددة من قبل المستخدم إذا تم تعريف طرق المقارنة الغنية (مثل ``__lt__``) عليها. تعمل مقارنة الأنواع المدمجة مثل Python:

* تتم مقارنة الأرقام حسابيا.
* تتم مقارنة السلاسل أبجديًا.
* يمكن مقارنة ``lists`` و ``tuples`` و ``dicts`` فقط مع ``lists`` و ``tuples`` و ``dicts`` أخرى من نفس النوع ويتم مقارنتها باستخدام مشغل المقارنة للعناصر المقابلة.

Membership Test Operations
""""""""""""""""""""""""""
تقوم المشغلات ``in`` و ``not in`` باختبار العضوية. ``x in s`` تقييمها إلى ``True`` إذا كان ``x`` عضوًا في ``s`` و ``False`` في حال العكس. ``x not in s`` مكافئ لـ ``not x in s``. هذا المشغل مدعوم لـ ``lists`` و ``dicts`` و ``tuples``، ويمكن استخدامه مع الأنواع المحددة من قبل المستخدم إذا تم تنفيذ طريقة ``__contains__`` عليها.

Identity Comparisons
""""""""""""""""""""
بالنسبة لجميع الأنواع باستثناء ``int`` و ``double`` و ``bool`` و ``torch.device``، تقوم المشغلات ``is`` و ``is not`` باختبار هوية الكائن؛ ``x is y`` هي ``True`` إذا وفقط إذا كان ``x`` و ``y`` هما نفس الكائن. بالنسبة لجميع الأنواع الأخرى، فإن ``is`` مكافئ لمقارنتها باستخدام ``==``. ``x is not y`` يعطي نتيجة عكس ``x is y``.

Boolean Operations
^^^^^^^^^^^^^^^^^^

::

    or_test  ::=  and_test | or_test 'or' and_test
    and_test ::=  not_test | and_test 'and' not_test
    not_test ::=  'bool' '(' or_expr ')' | comparison | 'not' not_test

يمكن للأنواع المحددة من قبل المستخدم تخصيص تحويلها إلى ``bool`` عن طريق تنفيذ طريقة ``__bool__``. يعطي المشغل ``not`` نتيجة ``True`` إذا كانت وسيطته خاطئة، و ``False`` في حال العكس. يتم تقييم التعبير ``x`` و ``y`` أولاً لـ ``x``؛ إذا كان ``False``، تتم إعادته (``False``)؛ وإلا، يتم تقييم ``y`` وإعادة قيمته (``False`` أو ``True``). يتم تقييم التعبير ``x`` أو ``y`` أولاً لـ ``x``؛ إذا كان ``True``، تتم إعادته (``True``)؛ وإلا، يتم تقييم ``y`` وإعادة قيمته (``False`` أو ``True``).

Conditional Expressions
^^^^^^^^^^^^^^^^^^^^^^^

::

   conditional_expression ::=  or_expr ['if' or_test 'else' conditional_expression]
    expression            ::=  conditional_expression

يقوم التعبير ``x if c else y`` أولاً بتقييم الشرط ``c`` بدلاً من x. إذا كان ``c`` يساوي ``True``، يتم تقييم ``x`` وإعادة قيمته؛ وإلا، يتم تقييم ``y`` وإعادة قيمته. كما هو الحال مع جمل if، يجب أن يكون لكل من ``x`` و ``y`` قيمة من نفس النوع.

Expression Lists
^^^^^^^^^^^^^^^^

::

    expression_list ::=  expression (',' expression)* [',']
    starred_item    ::=  '*' primary

يمكن أن يظهر العنصر النجمي فقط على الجانب الأيسر من عبارة التعيين، على سبيل المثال، ``a, *b, c = ...``.

.. statements:

Simple Statements
~~~~~~~~~~~~~~~~~

يوضح القسم التالي بناء جملة العبارات البسيطة المدعومة في TorchScript.
وهو يستند إلى 'فصل العبارات البسيطة في مرجع لغة Python <https://docs.python.org/3/reference/simple_stmts.html>`_.

Expression Statements
^^^^^^^^^^^^^^^^^^^^^^

::

    expression_stmt    ::=  starred_expression
    starred_expression ::=  expression | (starred_item ",")* [starred_item]
    starred_item       ::=  assignment_expression | "*" or_expr

Assignment Statements
^^^^^^^^^^^^^^^^^^^^^^

::

    assignment_stmt ::=  (target_list "=")+ (starred_expression)
    target_list     ::=  target ("," target)* [","]
    target          ::=  identifier
                         | "(" [target_list] ")"
                         | "[" [target_list] "]"
                         | attributeref
                         | subscription
                         | slicing
                         | "*" target

Augmented Assignment Statements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    augmented_assignment_stmt ::= augtarget augop (expression_list)
    augtarget                 ::= identifier | attributeref | subscription
    augop                     ::= "+=" | "-=" | "*=" | "/=" | "//=" | "%=" |
                                  "**="| ">>=" | "<<=" | "&=" | "^=" | "|="


Annotated Assignment Statements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    annotated_assignment_stmt ::= augtarget ":" expression
                                  ["=" (starred_expression)]

The ``raise`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^

::

    raise_stmt ::=  "raise" [expression ["from" expression]]

لا تدعم عبارات raise في TorchScript جمل ``try\except\finally``.

The ``assert`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    assert_stmt ::=  "assert" expression ["," expression]

لا تدعم عبارات التأكيد في TorchScript جمل ``try\except\finally``.

The ``return`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    return_stmt ::=  "return" [expression_list]

لا تدعم عبارات return في TorchScript جمل ``try\except\finally``.

The ``del`` Statement
^^^^^^^^^^^^^^^^^^^^^^

::

    del_stmt ::=  "del" target_list

The ``pass`` Statement
^^^^^^^^^^^^^^^^^^^^^^^

::

    pass_stmt ::= "pass"

The ``print`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^

::

    print_stmt ::= "print" "(" expression  [, expression] [.format{expression_list}] ")"

The ``break`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^

::

    break_stmt ::= "break"

The ``continue`` Statement:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    continue_stmt ::= "continue"

Compound Statements
~~~~~~~~~~~~~~~~~~~

يوضح القسم التالي بناء جملة العبارات المركبة المدعومة في TorchScript.
كما يسلط الضوء على الاختلافات بين عبارات Torchscript وعبارات Python العادية.
وهو يستند إلى 'فصل العبارات المركبة في مرجع لغة Python <https://docs.python.org/3/reference/compound_stmts.html>`_.

The ``if`` Statement
^^^^^^^^^^^^^^^^^^^^^

يدعم Torchscript كل من ``if/else`` الأساسي والترميزي.

Basic ``if/else`` Statement
""""""""""""""""""""""""""""

::

    if_stmt ::= "if" assignment_expression ":" suite
                ("elif" assignment_expression ":" suite)
                ["else" ":" suite]

يمكن تكرار عبارات ``elif`` لعدد عشوائي من المرات، ولكن يجب أن تكون قبل عبارة ``else``.

Ternary ``if/else`` Statement
""""""""""""""""""""""""""""""

::

    if_stmt ::= return [expression_list] "if" assignment_expression "else" [expression_list]

**Example 1**

يتم ترقية ``tensor`` ذو البعد الواحد إلى ``bool``:

.. testcode::

    import torch

    @torch.jit.script
    def fn(x: torch.Tensor):
        if x: # The tensor gets promoted to bool
            return True
        return False
    print(fn(torch.rand(1)))

ينتج المثال أعلاه الإخراج التالي:

.. testoutput::

    True

**Example 2**

لا يتم ترقية ``tensor`` متعدد الأبعاد إلى ``bool``:
::

    import torch

    # تسبب التنسورات متعددة الأبعاد حدوث أخطاء.

    @torch.jit.script
    def fn():
        if torch.rand(2):
            print("التنسور متوفر")

        if torch.rand(4,5,6):
            print("التنسور متوفر")

    print(fn())

تشغيل الكود أعلاه ينتج عنه خطأ ``RuntimeError`` التالي.

::

    RuntimeError: فشلت العملية التالية في مفسر TorchScript.
    تتبع خطوات TorchScript (آخر استدعاء أولاً):
    @torch.jit.script
    def fn():
        if torch.rand(2):
           ~~~~~~~~~~~~ <--- هنا
            print("التنسور متوفر")
    RuntimeError: قيمة الشرط المنطقي للتنسور ذات القيم المتعددة غير واضحة

إذا كان متغير الشرط مشروحًا على أنه ``final``، يتم تقييم إما فرع "صح" أو "خطأ" حسب تقييم المتغير الشرطي.

**المثال 3**

في هذا المثال، يتم تقييم فرع "صح" فقط، لأن "a" مشروح على أنه ``final`` ومحدد على أنه "صح":

::

    import torch

    a : torch.jit.final[Bool] = True

    if a:
        return torch.empty(2,3)
    else:
        return []


بيان ``while``
^^^^^^^^^^^^^^^

::

    while_stmt ::=  "while" assignment_expression ":" suite

بيانات ``while...else`` غير مدعومة في Torchscript. يؤدي ذلك إلى حدوث خطأ ``RuntimeError``.

بيان ``for-in``
^^^^^^^^^^^^^^

::

    for_stmt ::=  "for" target_list "in" expression_list ":" suite
                  ["else" ":" suite]

بيانات ``for...else`` غير مدعومة في Torchscript. يؤدي ذلك إلى حدوث خطأ ``RuntimeError``.

**المثال 1**

الحلقات التكرارية على التوبلات: تقوم بفك حلقة التكرار، وتوليد جسم لكل عضو في التوبل. يجب أن يكون الجسم صحيحًا من حيث النوع لكل عضو.

.. testcode::

    import torch
    from typing import Tuple

    @torch.jit.script
    def fn():
        tup = (3, torch.ones(4))
        for x in tup:
            print(x)

    fn()

ينتج المثال أعلاه الإخراج التالي:

.. testoutput::

    3
     1
     1
     1
     1
    [ CPUFloatType{4} ]


**المثال 2**

الحلقات التكرارية على القوائم: تقوم بفك حلقات التكرار على ``nn.ModuleList`` في وقت التجميع، مع كل عضو في قائمة الوحدات النمطية.

::

    class SubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2))

        def forward(self, input):
            return self.weight + input

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])

        def forward(self, v):
            for module in self.mods:
                v = module(v)
            return v

    model = torch.jit.script(MyModule())

بيان ``with``
^^^^^^^^^^^^^^
يستخدم بيان ``with`` لتغليف تنفيذ كتلة بالأساليب التي يحددها مدير السياق.

::

    with_stmt ::=  "with" with_item ("," with_item) ":" suite
    with_item ::=  expression ["as" target]

* إذا تم تضمين هدف في بيان ``with``، يتم تعيين قيمة الإرجاع من ``__enter__()`` لمدير السياق إلى ذلك الهدف. وعلى عكس بايثون، إذا تسبب استثناء في الخروج من الكتلة، فإن نوع الاستثناء وقيمته وتتبع مكدس الاستدعاءات لا يتم تمريرها كوسائط إلى ``__exit__()``. يتم توفير ثلاث وسائط ``None``.
* عبارات ``try`` و ``except`` و ``finally`` غير مدعومة داخل كتل ``with``.
* لا يمكن كبت الاستثناءات التي تحدث داخل كتلة ``with``.

بيان ``tuple``
^^^^^^^^^^^^^^

::

    tuple_stmt ::= tuple([iterables])

* تتضمن الأنواع القابلة للتكرار في TorchScript ``Tensors`` و ``lists`` و ``tuples`` و ``dictionaries`` و ``strings`` و ``torch.nn.ModuleList`` و ``torch.nn.ModuleDict``.
* لا يمكنك تحويل قائمة إلى توبل باستخدام دالة tuple المدمجة.

يتم تغطية فك جميع المخرجات إلى توبل بواسطة:

::

    abc = func() # دالة تُرجع توبل
    a,b = func()

بيان ``getattr``
^^^^^^^^^^^^^^^

::

    getattr_stmt ::= getattr(object, name[, default])

* يجب أن يكون اسم الخاصية سلسلة ثابتة.
* لا يتم دعم نوع الكائن النمطي (مثل torch._C).
* لا يتم دعم كائن الفئة المخصصة (مثل torch.classes.*).

بيان ``hasattr``
^^^^^^^^^^^^^^^^^

::

    hasattr_stmt ::= hasattr(object, name)

* يجب أن يكون اسم الخاصية سلسلة ثابتة.
* لا يتم دعم نوع الكائن النمطي (مثل torch._C).
* لا يتم دعم كائن الفئة المخصصة (مثل torch.classes.*).

بيان ``zip``
^^^^^^^^^^^^

::

    zip_stmt ::= zip(iterable1, iterable2)

* يجب أن تكون الوسائط قابلة للتكرار.
* يتم دعم وسائط قابلة للتكرار من نفس نوع الحاوية الخارجية ولكن بأطوال مختلفة.

**المثال 1**

يجب أن يكون كلا الوسيطين من نفس نوع الحاوية:

.. testcode::

    a = [1, 2] # قائمة
    b = [2, 3, 4] # قائمة
    zip(a, b) # يعمل

**المثال 2**

يفشل هذا المثال لأن الوسائط من أنواع حاويات مختلفة:

::

    a = (1, 2) # توبل
    b = [2, 3, 4] # قائمة
    zip(a, b) # خطأ وقت التشغيل

تشغيل الكود أعلاه ينتج عنه خطأ ``RuntimeError`` التالي.

::

    RuntimeError: لا يمكن التكرار على قائمة الوحدات النمطية أو
        توبل بقيمة ليس لها طول محدد بشكل ثابت.

**المثال 3**

يتم دعم وسيطين قابلين للتكرار من نفس نوع الحاوية ولكن من نوع بيانات مختلف:

.. testcode::

    a = [1.3, 2.4]
    b = [2, 3, 4]
    zip(a, b) # يعمل

تتضمن الأنواع القابلة للتكرار في TorchScript ``Tensors`` و ``lists`` و ``tuples`` و ``dictionaries`` و ``strings`` و ``torch.nn.ModuleList`` و ``torch.nn.ModuleDict``.

بيان ``enumerate``
^^^^^^^^^^^^^^^^^^
::

    enumerate_stmt ::= enumerate([iterable])

* يجب أن تكون الحجج قابلة للتحديد.
* تتضمن أنواع القابلة للتحديد في TorchScript ``Tensors`` و ``lists`` و ``tuples`` و ``dictionaries`` و ``strings`` و ``torch.nn.ModuleList`` و ``torch.nn.ModuleDict``.

.. _python-values-torch-script:

القيم في بايثون
~~~~~~~~~~~

.. _python-builtin-functions-values-resolution:

قواعد الحل
^^^^^^^^^
عند إعطاء قيمة بايثون، يحاول TorchScript حلها بالطرق الخمس المختلفة التالية:

* التنفيذ القابل للتجميع في بايثون:
    * عندما تكون قيمة بايثون مدعومة بواسطة تنفيذ بايثون الذي يمكن تجميعه بواسطة TorchScript، يقوم TorchScript بتجميع وتنفيذ التنفيذ الأساسي في بايثون.
    * مثال: ``torch.jit.Attribute``
* غلاف عملية بايثون:
    * عندما تكون قيمة بايثون عبارة عن غلاف لعملية بايثون أصلية، يقوم TorchScript بإصدار المشغل المقابل.
    * مثال: ``torch.jit._logging.add_stat_value``
* مطابقة هوية كائن بايثون:
    * لمجموعة محدودة من مكالمات واجهة برمجة التطبيقات ``torch.*`` (على شكل قيم بايثون) التي يدعمها TorchScript، يحاول TorchScript مطابقة قيمة بايثون مع كل عنصر في المجموعة.
    * عند المطابقة، يقوم TorchScript بتوليد مثيل ``SugaredValue`` المقابل الذي يحتوي على منطق خفض لهذه القيم.
    * مثال: ``torch.jit.isinstance()``
* مطابقة الاسم:
    * بالنسبة لوظائف بايثون المدمجة والثوابت، يقوم TorchScript بتحديدها حسب الاسم، وينشئ مثيل ``SugaredValue`` المقابل الذي ينفذ وظائفه.
    * مثال: ``all()``
* لقطة القيمة:
    * بالنسبة لقيم بايثون من وحدات غير معترف بها، يحاول TorchScript أخذ لقطة للقيمة وتحويلها إلى ثابت في رسم وظيفة (وظائف) أو طريقة (طرق) يتم تجميعها.
    * مثال: ``math.pi``

.. _python-builtin-functions-support:

دعم وظائف بايثون المدمجة
^^^^^^^^^^^^^^^^^^^
.. list-table:: دعم TorchScript لوظائف بايثون المدمجة
   :widths: 25 25 50
   :header-rows: 1

   * - الوظيفة المدمجة
     - مستوى الدعم
     - ملاحظات
   * - ``abs()``
     - جزئي
     - يدعم فقط إدخالات من نوع ``Tensor``/``Int``/``Float``. | لا يحترم التجاوز ``__abs__``.
   * - ``all()``
     - كامل
     -
   * - ``any()``
     - كامل
     -
   * - ``ascii()``
     - لا يوجد
     -
   * - ``bin()``
     - جزئي
     - يدعم فقط إدخال من نوع ``Int``.
   * - ``bool()``
     - جزئي
     - يدعم فقط إدخالات من نوع ``Tensor``/``Int``/``Float``.
   * - ``breakpoint()``
     - لا يوجد
     -
   * - ``bytearray()``
     - لا يوجد
     -
   * - ``bytes()``
     - لا يوجد
     -
   * - ``callable()``
     - لا يوجد
     -
   * - ``chr()``
     - جزئي
     - مجموعة الأحرف المدعومة الوحيدة هي ASCII.
   * - ``classmethod()``
     - كامل
     -
   * - ``compile()``
     - لا يوجد
     -
   * - ``complex()``
     - لا يوجد
     -
   * - ``delattr()``
     - لا يوجد
     -
   * - ``dict()``
     - كامل
     -
   * - ``dir()``
     - لا يوجد
     -
   * - ``divmod()``
     - كامل
     -
   * - ``enumerate()``
     - كامل
     -
   * - ``eval()``
     - لا يوجد
     -
   * - ``exec()``
     - لا يوجد
     -
   * - ``filter()``
     - لا يوجد
     -
   * - ``float()``
     - جزئي
     - لا يحترم التجاوز ``__index__``.
   * - ``format()``
     - جزئي
     - مواصفات الفهرس اليدوي غير مدعومة. | نوع التنسيق المعدل غير مدعوم.
   * - ``frozenset()``
     - لا يوجد
     -
   * - ``getattr()``
     - جزئي
     - يجب أن يكون اسم السمة حرفًا ثابتًا.
   * - ``globals()``
     - لا يوجد
     -
   * - ``hasattr()``
     - جزئي
     - يجب أن يكون اسم السمة حرفًا ثابتًا.
   * - ``hash()``
     - كامل
     - يتم حساب هاش "Tensor" بناءً على الهوية وليس القيمة العددية.
   * - ``hex()``
     - جزئي
     - يدعم فقط إدخال من نوع ``Int``.
   * - ``id()``
     - كامل
     - يدعم فقط إدخال من نوع ``Int``.
   * - ``input()``
     - لا يوجد
     -
   * - ``int()``
     - جزئي
     - حجة "base" غير مدعومة. | لا يحترم التجاوز ``__index__``.
   * - ``isinstance()``
     - كامل
     - توفر ``torch.jit.isintance`` دعمًا أفضل عند التحقق من أنواع الحاويات مثل ``Dict[str, int]``.
   * - ``issubclass()``
     - لا يوجد
     -
   * - ``iter()``
     - لا يوجد
     -
   * - ``len()``
     - كامل
     -
   * - ``list()``
     - كامل
     -
   * - ``ord()``
     - جزئي
     - مجموعة الأحرف المدعومة الوحيدة هي ASCII.
   * - ``pow()``
     - كامل
     -
   * - ``print()``
     - جزئي
     - حجج "separate" و "end" و "file" غير مدعومة.
   * - ``property()``
     - لا يوجد
     -
   * - ``range()``
     - كامل
     -
   * - ``repr()``
     - لا يوجد
     -
   * - ``reversed()``
     - لا يوجد
     -
   * - ``round()``
     - جزئي
     - حجة "ndigits" غير مدعومة.
   * - ``set()``
     - لا يوجد
     -
   * - ``setattr()``
     - لا يوجد
     -
   * - ``slice()``
     - كامل
     -
   * - ``sorted()``
     - جزئي
     - حجة "key" غير مدعومة.
   * - ``staticmethod()``
     - كامل
     -
   * - ``str()``
     - جزئي
     - حجج "encoding" و "errors" غير مدعومة.
   * - ``sum()``
     - كامل
     -
   * - ``super()``
     - جزئي
     - يمكن استخدامه فقط في طريقة ``__init__`` الخاصة بـ ``nn.Module``.
   * - ``type()``
     - لا يوجد
     -
   * - ``vars()``
     - لا يوجد
     -
   * - ``zip()``
     - كامل
     -
   * - ``__import__()``
     - لا يوجد
     -

.. _python-builtin-values-support:

دعم قيم بايثون المدمجة
^^^^^^^^^^^^^^
.. list-table:: دعم TorchScript لقيم بايثون المدمجة
   :widths: 25 25 50
   :header-rows: 1

   * - القيمة المدمجة
     - مستوى الدعم
     - ملاحظات
   * - ``False``
     - كامل
     -
   * - ``True``
     - كامل
     -
   * - ``None``
     - كامل
     -
   * - ``NotImplemented``
     - لا يوجد
     -
   * - ``Ellipsis``
     - كامل
     -

.. _torch_apis_in_torchscript:

واجهات برمجة التطبيقات في Torch.*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _torch_apis_in_torchscript_rpc:

استدعاءات الإجراءات البعيدة
^^^^^^^^^^^^^^^^^^^

يدعم TorchScript مجموعة فرعية من واجهات برمجة تطبيقات RPC التي تدعم تشغيل وظيفة على
عامل بعيد محدد بدلاً من تشغيلها محليًا.

على وجه التحديد، يتم دعم واجهات برمجة التطبيقات التالية بشكل كامل:

- ``torch.distributed.rpc.rpc_sync()``
    - يقوم ``rpc_sync()`` باستدعاء RPC حجب إلى عامل بعيد لتشغيل وظيفة. يتم إرسال رسائل RPC واستلامها بشكل متواز مع تنفيذ كود بايثون.
    - يمكن العثور على مزيد من التفاصيل حول استخدامه وأمثلة في: meth: ~ torch.distributed.rpc.rpc_sync.

- ``torch.distributed.rpc.rpc_async()``
    - يقوم ``rpc_async()`` باستدعاء RPC غير حجب لتشغيل وظيفة على عامل بعيد. يتم إرسال رسائل RPC واستلامها بشكل متواز مع تنفيذ كود بايثون.
    - يمكن العثور على مزيد من التفاصيل حول استخدامه وأمثلة في: meth: ~ torch.distributed.rpc.rpc_async.
- ``torch.distributed.rpc.remote()``
    - ينفذ ``remote.()`` استدعاءًا بعيدًا على عامل ويحصل على مرجع بعيد ``RRef`` كقيمة الإرجاع.
    - يمكن العثور على مزيد من التفاصيل حول استخدامه وأمثلة في: meth: ~ torch.distributed.rpc.remote.

.. _torch_apis_in_torchscript_async:

التنفيذ غير المتزامر
^^^^^^^^^^^^^^

يمكّنك TorchScript من إنشاء مهام حسابية غير متزامرة للاستفادة بشكل أفضل
من موارد الحساب. يتم ذلك من خلال دعم قائمة من واجهات برمجة التطبيقات التي يمكن
استخدامها فقط داخل TorchScript:

- ``torch.jit.fork()``
    - يقوم بإنشاء مهمة غير متزامرة لتنفيذ دالة وإرجاع مرجع إلى نتيجة هذا التنفيذ. وسوف يعود الشوكة فورا.
    - مرادف لـ ``torch.jit._fork()``، والذي يتم الاحتفاظ به فقط لأسباب التوافق مع الإصدارات السابقة.
    - يمكن العثور على مزيد من التفاصيل حول استخدامه وأمثلة في: meth: ~ torch.jit.fork.
- ``torch.jit.wait()``
    - يجبر استكمال مهمة ``torch.jit.Future[T]`` غير المتزامرة، وإرجاع نتيجة المهمة.
    - مرادف لـ ``torch.jit._wait()``، والذي يتم الاحتفاظ به فقط لأسباب التوافق مع الإصدارات السابقة.
    - يمكن العثور على مزيد من التفاصيل حول استخدامه وأمثلة في: meth: ~ torch.jit.wait.

.. _torch_apis_in_torchscript_annotation:

ملاحظات النوع
^^^^^^^^^^^^^^^^

TorchScript ثابت من حيث النوع. يوفر ويدعم مجموعة من المرافق للمساعدة في الإشارة إلى المتغيرات والسمات:

- ``torch.jit.annotate()``
    - يوفر تلميحًا للنوع إلى TorchScript حيث لا تعمل تلميحات النوع على طريقة بايثون 3 بشكل جيد.
    - أحد الأمثلة الشائعة هو الإشارة إلى نوع للتعبيرات مثل ``[]``. يتم التعامل مع ``[]`` على أنها ``List[torch.Tensor]`` بشكل افتراضي. عندما تكون هناك حاجة إلى نوع مختلف، يمكنك استخدام هذا الرمز للإشارة إلى TorchScript: ``torch.jit.annotate(List[int]، [])``.
    - يمكن العثور على مزيد من التفاصيل في: meth: ~ torch.jit.annotate
- ``torch.jit.Attribute``
    - تشمل حالات الاستخدام الشائعة توفير تلميح للنوع لسمات ``torch.nn.Module`` نظرًا لأن طرق ``__init__`` الخاصة بها لا يتم تحليلها بواسطة TorchScript، يجب استخدام ``torch.jit.Attribute`` بدلاً من ``torch.jit.annotate`` في طرق ``__init__`` الخاصة بالوحدة النمطية.
    - يمكن العثور على مزيد من التفاصيل في: meth: ~ torch.jit.Attribute
- ``torch.jit.Final``
    - مرادف لـ ``typing.Final`` في بايثون. يتم الاحتفاظ بـ ``torch.jit.Final`` فقط لأسباب التوافق مع الإصدارات السابقة.

.. _torch_apis_in_torchscript_meta_programming:

البرمجة الميتا
^^^^^^^^^^^

يوفر TorchScript مجموعة من المرافق لتسهيل البرمجة الميتا:

- ``torch.jit.is_scripting()``
    - يعيد قيمة منطقية تشير إلى ما إذا كان البرنامج الحالي مجمعًا بواسطة ``torch.jit.script`` أم لا.
    - عندما يتم استخدامه في عبارة تأكيد أو عبارة "if"، فإن النطاق أو الفرع الذي يتم فيه تقييم ``torch.jit.is_scripting()`` إلى "False" لا يتم تجميعه.
    - يمكن تقييم قيمته بشكل ثابت في وقت التجميع، وبالتالي يتم استخدامه بشكل شائع في عبارات "if" لوقف TorchScript من تجميع أحد الفروع.
    - يمكن العثور على مزيد من التفاصيل والأمثلة في: meth: ~ torch.jit.is_scripting
- ``torch.jit.is_tracing()``
    - يعيد قيمة منطقية تشير إلى ما إذا كان البرنامج الحالي يتم تتبعه بواسطة ``torch.jit.trace`` / ``torch.jit.trace_module`` أم لا.
    - يمكن العثور على مزيد من التفاصيل في: meth: ~ torch.jit.is_tracing
- ``@torch.jit.ignore``
    - يشير هذا الديكور إلى المترجم بأن الوظيفة أو الطريقة يجب تجاهلها وتركها كدالة بايثون.
    - يسمح لك ذلك بترك التعليمات البرمجية في نموذجك والتي لا تتوافق مع TorchScript بعد.
    - إذا تم استدعاء دالة مزينة بـ ``@torch.jit.ignore`` من TorchScript، فسيتم إرسال الاستدعاءات المهملة إلى مفسر بايثون.
    - لا يمكن تصدير النماذج التي تحتوي على وظائف مهملة.
    - يمكن العثور على مزيد من التفاصيل والأمثلة في: meth: ~ torch.jit.ignore
- ``@torch.jit.unused``
    - يشير هذا الديكور إلى المترجم بأن الوظيفة أو الطريقة يجب تجاهلها واستبدالها برمي استثناء.
    - يسمح لك ذلك بترك التعليمات البرمجية في نموذجك والتي لا تتوافق مع TorchScript بعد، ولا يزال يمكنك تصدير نموذجك.
    - إذا تم استدعاء دالة مزينة بـ ``@torch.jit.unused`` من TorchScript، فسيتم إلقاء خطأ وقت التشغيل.
    - يمكن العثور على مزيد من التفاصيل والأمثلة في: meth: ~ torch.jit.unused

.. _torch_apis_in_torchscript_type_refinement:

تنقيح النوع
^^^^^^^^

- ``torch.jit.isinstance()``
    - يعيد قيمة منطقية تشير إلى ما إذا كانت المتغير من النوع المحدد.
    - يمكن العثور على مزيد من التفاصيل حول استخدامه وأمثلة في: meth: ~ torch.jit.isinstance.