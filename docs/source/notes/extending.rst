.. _extending-torch:

توسيع PyTorch
=============

تعد قابلية التوسعة إحدى الميزات الرئيسية في PyTorch. يمكنك تخصيص كل شيء تقريبًا في PyTorch، بدءًا من إضافة عمليات جديدة إلى :ref:`extending-torch-tensor` وحتى تخصيص :ref:`autograd engine <extending-torch-autograd>` أو كتابة :ref:`your own C++ extension <extending-torch-c++extension>`.

.. toctree::
   :maxdepth: 1

   تمديد تنسور <tensor_extension>
   توسيع الإطار الآلي <autograd_extension>
   كتابة امتداد C ++ الخاص بك <cpp_extension>

.. _extending-torch-tensor:

توسيع تنسور
------------

يمكنك بسهولة إضافة عمليات جديدة إلى تنسور PyTorch. على سبيل المثال، يمكنك إضافة عملية "add_one_tensor" التي تضيف 1 إلى تنسور كما يلي:

.. code:: python

   import torch

   class MyTensor(torch.Tensor):
       def __new__(cls, *args, **kwargs):
           tensor = torch.Tensor(*args, **kwargs)
           return super(MyTensor, cls).__new__(cls, tensor)

       def add_one(self):
           return self + 1

يمكنك الآن استخدام عملية "add_one" على أي كائن "MyTensor":

.. code:: python

   >>> tensor = MyTensor([1, 2, 3])
   >>> tensor.add_one()
   tensor([2, 3, 4])

.. _extending-torch-autograd:

توسيع الإطار الآلي
------------------

يتيح لك PyTorch أيضًا تخصيص محرك الإطار الآلي. على سبيل المثال، يمكنك كتابة عملية "add_custom" الخاصة بك والتي تضيف تنسورين معًا مع تخصيص عملية الإطار الآلي:

.. code:: python

   import torch
   from torch.autograd import Function

   class AddCustom(Function):
       @staticmethod
       def forward(ctx, x, y):
           ctx.save_for_backward(x, y)
           return x + y

       @staticmethod
       def backward(ctx, grad_output):
           x, y = ctx.saved_tensors
           return grad_output, grad_output

يمكنك الآن استخدام عملية "add_custom" في حساباتك، وسيتم تتبع تدرجها بشكل صحيح:

.. code:: python

   >>> x = torch.tensor([1, 2, 3], requires_grad=True)
   >>> y = torch.tensor([4, 5, 6], requires_grad=True)
   >>> z = AddCustom.apply(x, y)
   >>> z.backward()
   >>> x.grad
   tensor([1, 1, 1])
   >>> y.grad
   tensor([1, 1, 1])

.. _extending-torch-c++extension:

كتابة امتداد C ++ الخاص بك
------------------------

في بعض الأحيان، قد تحتاج إلى كتابة امتداد C ++ الخاص بك لتحسين الأداء أو للاستفادة من مكتبات C ++ الخارجية. يوفر PyTorch أدوات وطبقات تجريدية لجعل هذه العملية سهلة قدر الإمكان.

يمكنك العثور على دليل تفصيلي حول كيفية كتابة امتداد C ++ الخاص بك في :ref:`إرشادات امتداد C ++ <extending-torch-c++-guide>`.

للحصول على أمثلة عملية، يمكنك الاطلاع على الأمثلة الموجودة في مستودع PyTorch على GitHub: `أمثلة امتداد C ++ <https://github.com/pytorch/pytorch/tree/master/cpp_extensions>`_.
هذا هو النص المترجم إلى اللغة العربية بتنسيق ReStructuredText:

=================

في هذه الملاحظة، سنغطي طرق تمديد :mod:`torch.nn`،
:mod:`torch.autograd`، :mod:`torch`، وكتابة امتدادات C++ مخصصة.

إضافة مشغلات جديدة
---------------------

يوفر PyTorch مكتبة كبيرة من المشغلات التي تعمل على الصّناديق (Tensors) (مثل :func:`torch.add`،
:func:`torch.sum`، إلخ). ومع ذلك، قد ترغب في جلب عملية مخصصة جديدة إلى PyTorch
وجعلها تتصرف مثل المشغلات المدمجة في PyTorch. للقيام بذلك، يجب عليك
تسجيل العملية المخصصة مع PyTorch عبر واجهات برمجة التطبيقات (APIs) لـ Python :ref:`torch-library-docs` أو C++ TORCH_LIBRARY.

يرجى الاطلاع على :ref:`custom-ops-landing-page` لمزيد من التفاصيل.

.. _extending-autograd:

تمديد :mod:`torch.autograd`
-------------------------------

.. currentmodule:: torch.autograd

يتطلب إضافة عمليات إلى: mod: ~ torch.autograd تنفيذ فئة فرعية جديدة: class: ~ Function لكل عملية. تذكر أن الوظائف هي ما: mod: ~ torch.autograd يستخدم لتشفير تاريخ العملية وحساب التدرجات.

يركز الجزء الأول من هذه الوثيقة على وضع AD الخلفي لأنه الميزة الأكثر استخدامًا على نطاق واسع. ويناقش قسم في النهاية ملحقات وضع AD الأمامي.

متى تستخدم
^^^^^^^^^^^

بشكل عام، قم بتنفيذ دالة مخصصة إذا كنت تريد إجراء حسابات في نموذجك
التي ليست قابلة للتفاضل أو تعتمد على مكتبات غير PyTorch (مثل NumPy)، ولكن
ما زلت ترغب في تشغيل عملية مع العمليات الأخرى والعمل مع محرك autograd.

في بعض الحالات، يمكن أيضًا استخدام الوظائف المخصصة لتحسين الأداء
استخدام الذاكرة: إذا قمت بتنفيذ التمريرات الأمامية والخلفية باستخدام
`امتداد C ++ <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_،
يمكنك لفها في: class: ~ Function للتواصل مع محرك autograd. إذا كنت ترغب في تقليل عدد المؤشرات العكسية المخزنة للتمرير الخلفي،
يمكن استخدام الوظائف المخصصة لدمج العمليات معًا.

متى لا تستخدم
^^^^^^^^^^^^^^^

إذا كنت تستطيع بالفعل كتابة دالتك من حيث عمليات PyTorch المدمجة، فإنها
الرسم البياني الخلفي (على الأرجح) قادر بالفعل على تسجيله بواسطة autograd. في هذه الحالة، أنت لا
لا تحتاج إلى تنفيذ الدالة الخلفية بنفسك. فكر في استخدام وظيفة Python عادية.

إذا كنت بحاجة إلى الاحتفاظ بحالة، أي معلمات قابلة للتدريب، فيجب عليك (أيضًا) استخدام
وحدة نمطية مخصصة. راجع القسم أدناه للحصول على مزيد من المعلومات حول تمديد: mod: ~ torch.nn.

إذا كنت ترغب في تعديل التدرجات أثناء التمرير الخلفي أو إجراء تأثير جانبي، ففكر في التسجيل
`tensor <https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch.Tensor.register_hook>`_ أو
`Module <https://pytorch.org/docs/stable/notes/modules.html#module-hooks>`_ hook.

كيفية الاستخدام
^^^^^^^^^^

اتبع الخطوات التالية:

1. قم بإنشاء فئة فرعية من: class: ~ Function وقم بتنفيذ: meth: ~ Function.forward،
(اختياري) : meth: ~ Function.setup_context و
: meth: ~ Function.backward الأساليب.

2. استدعاء الأساليب الصحيحة على الحجة "ctx".

3. أعلن ما إذا كانت دالتك تدعم
`التدرج المزدوج <https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>`_.

4. تحقق مما إذا كانت تدرجاتك صحيحة باستخدام gradcheck.

**الخطوة 1:** بعد إنشاء فئة فرعية من: class: ~ Function، ستحتاج إلى تحديد 3 طرق:

- : meth: ~ Function.forward هو الكود الذي يؤدي العملية. يمكن أن يستغرق
العديد من الحجج كما تريد، مع بعضها اختياري، إذا كنت
تحديد القيم الافتراضية. يتم قبول جميع أنواع كائنات Python هنا.
: class: ~ Tensor الحجج التي تتتبع التاريخ (أي مع
"requires_grad = True") سيتم تحويلها إلى تلك التي لا تتتبع التاريخ
قبل الاستدعاء، وسيتم تسجيل استخدامها في الرسم البياني. لاحظ أن هذا
منطق لن يقوم بالتنقل عبر القوائم / القواميس / أي هياكل بيانات أخرى وسوف ينظر فقط في الموترات التي تكون حججًا مباشرة للاستدعاء. يمكنك
إرجاع إما: class: ~ Tensor الإخراج الفردي، أو: class: ~ tuple من
موترات إذا كان هناك عدة مخارج. أيضًا، يرجى الرجوع إلى
وثائق: class: ~ Function للعثور على أوصاف الأساليب المفيدة التي يمكن أن تكون
يتم استدعاؤها فقط من: meth: ~ Function.forward.

- : meth: ~ Function.setup_context (اختياري). يمكن للمرء إما كتابة "المدمجة" : meth: ~ Function.forward التي
  يقبل كائن "ctx" أو (اعتبارًا من PyTorch 2.0) : meth: ~ Function.forward منفصل لا
  لا تقبل "ctx" و: meth: ~ Function.setup_context طريقة حيث يحدث تعديل "ctx".
  يجب أن يحتوي : meth: ~ Function.forward على الحساب و: meth: ~ Function.setup_context يجب
  أن تكون مسؤولة فقط عن تعديل "ctx" (وليس لها أي حساب).
  بشكل عام، فإن : meth: ~ Function.forward و: meth: ~ Function.setup_context المنفصلين أقرب إلى كيفية
  تعمل عمليات PyTorch الأصلية وبالتالي فهي أكثر قابلية للتركيب مع العديد من الأنظمة الفرعية لـ PyTorch.
  راجع : ref: `combining-forward-context` لمزيد من التفاصيل.

- : meth: ~ Function.backward (أو: meth: ~ Function.vjp) تحدد صيغة التدرج.
  سيكون لديه العديد من: class: ~ Tensor الحجج كما كانت هناك مخارج، مع كل
  منها يمثل التدرج فيما يتعلق بهذا الإخراج. من المهم ألا تعدل أبدًا
  هذه في المكان. يجب أن يعيد العديد من الموترات كما كان هناك
  كانت المدخلات، مع كل منها يحتوي على التدرج فيما يتعلق به
  المدخلات المقابلة. إذا لم تكن مدخلاتك تتطلب التدرج
  : attr: ~ ctx.needs_input_grad عبارة عن مجموعة من القيم المنطقية التي تشير إلى
  ما إذا كان كل إدخال يحتاج إلى حساب التدرج)، أو كانت كائنات غير Tensor،
  يمكنك إرجاع: class: ~ python: None. أيضًا، إذا كان لديك حجج اختيارية إلى: meth: ~ Function.forward يمكنك إرجاع المزيد من التدرجات أكثر من المدخلات، طالما أنها
جميع: any: ~ python: None.

**الخطوة 2:** من مسؤوليتكم استخدام الوظائف في "ctx"
بالشكل الصحيح لضمان عمل الدالة الجديدة بشكل صحيح مع
محرك autograd.

- : meth: ~ torch.autograd.function.FunctionCtx.save_for_backward يجب
  تستخدم لحفظ أي موترات لاستخدامها في التمرير الخلفي. يجب تخزين الكائنات غير الموتر مباشرة على "ctx". إذا تم حفظ الموترات التي ليست مدخلات أو مخرجات
  للتمرير الخلفي، فقد لا تدعم دالتك التمرير المزدوج
  انظر الخطوة 3).

- : meth: ~ torch.autograd.function.FunctionCtx.mark_dirty يجب استخدامها
  قم بوضع علامة على أي إدخال يتم تعديله في مكانه بواسطة الدالة الأمامية.

- : meth: ~ torch.autograd.function.FunctionCtx.mark_non_differentiable يجب
  يتم استخدامه لإخبار المحرك إذا كان الإخراج غير قابل للتفاضل. حسب
  الافتراضي، يتم تعيين جميع موترات الإخراج من النوع القابل للتفاضل
  لطلب التدرج. لا يتم أبدًا وضع علامة على الموترات من النوع غير القابل للتفاضل (أي الأنواع الصحيحة) كموترات تتطلب التدرجات.

- : meth: ~ torch.autograd.function.FunctionCtx.set_materialize_grads يمكن
  يتم استخدامه لإخبار محرك autograd بتحسين حسابات التدرج في الحالات التي
  لا يعتمد الإخراج على الإدخال عن طريق عدم مادة موترات التدرج المعطاة للدالة الخلفية. أي، إذا تم تعيينه إلى False، فسيتم تحويل كائن None في Python أو "tensor غير محدد" (tensor x لـ
  الذي x.defined () هو كاذب) في C ++ إلى موتر مملوء بالأصفار قبل استدعاء الخلفي، لذلك يجب أن تتمكن التعليمات البرمجية من التعامل مع مثل هذه الكائنات كما لو كانت
  الموترات المملوءة بالأصفار. القيمة الافتراضية لهذا الإعداد هي True.

**الخطوة 3:** إذا لم تكن دالتك: class: ~ Function تدعم التمرير المزدوج
يجب عليك الإعلان عن ذلك صراحةً عن طريق تزيين الخلفي باستخدام
: func: ~ function.once_differentiable. مع هذا الديكور، ستؤدي محاولات
أداء التمرير المزدوج عبر دالتك إلى حدوث خطأ.
راجع تعليماتنا التفصيلية حول التمرير المزدوج للحصول على مزيد من المعلومات حول التمرير المزدوج.

**الخطوة 4:** يُنصح باستخدام: func: ~ torch.autograd.gradcheck
لتحقق مما إذا كانت دالتك الخلفية تحسب التدرجات بشكل صحيح للأمام عن طريق حساب المصفوفة جاكوبيانية باستخدام دالتك الخلفية
ومقارنة القيمة عنصرًا تلو الآخر مع جاكوبي المحسوب عدديًا باستخدام
التفاضل المحدود.

مثال
^^^^^^^

فيما يلي يمكنك العثور على التعليمات البرمجية لـ ``Linear`` function، مع
تعليقات إضافية::

    # ورث من الدالة
    class LinearFunction(Function):

        # لاحظ أن forward و setup_context و backward عبارة عن @staticmethods
        @staticmethod
        def forward(input، weight، bias):
            output = input.mm (weight.t ())
            إذا كان التحيز غير موجود:
                الإخراج += bias.unsqueeze (0).expand_as (الإخراج)
            return output

        @staticmethod
        # المدخلات عبارة عن مجموعة من جميع المدخلات التي تم تمريرها إلى الأمام.
        # الإخراج هو إخراج forward ().
        def setup_context (ctx، inputs، output):
            input، weight، bias = inputs
            ctx.save_for_backward (input، weight، bias)

        # تحتوي هذه الدالة على إخراج واحد فقط، لذا فهي تحصل على تدرج واحد فقط
        @staticmethod
        def backward (ctx، grad_output):
            # هذا نمط مريح للغاية - في الجزء العلوي من الخلفي
            # فك حزم saved_tensors وتهيئة جميع التدرجات فيما يتعلق بالمدخلات إلى
            # لا شيء. بفضل حقيقة أن إضافات Nones الإضافية
            # يتم تجاهلها، يكون بيان الإرجاع بسيطًا حتى عندما يكون للدالة مدخلات اختيارية.
            input، weight، bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            # هذه عمليات التحقق من needs_input_grad اختيارية وهناك فقط لتحسين الكفاءة. إذا كنت تريد جعل التعليمات البرمجية الخاصة بك أبسط، يمكنك
            # تخطيها. إرجاع التدرجات للمدخلات التي لا تتطلبها ليس خطأ.
            إذا ctx.needs_input_grad [0]:
                grad_input = grad_output.mm (الوزن)
            إذا ctx.needs_input_grad [1]:
                grad_weight = grad_output.t (). mm (الإدخال)
            إذا كان التحيز موجودًا و ctx.needs_input_grad [2]:
                grad_bias = grad_output.sum (0)

            return grad_input، grad_weight، grad_bias

الآن، لتسهيل استخدام هذه العمليات المخصصة، نوصي إما بتعيين اسم مستعار
أو لفها في دالة. يسمح التغليف في دالة بدعم الافتراضي
الحجج وحجج الكلمات الرئيسية::

    # الخيار 1: الاسم المستعار
    الخطي = LinearFunction.apply

    # الخيار 2: لف في دالة، لدعم الافتراضي وحجج الكلمات الرئيسية.
    def linear (الإدخال، الوزن، التحيز = None):
        return LinearFunction.apply (الإدخال، الوزن، التحيز)

هنا، نقدم مثالًا إضافيًا على دالة يتم تحديدها بواسطة
حجج غير موترة::

    class MulConstant (الدالة):
        @staticmethod
        def forward (tensor، constant):
            return tensor * constant

        @staticmethod
        def setup_context (ctx، inputs، output):
            # ctx هو كائن سياق يمكن استخدامه لتخزين المعلومات
            # للحساب الخلفي
            tensor، constant = inputs
            ctx.constant = constant

        @staticmethod
        def backward (ctx، grad_output):
            # نعيد عددًا من تدرجات الإدخال مثل عدد الحجج.
            # يجب أن تكون تدرجات الحجج غير الموترة إلى الأمام None.
            return grad_output * ctx.constant، None

وهنا، نقوم بتحسين المثال أعلاه عن طريق استدعاء set_materialize_grads (False)::

    class MulConstant (الدالة):
        @staticmethod
        def forward (tensor، constant):
            return tensor * constant

        @staticmethod
        def setup_context (ctx، inputs، output):
            tensor، constant = inputs
            ctx.set_materialize_grads (False)
            ctx.constant = constant

        @staticmethod
        def backward (ctx، grad_output):
            # هنا يجب علينا التعامل مع كائن grad_output None. في هذه الحالة يمكننا
            # تخطي الحسابات غير الضرورية والعودة ببساطة None.
            إذا كان grad_output عبارة عن كائن None:
                return None، None

            # نعيد عددًا من تدرجات الإدخال مثل عدد الحجج.
            # يجب أن تكون تدرجات الحجج غير الموترة إلى الأمام None.
            return grad_output * ctx.constant، None

إذا كنت بحاجة إلى أي موترات "متوسطة" محسوبة في: meth: ~ Function.forward ليتم حفظها،
إما يجب إرجاعها كمخرجات، أو دمج "forward" و: meth: ~ Function.setup_context
(راجع : ref: `combining-forward-context`).
لاحظ أن هذا يعني أنه إذا كنت تريد تدفق التدرجات عبر تلك القيم المتوسطة، فأنت
تحتاج إلى تحديد صيغة التدرج لها (راجع أيضًا
`تعليمات التمرير المزدوج التفصيلية <https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>`_
)::

    class MyCube (torch.autograd.Function):
        @staticmethod
        def forward (x):
            # نريد حفظ dx للتمرير الخلفي. للقيام بذلك، يجب
            # يتم إرجاعه كمخرج.
            dx = 3 * x ** 2
            النتيجة = x ** 3
            return result، dx

        @staticmethod
        def setup_context (ctx، inputs، output):
            x، = المدخلات
            النتيجة، dx = الإخراج
            ctx.save_for_backward (x، dx)
هذا هو النص المترجم إلى اللغة العربية بتنسيق ReStructuredText:

@staticmethod
def backward(ctx, grad_output, grad_dx):
    """
    يحسب المشتق العكسي للدالة المكعبة.

    المعاملات:
    - ctx: كائن السياق الذي يحتوي على البيانات المخزنة أثناء عملية التقديم.
    - grad_output: المشتق العكسي للإخراج.
    - grad_dx: المشتق العكسي لـ dx.

    المخرجات:
    - النتيجة: المشتق العكسي للدالة المكعبة.
    """
    x, dx = ctx.saved_tensors
    # لكي يعمل الدالة مع المشتقات من الرتب العليا، يجب أن نضيف مساهمة المشتق
    # والتي تساوي grad_dx * 6 * x.
    النتيجة = grad_output * dx + grad_dx * 6 * x
    return النتيجة

# نغلف دالة MyCube في دالة بحيث يكون من الواضح ما هو الإخراج
def my_cube(x):
    """
    يحسب دالة التكعيب وتدرجها.

    المعاملات:
    - x: الإدخال إلى دالة التكعيب.

    المخرجات:
    - النتيجة: إخراج دالة التكعيب.
    - dx: تدرج دالة التكعيب.
    """
    النتيجة، dx = MyCube.apply(x)
    return النتيجة

.. note::
    يمكن أن تكون المدخلات إلى ``backward``، أي :attr: `grad_output`، أيضًا
    المنسوجات التي تتتبع التاريخ. لذا إذا تم تنفيذ "backward" باستخدام عمليات قابلة للاشتقاق
    (على سبيل المثال، استدعاء دالة مخصصة أخرى
    : class: `~ torch.autograd.Function`)، فإن المشتقات من الرتب العليا ستعمل.
    في هذه الحالة، يمكن أيضًا استخدام المنسوجات المحفوظة باستخدام ``save_for_backward``
    في الخلف ولها تدرجات تتدفق مرة أخرى ولكن المنسوجات المحفوظة في ``ctx``
    لن يكون لديهم تدرجات تتدفق مرة أخرى بالنسبة لهم.
    إذا كنت بحاجة إلى تدرجات للتدفق مرة أخرى لمنسوجة محفوظة في "ctx"، فيجب عليك
    جعله إخراجًا للدالة المخصصة واحفظه باستخدام ``save_for_backward``.

من المحتمل أنك تريد التحقق مما إذا كانت طريقة التقديم العكسي التي قمت بتنفيذها
تقوم بالفعل بحساب مشتقات دالتك. هذا ممكن من خلال المقارنة مع
التقريبات العددية باستخدام اختلافات محدودة صغيرة::

    من torch.autograd استيراد gradcheck

    # يأخذ gradcheck زوجًا من المنسوجات كإدخال، والتحقق مما إذا كان تدرجك
    # تم تقييمها بهذه المنسوجات قريبة بما فيه الكفاية من التقريبات العددية
    # ويعيد True إذا تحققت جميع هذه الشروط.
    المدخلات = (torch.randn(20، 20، dtype=torch.double، requires_grad=True)، torch.randn(30، 20، dtype=torch.double، requires_grad=True))
    الاختبار = gradcheck(linear، input، eps=1e-6، atol=1e-4)
    طباعة (اختبار)

راجع :ref: `grad-check` لمزيد من التفاصيل حول مقارنات تدرج الاختلافات المحدودة.
إذا تم استخدام دالتك في مشتقات من الرتب العليا (مشتقة من التمرير الخلفي) فيمكنك
استخدم دالة "gradgradcheck" من نفس الحزمة للتحقق من المشتقات من الرتب العليا.

.. _combining-forward-context:

الجمع بين :meth: `~ Function.forward` و: meth: `~ Function.setup_context`
أو فصلهما
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

هناك طريقتان رئيسيتان لتعريف : class: `~ Function`. إما:

- تعريف :meth: `~ Function.forward` الذي يجمع منطق الحساب الأمامي مع: meth: `~ Function.setup_context`
- (اعتبارًا من PyTorch 2.0) تعريف منفصل :meth: `~ Function.forward` و: meth: `~ Function.setup_context`

نوصي بالخيار الثاني (منفصل :meth: `~ Function.forward` و: meth: `~ Function.setup_context`)
لأنه أقرب إلى كيفية تنفيذ العمليات الأصلية PyTorch ويتألف
مع :mod: `torch.func` يحول. ومع ذلك، فإننا نخطط لدعم كلا النهجين في المستقبل؛
يؤدي الجمع بين: meth: `~ Function.forward` و: meth: `~ Function.setup_context`: إلى مزيد من المرونة
نظرًا لأنه يمكنك حفظ المتوسطات دون إعادتها كإخراج.

يرجى الاطلاع على القسم السابق للحصول على كيفية تعريف : class: `~ Function` مع منفصل
:meth: `~ Function.forward` و: meth: `~ Function.setup_context`.

فيما يلي مثال على كيفية تعريف : class: `Function` مع الجمع بين: meth: `~ Function.forward` و
:meth: `~ Function.setup_context`::

    class LinearFunction(Function):
        @staticmethod
        # ctx هو الحجة الأولى للأمام
        def forward(ctx، input، weight، bias=None):
            # يمكن للمرور الأمامي استخدام ctx.
            ctx.save_for_backward(input، weight، bias)
            الإخراج = input.mm(weight.t())
            إذا كان التحيز غير موجود:
                الإخراج += bias.unsqueeze(0).expand_as(output)
            return output

        @staticmethod
        def backward(ctx، grad_output):
            input، weight، bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            إذا كان ctx.needs_input_grad [0]:
                grad_input = grad_output.mm(weight)
            إذا كان ctx.needs_input_grad [1]:
                grad_weight = grad_output.t().mm(input)
            إذا كان التحيز موجودًا وكان ctx.needs_input_grad [2]:
                grad_bias = grad_output.sum(0)

            return grad_input، grad_weight، grad_bias

.. _forward-ad-autograd-function:

وضع AD الأمامي
^^^^^^^^^^^^^^^

يتمتع تجاوز صيغة AD الأمامية بمشابهة وثيقة جدًا لـ API مع بعض الاختلافات الدقيقة المختلفة.
يمكنك تنفيذ دالة :meth: `~ Function.jvp`.

سيتم منحه العديد من كائنات : class: `Tensor` كمعاملات مثل الإدخالات الموجودة، مع كل
تمثل تدرجًا بالنسبة إلى هذا الإدخال. يجب أن تعيد العديد من المنسوجات مثل هناك
كانت الإخراج، مع كل منها يحتوي على تدرج فيما يتعلق الإخراج المقابلة.
سيتم استدعاء :meth: `~ Function.jvp` مباشرة بعد :meth: `~ Function.forward`
الطريقة، قبل أن :meth: `~ Function.apply` العائدات.

:meth: `~ Function.jvp` لديه بعض الاختلافات الدقيقة مع :meth: `~ Function.backward`
دالة:

- يمكنك استخدام `ctx` لنقل أي بيانات من: meth: `~ Function.forward` إلى: meth: `~ Function.jvp`
  الدالة. إذا لم تكن هذه الحالة مطلوبة لـ :meth: `~ Function.backward`،
  يمكنك تحريرها بشكل صريح عن طريق القيام ``del ctx.foo`` في نهاية :meth: `~ Function.jvp`
  الدالة.
- يجب أن يكون تنفيذ :meth: `~ Function.jvp` قابلًا للاشتقاق العكسي أو التحقق صراحةً من أن
  لا يوجد لدى أي من تدرجات وضع التقديم الأمامي المحدد ``requires_grad`` المحدد.
- يجب أن تتطابق دالة :meth: `~ Function.jvp` مع سلوك العرض/في المكان لـ :meth: `~ Function.forward`.
  على سبيل المثال، إذا تم تعديل الإدخال "i" في المكان، فيجب تحديث تدرج "i" في المكان.
  وبالمثل، إذا كان الإخراج "j" هو عرض للإدخال "k". ثم يجب أن يكون الإخراج "j" المرتجع
  يكون الإخراج تدرجًا للعرض "k" المدرج في الإدخال.
- نظرًا لأنه لا يمكن للمستخدم تحديد التدرج الذي يجب حسابه، فيجب أن تقوم دالة :meth: `~ Function.jvp`
  دائمًا بحساب التدرجات لجميع الإخراج.
- تحترم تدرجات وضع التقديم الأمامي العلم الذي حدده :meth: `~ torch.autograd.function.FunctionCtx.set_materialize_grads`
  ويمكنك الحصول على تدرجات إدخال "None" عند تعطيل ذلك.

:mod: `torch.func` يحول و/أو: func: `torch.vmap`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يرجى الاطلاع على :ref: `func-autograd-function` لمزيد من التفاصيل.


توسيع :mod: `torch.nn`
-------------------------

.. currentmodule:: torch.nn

تقوم الوحدة النمطية :mod:`~torch.nn` بتصدير نوعين من الواجهات - الوحدات النمطية والنسخ الوظيفية الخاصة بها. يمكنك التوسع في كلا الاتجاهين، ولكن نوصي باستخدام الوحدات النمطية لجميع أنواع الطبقات التي تحتوي على أي معلمات أو مخازن مؤقتة، ونوصي باستخدام الشكل الوظيفي للعمليات عديمة المعلمات مثل وظائف التنشيط والتجميع، وما إلى ذلك.

تمت تغطية إضافة نسخة وظيفية من عملية بالفعل بشكل كامل في القسم أعلاه.

إضافة :class:`Module`
^^^^^^^^^^^^^^^^^^^^^^^^

نظرًا لأن :mod:`~torch.nn` يستخدم :mod:`~torch.autograd` بشكل مكثف، فإن إضافة :class:`Module` جديدة تتطلب تنفيذ :class:`~torch.autograd.Function` الذي يؤدي العملية ويمكنه حساب التدرج. من الآن فصاعدًا، دعنا نفترض أننا نريد تنفيذ وحدة ``Linear`` ولقد قمنا بتنفيذ الدالة كما هو موضح في القائمة أعلاه. هناك القليل جدًا من التعليمات البرمجية المطلوبة لإضافة هذا. الآن، هناك دالتان يجب تنفيذهما:

- ``__init__`` (*اختياري*) - يأخذ حججًا مثل أحجام النواة، وعدد الميزات، وما إلى ذلك، ويبدأ المعلمات والمخازن المؤقتة.
- :meth:`~Module.forward` - ينشئ :class:`~torch.autograd.Function` ويستخدمه لأداء العملية. إنه مشابه جدًا للغلاف الوظيفي الموضح أعلاه.

هذه هي الطريقة التي يمكن بها تنفيذ وحدة ``Linear``::

    class Linear(nn.Module):
        def __init__(self, input_features, output_features, bias=True):
            super().__init__()
            self.input_features = input_features
            self.output_features = output_features

            # nn.Parameter هو نوع خاص من Tensor، سيتم تسجيله تلقائيًا
            # كمعلمة الوحدة النمطية بمجرد تعيينه كسمة. يجب تسجيل المعلمات والمخازن المؤقتة،
            # أو لن تظهر في .parameters() (لا ينطبق على المخازن المؤقتة)، ولن يتم تحويلها عند
            # يتم استدعاء .cuda() على سبيل المثال. يمكنك استخدام .register_buffer() لتسجيل المخازن المؤقتة.
            # تتطلب معلمات nn.Parameters تدرجات بشكل افتراضي.
            self.weight = nn.Parameter(torch.empty(output_features, input_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(output_features))
            else:
                # يجب عليك دائمًا تسجيل جميع المعلمات المحتملة، ولكن
                # يمكن أن تكون تلك الاختيارية Null إذا أردت ذلك.
                self.register_parameter('bias', None)

            # ليست طريقة ذكية جدًا لتهيئة الأوزان
            nn.init.uniform_(self.weight, -0.1, 0.1)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -0.1, 0.1)

        def forward(self, input):
            # راجع قسم autograd لشرح ما يحدث هنا.
            return LinearFunction.apply(input, self.weight, self.bias)

        def extra_repr(self):
            # (اختياري) قم بتعيين المعلومات الإضافية حول هذه الوحدة النمطية. يمكنك اختباره
            # عن طريق طباعة كائن من هذه الفئة.
            return 'input_features={}, output_features={}, bias={}'.format(
                self.input_features, self.output_features, self.bias is not None
            )

.. _extending-torch-python:

توسيع واجهة برمجة التطبيقات :mod:`torch` Python
يمكنك إنشاء أنواع مخصصة تحاكي فئة "Tensor" من خلال تعريف فئة مخصصة مع طرق مطابقة لـ "Tensor". ولكن ماذا لو أردت أن تكون قادرًا على تمرير هذه الأنواع إلى وظائف مثل "torch.add" في مساحة الاسماء "torch" التي تقبل المعاملات من فئة "Tensor"؟

إذا كان نوع Python المخصص الخاص بك يحدد طريقة تسمى "__torch_function__"، فسيقوم PyTorch باستدعاء تنفيذ "__torch_function__" عندما يتم تمرير مثيل لفئتك المخصصة إلى دالة في مساحة الاسماء "torch". يسمح هذا بتعريف تنفيذ مخصص لأي من الوظائف في مساحة الاسماء "torch" والتي يمكن أن يستدعيها تنفيذ "__torch_function__" الخاص بك، مما يسمح لمستخدميك باستخدام نوعك المخصص مع سير عمل PyTorch الموجودة التي قاموا بالفعل بكتابتها لفئة "Tensor". يعمل هذا مع أنواع "duck" غير المرتبطة بفئة "Tensor" وكذلك الفئات الفرعية المحددة من قبل المستخدم من فئة "Tensor".

توسيع مساحة الاسماء "torch" بنوع مشابه لـ "Tensor"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ملاحظة:: هذه الوظيفة مستوحاة من بروتوكول "array_function" في NumPy. راجع وثائق NumPy وNEP-0018 لمزيد من التفاصيل.

لتوضيح ذلك، دعنا نبدأ بمثال بسيط يوضح آلية استدعاء واجهة برمجة التطبيقات. سنقوم بإنشاء نوع مخصص يمثل مصفوفة قياسية ثنائية الأبعاد، معلمة حسب الترتيب "N" والقيمة على طول الإدخالات القطرية، "value"::

     class ScalarTensor(object):
        def __init__(self, N, value):
            self._N = N
            self._value = value

        def __repr__(self):
            return "ScalarTensor(N={}, value={})".format(self._N, self._value)

        def tensor(self):
            return self._value * torch.eye(self._N)

هذا الإصدار الأول من التصميم ليس مفيدًا جدًا. الوظيفة الرئيسية لـ "ScalarTensor" هي توفير تمثيل سلسلة أكثر إحكاما لمصفوفة قياسية من فئة المصفوفة الأساسية::

  >>> d = ScalarTensor(5, 2)
  >>> d
  ScalarTensor(N=5, value=2)
  >>> d.tensor()
  tensor([[2., 0., 0., 0., 0.],
          [0., 2., 0., 0., 0.],
          [0., 0., 2., 0., 0.],
          [0., 0., 0., 2., 0.],
          [0., 0., 0., 0., 2.]])

إذا حاولنا استخدام هذا الكائن مع واجهة برمجة التطبيقات "torch"، فسوف نواجه بعض المشكلات::

  >>> import torch
  >>> torch.mean(d)
  TypeError: mean(): argument 'input' (position 1) must be Tensor, not ScalarTensor

إن إضافة تنفيذ "__torch_function__" إلى "ScalarTensor" يجعل من الممكن نجاح العملية المذكورة أعلاه. دعنا نعيد تنفيذنا، هذه المرة بإضافة تنفيذ "__torch_function__" ::

  HANDLED_FUNCTIONS = {}
  class ScalarTensor(object):
      def __init__(self, N, value):
          self._N = N
          self._value = value

      def __repr__(self):
          return "ScalarTensor(N={}, value={})".format(self._N, self._value)

      def tensor(self):
          return self._value * torch.eye(self._N)

      @classmethod
      def __torch_function__(cls, func, types, args=(), kwargs=None):
          if kwargs is None:
              kwargs = {}
          if func not in HANDLED_FUNCTIONS or not all(
              issubclass(t, (torch.Tensor, ScalarTensor))
              for t in types
          ):
              return NotImplemented
          return HANDLED_FUNCTIONS[func](*args, **kwargs)

تأخذ طريقة "__torch_function__" أربعة حجج: "func"، مرجع إلى دالة واجهة برمجة التطبيقات "torch" التي يتم تجاوزها، و "types"، قائمة بأنواع Tensor-likes التي تنفذ "__torch_function__"، و "args"، مجموعة من الحجج التي تم تمريرها إلى الدالة، و "kwargs"، قاموس من الحجج الكلمة الرئيسية التي تم تمريرها إلى الدالة. يستخدم جدول التوزيع العالمي المسمى "HANDLED_FUNCTIONS" لتخزين التنفيذات المخصصة. مفاتيح هذا القاموس هي وظائف في مساحة الاسماء "torch" والقيم هي التنفيذات لـ "ScalarTensor".

.. ملاحظة:: استخدام جدول التوزيع العالمي ليس جزءًا إلزاميًا من واجهة برمجة التطبيقات "__torch_function__"، ولكنه مجرد نمط تصميم مفيد لهياكل تنفيذ التجاوزات الخاصة بك.

هذا التعريف للفئة ليس كافيًا لجعل "torch.mean" يقوم بالشيء الصحيح عندما نمرر إليه "ScalarTensor" - نحتاج أيضًا إلى تعريف تنفيذ لـ "torch.mean" لـ "ScalarTensor" المعاملات وإضافة التنفيذ إلى قاموس جدول التوزيع "HANDLED_FUNCTIONS". إحدى طرق القيام بذلك هي تعريف الديكور::

  import functools
  def implements(torch_function):
      """Register a torch function override for ScalarTensor"""
      def decorator(func):
          functools.update_wrapper(func, torch_function)
          HANDLED_FUNCTIONS[torch_function] = func
          return func
      return decorator

والذي يمكن تطبيقه على تنفيذ التجاوز الخاص بنا::

  @implements(torch.mean)
  def mean(input):
      return float(input._value) / input._N

مع هذا التغيير، يمكننا الآن استخدام "torch.mean" مع "ScalarTensor"::

  >>> d = ScalarTensor(5, 2)
  >>> torch.mean(d)
  0.4

بالطبع، "torch.mean" هو مثال على أبسط نوع من الوظائف التي يتم تجاوزها حيث أنها تأخذ معامل واحد فقط. يمكننا استخدام نفس الآلية لتجاوز وظيفة تأخذ أكثر من معامل واحد، أي منها قد يكون مصفوفة أو مصفوفة تشبه التي تحدد "__torch_function__"، على سبيل المثال لـ:func: `torch.add`::

  def ensure_tensor(data):
      if isinstance(data, ScalarTensor):
          return data.tensor()
      return torch.as_tensor(data)

  @implements(torch.add)
  def add(input, other):
     try:
         if input._N == other._N:
             return ScalarTensor(input._N, input._value + other._value)
         else:
             raise ValueError("Shape mismatch!")
     except AttributeError:
         return torch.add(ensure_tensor(input), ensure_tensor(other))

يحتوي هذا الإصدار على مسار سريع عندما يكون كلا المعاملين مثيلين لـ "ScalarTensor" ومسار أبطأ يتدهور إلى تحويل البيانات إلى مصفوفات عندما لا يكون أي من المعاملات "ScalarTensor". وهذا يجعل دالة التجاوز تعمل بشكل صحيح عندما يكون أي من المعاملات "ScalarTensor" أو "Tensor" عادي::

  >>> s = ScalarTensor(2, 2)
  >>> torch.add(s, s)
  ScalarTensor(N=2, value=4)
  >>> t = torch.tensor([[1, 1,], [1, 1]])
  >>> torch.add(s, t)
  tensor([[3., 1.],
          [1., 3.]])

لاحظ أن تنفيذنا لـ "add" لا يأخذ "alpha" أو "out" كحجج الكلمة الرئيسية مثل:func: `torch.add` يفعل::

  >>> torch.add(s, s, alpha=2)
  TypeError: add() got an unexpected keyword argument 'alpha'

لأغراض السرعة والمرونة، لا تتحقق آلية استدعاء "__torch_function__" من أن توقيع دالة التجاوز يتطابق مع توقيع الدالة التي يتم تجاوزها في واجهة برمجة التطبيقات "torch". بالنسبة لبعض التطبيقات، قد يكون تجاهل الحجج الاختيارية أمرًا جيدًا، ولكن لضمان التوافق الكامل مع فئة "Tensor"، يجب أن تهتم التنفيذات الخاصة بواجهة برمجة التطبيقات "torch" بمحاكاة واجهة برمجة التطبيقات للدالة التي يتم تجاوزها بدقة.

ستعيد الوظائف في واجهة برمجة التطبيقات "torch" التي ليس لها تجاوزات صريحة "NotImplemented" من "__torch_function__". إذا أعادت جميع المعاملات التي تم تعريف "__torch_function__" عليها "NotImplemented"، فسيقوم PyTorch بإطلاق "TypeError". وهذا يعني أن معظم العمليات التي ليس لها تجاوزات صريحة لنوع ما سترفع "TypeError" عند تمرير مثيل لهذا النوع::

  >>> torch.mul(s, 3)
  TypeError: no implementation found for 'torch.mul' on types that
  implement __torch_function__: [ScalarTensor]

في الممارسة العملية، هذا يعني أنه إذا كنت ترغب في تنفيذ تجاوزاتك باستخدام تنفيذ "__torch_function__" على طول هذه الخطوط، فسيتعين عليك تنفيذ واجهة برمجة التطبيقات "torch" بالكامل أو المجموعة الفرعية الكاملة من واجهة برمجة التطبيقات التي تهتم بها لحالتك الاستخدام. قد يكون هذا أمرًا صعبًا نظرًا لاتساع واجهة برمجة التطبيقات "torch" بالكامل.

الخيار الآخر هو عدم إعادة "NotImplemented" للعمليات التي لا تتم معالجتها ولكن بدلاً من ذلك تمرير "Tensor" إلى الدالة "torch" الأصلية عندما لا يتوفر أي تجاوز. على سبيل المثال، إذا قمنا بتغيير تنفيذنا لـ "__torch_function__" لـ "ScalarTensor" إلى ما يلي::

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
      if kwargs is None:
          kwargs = {}
      if func not in HANDLED_FUNCTIONS or not all(
              issubclass(t, (torch.Tensor, ScalarTensor))
              for t in types
          ):
          args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
          return func(*args, **kwargs)
      return HANDLED_FUNCTIONS[func](*args, **kwargs)

سيعمل "torch.mul" بشكل صحيح، على الرغم من أن نوع الإرجاع سيكون دائمًا "Tensor" بدلاً من "ScalarTensor"، حتى إذا كان كلا المعاملين مثيلين لـ "ScalarTensor"::

  >>> s = ScalarTensor(2, 2)
  >>> torch.mul(s, s)
  tensor([[4., 0.],
          [0., 4.]])

راجع أيضًا مثال "MetadataTensor" أدناه لمزيد من التنوع على هذا النمط ولكنه بدلاً من ذلك يعيد دائمًا "MetadataTensor" لنشر البيانات الوصفية عبر العمليات في واجهة برمجة التطبيقات "torch".

تم تصميم بروتوكول "__torch_function__" لتغطية كاملة لواجهة برمجة التطبيقات، وقد يؤدي التغطية الجزئية إلى نتائج غير مرغوب فيها، خاصة أن بعض الوظائف ترفع "TypeError". ينطبق هذا بشكل خاص على الفئات الفرعية، حيث يجب تغطية كل من "torch.add" و "torch.Tensor.__add__" و "torch.Tensor.add"، حتى إذا أعادوا نفس النتيجة بالضبط. قد يؤدي الفشل في القيام بذلك أيضًا إلى حدوث استدعاء ذاتي لا نهائي. إذا تطلب أحد تنفيذ دالة من الفئات الفرعية لـ "torch.Tensor"، فيجب عليهم استخدام "super().__torch_function__" داخل تنفيذهم.

الاشتقاق من فئة "torch.Tensor"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
اعتبارًا من الإصدار 1.7.0، ستعيد الطرق على "torch.Tensor" والوظائف في مساحات الأسماء العامة "torch.*" المطبقة على الفئات الفرعية لـ "torch.Tensor" مثيلات الفئة الفرعية بدلاً من مثيلات "torch.Tensor"::

  >>> class SubTensor(torch.Tensor):
  ...     pass
  >>> type(torch.add(SubTensor([0]), SubTensor([1]))).__name__
  'SubTensor'
  >>> type(torch.add(SubTensor([0]), torch.tensor([1]))).__name__
  'SubTensor'

إذا كانت هناك فئات فرعية متعددة، فسيتم اختيار أدنى منها في التسلسل الهرمي بشكل افتراضي. إذا لم تكن هناك طريقة فريدة لتحديد هذه الحالة، فسيتم رفع "TypeError"::

  >>> type(torch.add(SubTensor2([0]), SubTensor([1]))).__name__
  'SubTensor2'
  >>> type(torch.add(SubTensor2([0]), torch.tensor([1]))).__name__
  'SubTensor2'
  >>> torch.add(SubTensor([0]), OtherSubTensor([1]))
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: no implementation found for 'torch.add' on types that implement __torch_function__: [SubTensor, OtherSubTensor]

إذا كنت ترغب في الحصول على تجاوز عالمي لجميع طرق المصفوفة، فيمكنك استخدام "__torch_function__". فيما يلي مثال يقوم بتسجيل جميع استدعاءات الدوال/الطرق::

  class LoggingTensor(torch.Tensor):
      @classmethod
      def __torch_function__(cls, func, types, args=(), kwargs=None):
          # NOTE: Logging calls Tensor.__repr__, so we can't log __repr__ without infinite recursion
          if func is not torch.Tensor.__repr__:
              logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
          if kwargs is None:
              kwargs = {}
          return super().__torch_function__(func, types, args, kwargs)

ومع ذلك، إذا كنت ترغب بدلاً من ذلك في تجاوز طريقة على الفئة الفرعية للمصفوفة، فيمكنك القيام بذلك إما عن طريق تجاوز الطريقة مباشرة (من خلال تعريفها للفئة الفرعية)، أو باستخدام "__torch_function__" ومطابقتها مع "func".

يجب توخي الحذر داخل "__torch_function__" للفئات الفرعية للاتصال دائمًا بـ "super().__torch_function__(func، ...)" بدلاً من "func" مباشرة، كما كان الحال قبل الإصدار 1.7.0. قد يؤدي الفشل في القيام بذلك إلى استدعاء "func" مرة أخرى إلى "__torch_function__" وبالتالي يسبب استدعاء ذاتي لا نهائي.
بالتأكيد! فيما يلي النص المترجم إلى اللغة العربية مع الحفاظ على تنسيق ReStructuredText:

توسيع :mod:`torch` باستخدام نوع wrapper لـ :class:`Tensor`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

هناك حالة مفيدة أخرى وهي نوع يقوم بتغليف :class:`Tensor`، إما كسمة أو عن طريق الوراثة. نقوم بتنفيذ حالة خاصة من هذا النوع من الأنواع أدناه، وهي ``MetadataTensor`` التي تقوم بتعليق قاموس من البيانات الوصفية على :class: `Tensor` التي يتم نشرها عبر عمليات :mod:`torch`. نظرًا لأن هذا نوع عام من التغليف لواجهة برمجة التطبيقات (API) الكاملة لـ :mod:`torch`، فلا يلزم تنفيذ كل استبدال بشكل فردي، لذا يمكننا جعل تنفيذ ``__torch_function__`` أكثر تساهلاً بشأن العمليات المسموح بها::

  class MetadataTensor(object):
      def __init__(self, data, metadata=None, **kwargs):
          self._t = torch.as_tensor(data, **kwargs)
          self._metadata = metadata

      def __repr__(self):
          return "Metadata:\n{}\n\ndata:\n{}".format(self._metadata, self._t)

      @classmethod
      def __torch_function__(cls, func, types, args=(), kwargs=None):
          if kwargs is None:
              kwargs = {}
          metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
          args = [getattr(a, '_t', a) for a in args]
          assert len(metadatas) > 0
          ret = func(*args, **kwargs)
          return MetadataTensor(ret, metadata=metadatas[0])

لن يعمل هذا التنفيذ البسيط بالضرورة مع كل دالة في واجهة برمجة تطبيقات :mod:`torch` ولكنه جيد بما يكفي لالتقاط معظم العمليات الشائعة::

  >>> metadata = {'owner': 'Ministry of Silly Walks'}
  >>> m = MetadataTensor([[1, 2], [3, 4]], metadata=metadata)
  >>> t = torch.tensor([[1, 2], [1, 2]])
  >>> torch.add(t, m)
  Metadata:
  {'owner': 'Ministry of Silly Walks'}

  data:
  tensor([[2, 4],
          [4, 6]])
  >>> torch.mul(t, m)
  Metadata:
  {'owner': 'Ministry of Silly Walks'}

  data:
  tensor([[1, 4],
          [3, 8]])

العمليات على الأنواع المتعددة التي تحدد ``__torch_function__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

من الممكن استخدام واجهة برمجة تطبيقات PyTorch مع أنواع متميزة متعددة تحتوي كل منها على تنفيذ لـ ``__torch_function__``، ولكن يجب توخي الحذر الخاص. في مثل هذه الحالة، تكون القواعد على النحو التالي:

* تجمع عملية التوزيع جميع التنفيذات المتميزة لـ ``__torch_function__`` لكل عامل تشغيل وتستدعيها بالترتيب: الفئات الفرعية قبل الفئات الأساسية، وإلا من اليسار إلى اليمين في تعبير المشغل.
* إذا تم إرجاع أي قيمة أخرى غير ``NotImplemented``، يتم إرجاع تلك القيمة كنتيجة. يمكن للتنفيذات تسجيل أنها لا تنفذ عملية عن طريق إرجاع ``NotImplemented``.
* إذا أعادت جميع تنفيذات ``__torch_function__`` إرجاع ``NotImplemented``، يرمي PyTorch ``TypeError``.

اختبار تغطية التجاوزات لواجهة برمجة تطبيقات PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

أحد الجوانب المزعجة لتنفيذ ``__torch_function__`` هو أنه إذا كانت بعض العمليات تحتوي على تجاوزات والبعض الآخر لا يحتوي عليها، فسيحصل المستخدمون على أفضل تجربة غير متسقة، أو على أسوأ تقدير، سيشاهدون أخطاءً يتم إلقاؤها في وقت التشغيل عند استخدامهم لدالة لا تحتوي على تجاوز. لتسهيل هذه العملية، يوفر PyTorch واجهة برمجة تطبيقات للمطورين لضمان الدعم الكامل لتجاوزات ``__torch_function__``. تعد واجهة برمجة التطبيقات هذه خاصة وقد تخضع لتغييرات دون سابق إنذار في المستقبل.

أولاً، للحصول على قائمة بجميع الوظائف التي يمكن تجاوزها، استخدم ``torch.overrides._get_overridable_functions``. ويعيد هذا قاموسًا تكون مفاتيحه مساحات أسماء في واجهة برمجة تطبيقات Python لـ ``PyTorch`` وقيمه قائمة بالوظائف في تلك المساحة التي يمكن تجاوزها. على سبيل المثال، دعنا نقوم بطباعة أسماء الدالات الخمس الأولى في ``torch.nn.functional`` التي يمكن تجاوزها::

  >>> from torch.overrides import get_overridable_functions
  >>> func_dict = get_overridable_functions()
  >>> nn_funcs = func_dict[torch.nn.functional]
  >>> print([f.__name__ for f in nn_funcs[:5])
  ['adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
   'adaptive_max_pool1d', 'adaptive_max_pool1d_with_indices']

يتيح هذا الإدراج للوظائف إمكانية التكرار عبر جميع الوظائف التي يمكن تجاوزها، ومع ذلك، في الممارسة العملية، لا يكفي هذا لكتابة اختبارات لجميع هذه الوظائف دون عناء ونسخ توقيع كل دالة يدويًا لكل اختبار. لتسهيل هذه العملية، تقوم دالة ``torch.overrides._get_testing_overrides`` بإرجاع قاموس يرسم وظائف قابلة للتجاوز في واجهة برمجة تطبيقات PyTorch إلى دالات وهمية لامدا لها نفس التوقيع مثل الدالة الأصلية ولكنها تعيد بشكل غير مشروط -1. تعد هذه الوظائف الأكثر فائدة لاستخدامها مع ``inspect`` لتحليل توقيع الدالة للدالة الأصلية لـ PyTorch::

  >>> import inspect
  >>> from torch.overrides import get_testing_overrides
  >>> override_dict = get_testing_overrides()
  >>> dummy_add = override_dict[torch.add]
  >>> inspect.signature(dummy_add)
  <Signature (input, other, out=None)>

أخيرًا، تقوم دالة ``torch.overrides.get_ignored_functions`` بإرجاع مجموعة من الوظائف التي لا يمكن تجاوزها بشكل صريح بواسطة ``__torch_function__``. يمكن استخدام هذه القائمة للتأكيد على أن الدالة التي لا تظهر في القاموس الذي تم إرجاعه بواسطة ``get_overridable_functions`` لا يمكن تجاوزها.


.. _extending-torch-c++:

توسيع واجهة برمجة التطبيقات الأصلية لـ :mod:`torch`
--------------------------------------

بينما تسمح ``__torch_function__`` بتوسيع سلوك المكونات النقية لـ PyTorch المكتوبة بلغة Python بشكل فعال، إلا أنها لا تسمح بتوسيع الأجزاء المكتوبة بلغة C++ في PyTorch. ولهذا الغرض، يمكن لصنف فرعي لـ ``Tensor`` أيضًا أن يحدد ``__torch_dispatch__`` الذي يمكنه تجاوز السلوك على مستوى C++.

لاستخدام هذه الميزة بشكل فعال، من المهم معرفة كيفية تنفيذ الجزء الأصلي من PyTorch. المكون الأكثر أهمية هناك هو ما نسميه "المُرسل" (يمكن العثور على أفضل وصف له في هذه `التدوينة <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ على الرغم من أنها قديمة بعض الشيء). وكما يوحي اسمها، فهي مسؤولة عن استدعاء دالة backend الصحيحة لمكالمة محددة لدالة. على سبيل المثال، عند استدعاء ``torch.add(a, b)``، يقوم المُرسل بفحص كلا وسيطي الدالة، ويحدد أي "ميزة" (autograd أو autocast أو functionalization، إلخ) وأي "backend" (CPU أو CUDA أو MPS، إلخ) يجب استخدامها لهذه المكالمة المحددة، وأخيرًا استدعاء جميع kernels الصحيحة.

من الشائع جدًا أن تقوم kernel "بإعادة التوزيع". على سبيل المثال، عند تشغيل الشبكة العصبية على GPU مع autocast، ستكون المكالمة الأولى هي kernel autocast التي ستتعامل مع أي منطق autocast وستعيد التوزيع. ستكون الميزة التالية في الخط هي autograd التي ستنشئ مخطط autograd بشكل صحيح ثم تعيد التوزيع. أخيرًا، نصل إلى kernel backend لـ CUDA التي ستطلق kernel CUDA الصحيح وتعيد النتيجة النهائية. وفي طريق الخروج، سيرفق autograd المخطط بالناتج، وأخيرًا، ستحصل autocast على فرصة للقيام بأي تحديث تحتاجه عند الخروج.

تتمثل إحدى طرق تكوين المُرسل في ترتيب جميع مفاتيح الميزات ومفاتيح backends هذه. يمكن العثور على أحدث قائمة وترتيبها في ``DispatchKey.h`` داخل enum ``DispatchKey``. ولغرض توسيع نطاق torch، فإن المجموعة الفرعية المهمة من الترتيب لهذه المناقشة هي:

vmap -> Autocast -> Autograd -> ZeroTensor -> Neg/Conj -> Functionalize -> Python -> Backends

المفتاح الأكثر أهمية لغرض هذه المناقشة هو "Python" لأن كل صنف فرعي لـ Tensor مع طريقة ``__torch_dispatch__`` المحددة سيستدعي هذه الميزة. ومن هناك يتم استدعاء الطريقة المحددة من قبل المستخدم ويمكن إعادة كتابة السلوك بشكل تعسفي. ومن هناك، فإن استدعاء ``func`` المقدم مرة أخرى سيؤدي إلى "إعادة التوزيع".

بعض الآثار المترتبة على هذا التنفيذ هي:

- يعمل هذا الكود "أسفل جميع الميزات". وبالتالي، فهو مسؤول فقط، مثل backend العادي، عن توليد قيمة الإخراج لكل Tensor (ويمكنه، وينبغي عليه، تجاهل جميع الميزات المتقدمة مثل autograd و autocast، إلخ).
- إذا نفذت أي ميزة عالية المستوى دالة معينة دون إعادة التوزيع، فلن تصل إلى مفتاح "Python" ولن يتم تشغيل استدعاء الإرجاع ``__torch_dispatch__`` أبدًا. يحدث هذا بشكل خاص لوظائف CompositeImplicitAutograd التي يتم تقييمها على مستوى Autograd دون إعادة التوزيع. ويرجع ذلك إلى أن دالة CompositeImplicitAutograd تحدد صيغة autograd الخاصة بها عن طريق استدعاء عمليات أصلية أخرى ضمنيًا، لذا يتم تفكيك الدالة، على مستوى Autograd، إلى عملياتها الأصلية ويتم تقييم تلك العمليات بدلاً من ذلك.
- عند الاستدعاء مرة أخرى إلى Python وعند لف النتائج، يتم استخدام نفس التحويلات مثل ارتباط PyTorch Python/C++ العادي. على وجه الخصوص، لا يمكن تمثيل بعض الكائنات في Python وتحتاج إلى معالجة خاصة (على سبيل المثال، تصبح Tensors غير المحددة None).
- يتم ملء وظائفنا الأصلية بشكل متأخر ككائنات Python قابلة للاستدعاء ``torch.ops.{namespace}.{func_name}.{overload_name}`` للسماح بالتفاعل معها بسهولة من Python. يكون كائن ``func`` المقدم إلى ``__torch_dispatch__`` دائمًا إدخالًا من هذا namespace. يمكن استخدام هذا namespace لاستدعاء العمليات الأصلية مباشرة والالتفاف حول واجهة برمجة التطبيقات Python ورمز الارتباط المعتاد.

وبطريقة مماثلة حيث يمكن لـ ``__torch_function__`` التدخل في جميع واجهات برمجة التطبيقات لـ PyTorch في Python وجميع طرق Tensor، يمكن لـ ``__torch_dispatch__`` اعتراض جميع الاستدعاءات إلى واجهة برمجة التطبيقات الأصلية aten. لاحظ أن جميع الطرق على Tensors يتم تحويلها إلى استدعاءات دالة قبل دخول المُرسل وبالتالي ستظهر كمكالمات دالة هنا: ``torch.add(a, 2)`` و ``a + 2`` سيؤديان إلى نفس استدعاء aten تمامًا.

يتم تحديد معظم هذه الوظائف في ``native_functions.yaml`` الذي يحدد خصائص هذه الوظائف بالإضافة إلى تنفيذها في backend. يتم بعد ذلك تسجيل تنفيذها جنبًا إلى جنب مع الميزات المحددة تلقائيًا عبر codegen.

يتم أيضًا تسجيل بعض الوظائف أو الميزات الأكثر غرابة في أماكن أخرى في قاعدة كود C++ أو في ملحقات C++ المحددة من قبل المستخدم.

من الممكن أيضًا إضافة وظائف أصلية `جديدة` باستخدام :mod:`torch.library`. تتيح ميزة Python هذه تحديد و/أو إضافة تنفيذي جديد إلى الوظائف الأصلية. يمكن استخدامه لإضافة kernels المفقودة أو استبدال الموجودة أو تحديد وظائف أصلية جديدة تمامًا.

يمكنك العثور على العديد من الأمثلة على فئات فرعية قائمة على ``__torch_dispatch__`` في مستودع `subclass zoo <https://github.com/albanD/subclass_zoo>`_.

.. _torch-dispatch-calling-convention:

اتفاقية استدعاء ``__torch_dispatch__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        pass

عندما يستدعي المستخدم عامل تشغيل مع وسائط تحتوي على ``__torch_dispatch__``، فقد يتم إعادة توجيه تلك المكالمة إلى ``__torch_dispatch__``. يتم تطبيع الوسائط والحجج قبل استدعاء ``__torch_dispatch__``، أي:

- تتكون ``kwargs`` من وسائط الكلمة الأساسية فقط في مخطط المشغل.
  إذا كانت وسيطة الكلمة الأساسية تساوي قيمتها الافتراضية (في المخطط)، فلن يتم تمريرها.
- تتكون ``args`` من جميع الوسائط الأخرى، بغض النظر عن كيفية تمريرها
  إلى المشغل (موضعي مقابل الكلمة الأساسية).
  إذا كانت وسيطة تساوي قيمتها الافتراضية، وكانت
  وسيطة الموضع الأيمن أو جميع الوسائط إلى يمينها
  لا يتم تمريرها، فلن يتم تمريرها.

توسيع جميع واجهات برمجة التطبيقات :mod:`torch` باستخدام الأوضاع
------------------------------------------------

لسوء الحظ، هناك وظائف لا تأخذ وسائط Tensor. وهذا يعني أنه لا يمكن استخدام نهج الصنف الفرعي الموصوف أعلاه لتجاوز سلوك جميع وظائف PyTorch. أيضًا، إذا كانت حالة الاستخدام تتطلب اعتراض كل مكالمة دالة، فقد يكون تغيير كل Tensor إلى صنف فرعي أمرًا متطفلًا للغاية.

لمعالجة حالة الاستخدام هذه، قدمنا مفهوم "الوضع". توجد هذه الأوضاع لـ ``__torch_function__`` و ``__torch_dispatch__`` تجاوزات، يتم إنشاؤها عن طريق إنشاء صنف فرعي لـ :class:`torch.overrides.TorchFunctionMode` و :class:`torch.utils._python_dispatch.TorchDispatchMode`، على التوالي، ويتم استخدامها كمدير سياق.

لتبسيط وصف كيفية تفاعلها مع الأصناف الفرعية والأوضاع الأخرى، عندما يتم إدخال مدير سياق لوضع ما، يتصرف كل دالة كما لو كان هناك وسيط Tensor إضافي في بداية قائمة وسائط الدالة مع الوضع كصنف فرعي.

هذا يعني على وجه الخصوص أن جميع معالجات الأوضاع ستتم استدعاؤها قبل أي معالج للصنف الفرعي وأن الأوضاع المقابلة لمدير السياق الداخلي ستعمل دائمًا أولاً.

من المهم أيضًا ملاحظة أنه داخل معالج وضع معين، يتم تعطيل هذا الوضع المحدد ويمكن إعادة تمكينه يدويًا عن طريق القيام بـ ``with self:``.

فيما يلي مثال يوضح أوضاع تسجيل كل نوع::

  import torch
  from torch.overrides import TorchFunctionMode, resolve_name
  from torch.utils._python_dispatch import TorchDispatchMode

  class FunctionLog(TorchFunctionMode):
      def __torch_function__(self, func, types, args, kwargs=None):
          print(f"Function Log: {resolve_name(func)}(*{args}, **{kwargs})")
          return func(*args, **(kwargs or {}))

  class DispatchLog(TorchDispatchMode):
      def __torch_dispatch__(self, func, types, args, kwargs=None):
          print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
          return func(*args, **(kwargs or {}))

  def f():
      a = torch.rand(10, requires_grad=True)
      b = a * 2
      b.sum().backward()

  print("TorchFunctionMode logging:")
  with FunctionLog():
      f()

  print("TorchDispatchMode logging:")
  with DispatchLog():
      f()

والذي يطبع ما يلي، مع تعليقات إضافية::

  TorchFunctionMode logging:
  Function Log: torch.rand(*(10,), **{'requires_grad': True})
  Function Log: torch.Tensor.mul(*(tensor([0.7164, 0.9897, 0.1745, 0.9336, 0.4287, 0.7989, 0.2169, 0.7474, 0.5624,
          0.5970], requires_grad=True), 2), **None)
  Function Log: torch.Tensor.sum(*(tensor([1.4328, 1.9794, 0.3490, 1.8671, 0.8573, 1.5977, 0.4338, 1.4948, 1.1249,
          1.1939], grad_fn=<MulBackward0>),), **None)
  # Note that at the python level, we only see the call to backward but not what happens in the autograd engine.
  Function Log: torch.Tensor.backward(*(tensor(12.3307, grad_fn=<SumBackward0>),), **{'gradient': None, 'retain_graph': None, 'create_graph': False, 'inputs': None})

  TorchDispatchMode logging:
  # Here the requires_grad flag from autograd is removed while default arguments were populated.
  Dispatch Log: aten.rand.default(*([10],), **{'device': device(type='cpu'), 'pin_memory': False})
  Dispatch Log: aten.mul.Tensor(*(tensor([0.2151, 0.6018, 0.8415, 0.9060, 0.2974, 0.7708, 0.6668, 0.0352, 0.7948,
          0.6023], requires_grad=True), 2), **{})
  Dispatch Log: aten.sum.default(*(tensor([0.4303, 1.2036, 1.6831, 1.8120, 0.5949, 1.5416, 1.3335, 0.0705, 1.5897,
          1.2046], grad_fn=<MulBackward0>),), **{})
  # Here we don't see the call to backward itself, but its constituents. Starting here with the factory function that creates the initial gradient.
  Dispatch Log: aten.ones_like.default(*(tensor(11.4637, grad_fn=<SumBackward0>),), **{'pin_memory': False, 'memory_format': torch.preserve_format})
  # This is the backward of the sum
  Dispatch Log: aten.expand.default(*(tensor(1.), [10]), **{})
  Dispatch Log: aten.mul.Tensor(*(tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 2), **{})
  Dispatch Log: aten.detach.default(*(tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]),), **{})
  Dispatch Log: aten.detach.default(*(tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]),), **{})