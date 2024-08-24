.. automodule:: torch.masked
.. automodule:: torch.masked.maskedtensor

.. currentmodule:: torch

.. _masked-docs:

torch.masked
============

مقدمة
+++++

الدافع
----

.. warning::

  واجهة برمجة التطبيقات PyTorch الخاصة بالتوابع ذات القناع هي في مرحلة النموذج الأولي وقد تتغير أو لا تتغير في المستقبل.

يُستخدم MaskedTensor كتوسيع لـ :class:`torch.Tensor` يوفر للمستخدم القدرة على:

* استخدام أي دلالة مقنعة (على سبيل المثال، التوابع ذات الأطوال المتغيرة، مشغلات Nan*، إلخ)
* التمييز بين 0 و NaN gradients
* تطبيقات متنوعة نادرة (راجع البرنامج التعليمي أدناه)

لقد كان لكل من "محدد" و "غير محدد" تاريخ طويل في PyTorch دون دلالة رسمية وبالتأكيد دون اتساق؛ في الواقع، ولد MaskedTensor من تراكم المشكلات التي لم تتمكن فئة :class:`torch.Tensor`
العادية من معالجتها بشكل صحيح. وبالتالي، فإن أحد الأهداف الرئيسية لـ MaskedTensor هو أن يصبح مصدر الحقيقة لـ
القيم "المحددة" و "غير المحددة" في PyTorch حيث تكون من الدرجة الأولى بدلاً من فكرة لاحقة.
بدوره، يجب أن يفتح هذا أيضًا إمكانات `التناثر <https://pytorch.org/docs/stable/sparse.html>`_،
ويمكن من المشغلات الأكثر أمانًا واتساقًا، ويوفر تجربة أكثر سلاسة وبديهية
للمستخدمين والمطورين على حد سواء.

ما هو MaskedTensor؟
--------------------

MaskedTensor هو فئة فرعية من التوابع تتكون من 1) إدخال (بيانات)، و 2) قناع. يخبرنا القناع
بالمدخلات التي يجب تضمينها أو تجاهلها.

على سبيل المثال، افترض أننا أردنا إخفاء جميع القيم التي تساوي 0 (تمثلها اللون الرمادي)
وخذ الحد الأقصى:

.. image:: _static/img/masked/tensor_comparison.jpg
      :scale: 50%

في الأعلى مثال التابع العادي بينما في الأسفل MaskedTensor حيث يتم إخفاء جميع الأصفار.
من الواضح أن هذا يعطي نتيجة مختلفة اعتمادًا على ما إذا كان لدينا القناع، ولكن يسمح هذا الهيكل المرن للمستخدم
باستبعاد أي عناصر بشكل منهجي أثناء الحساب.

هناك بالفعل عدد من البرامج التعليمية الموجودة التي كتبناها لمساعدة المستخدمين على الصعود، مثل:

-  `نظرة عامة - المكان الذي يبدأ منه المستخدمون الجدد، ويناقش كيفية استخدام MaskedTensors ولماذا هي مفيدة`_
-  `التناثر - يدعم MaskedTensor بيانات قناع COO و CSR النادرة`_
-  `Adagrad نادرة الدلالة - مثال عملي على كيفية تبسيط MaskedTensor لدلالة نادرة وتنفيذها`_
-  `الدلالات المتقدمة - مناقشة حول سبب اتخاذ قرارات معينة (على سبيل المثال، تتطلب الأقنعة المطابقة لعمليات ثنائية/اختزال)،
   الاختلافات مع مصفوفة NumPy المقنعة، ودلالة الاختزال`_

.. _نظرة عامة - المكان الذي يبدأ منه المستخدمون الجدد، ويناقش كيفية استخدام MaskedTensors ولماذا هي مفيدة: https://pytorch.org/tutorials/prototype/maskedtensor_overview
.. _التناثر - يدعم MaskedTensor بيانات قناع COO و CSR النادرة: https://pytorch.org/tutorials/prototype/maskedtensor_sparsity
.. _Adagrad نادرة الدلالة - مثال عملي على كيفية تبسيط MaskedTensor لدلالة نادرة وتنفيذها: https://pytorch.org/tutorials/prototype/maskedtensor_adagrad
.. _الدلالات المتقدمة - مناقشة حول سبب اتخاذ قرارات معينة (على سبيل المثال، تتطلب الأقنعة المطابقة لعمليات ثنائية/اختزال)، الاختلافات مع مصفوفة NumPy المقنعة، ودلالة الاختزال: https://pytorch.org/tutorials/prototype/maskedtensor_advanced_semantics

المشغلون المدعومون
+++++++++++++

المشغلون الأحاديين
------------

المشغلون الأحاديين هم مشغلون يحتوي كل منهم على إدخال واحد فقط.
تطبيقها على MaskedTensors مباشر إلى حد ما: إذا تم إخفاء البيانات في فهرس معين،
نطبق المشغل، وإلا فسوف نواصل إخفاء البيانات.

المشغلون الأحاديون المتاحون هم:

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    absolute
    acos
    arccos
    acosh
    arccosh
    angle
    asin
    arcsin
    asinh
    arcsinh
    atan
    arctan
    atanh
    arctanh
    bitwise_not
    ceil
    clamp
    clip
    conj_physical
    cos
    cosh
    deg2rad
    digamma
    erf
    erfc
    erfinv
    exp
    exp2
    expm1
    fix
    floor
    frac
    lgamma
    log
    log10
    log1p
    log2
    logit
    i0
    isnan
    nan_to_num
    neg
    negative
    positive
    pow
    rad2deg
    reciprocal
    round
    rsqrt
    sigmoid
    sign
    sgn
    signbit
    sin
    sinc
    sinh
    sqrt
    square
    tan
    tanh
    trunc

المشغلون الأحاديون المتاحون في المكان هم جميع ما سبق **باستثناء**:

.. autosummary::
    :toctree: generated
    :nosignatures:

    angle
    positive
    signbit
    isnan

المشغلون الثنائيون
----------------

كما رأيت في البرنامج التعليمي، :class:`MaskedTensor` لديه أيضًا عمليات ثنائية تم تنفيذها مع التحذير
أن الأقنعة في MaskedTensors يجب أن تتطابق وإلا فسيتم إلقاء خطأ. كما هو مذكور في الخطأ، إذا كنت
تحتاج إلى دعم مشغل معين أو لديك دلالات مقترحة لكيفية تصرفها بدلاً من ذلك، يرجى فتح
قضية على GitHub. بالنسبة الآن، قررنا الذهاب إلى التنفيذ الأكثر تحفظًا لضمان معرفة المستخدمين بالضبط
ما يحدث وأنهم متعمدون بشأن قراراتهم بالدلالات المقنعة.

المشغلون الثنائيون المتاحون هم:

.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    atan2
    arctan2
    bitwise_and
    bitwise_or
    bitwise
_xor
    bitwise_left_shift
    bitwise_right_shift
    div
    divide
    floor_divide
    fmod
    logaddexp
    logaddexp2
    mul
    multiply
    nextafter
    remainder
    sub
    subtract
    true_divide
    eq
    ne
    le
    ge
    greater
    greater_equal
    gt
    less_equal
    lt
    less
    maximum
    minimum
    fmax
    fmin
    not_equal

المشغلون الثنائيون المتاحون في المكان هم جميع ما سبق **باستثناء**:

.. autosummary::
    :toctree: generated
    :nosignatures:

    logaddexp
    logaddexp2
    equal
    fmin
    minimum
    fmax

اختزالات
--------

فيما يلي الاختزالات المتاحة (مع دعم autograd). لمزيد من المعلومات، فإن البرنامج التعليمي
`نظرة عامة <https://pytorch.org/tutorials/prototype/maskedtensor_overview.html>`_ تفاصيل بعض الأمثلة على الاختزالات، في حين أن
البرنامج التعليمي `الدلالات المتقدمة <https://pytorch.org/tutorials/prototype/maskedtensor_advanced_semantics.html>`_ لديه بعض المناقشات المعمقة حول كيفية اتخاذ قرار بشأن دلالات اختزال معينة.

.. autosummary::
    :toctree: generated
    :nosignatures:

    sum
    mean
    amin
    amax
    argmin
    argmax
    prod
    all
    norm
    var
    std

عرض واختيار الدوال
--------------

لقد تضمنا عددًا من وظائف العرض والاختيار أيضًا؛ بديهيًا، ستطبق هذه المشغلين
على كل من البيانات والقناع ثم تغلف النتيجة في :class:`MaskedTensor`. على سبيل المثال سريع،
ضع في اعتبارك :func:`select`:

    >>> data = torch.arange(12, dtype=torch.float).reshape(3, 4)
    >>> data
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]])
    >>> mask = torch.tensor([[True, False, False, True], [False, True, False, False], [True, True, True, True]])
    >>> mt = masked_tensor(data, mask)
    >>> data.select(0, 1)
    tensor([4., 5., 6., 7.])
    >>> mask.select(0, 1)
    tensor([False,  True, False, False])
    >>> mt.select(0, 1)
    MaskedTensor(
      [      --,   5.0000,       --,       --]
    )

عمليات العرض والاختيار المدعومة حاليًا هي:

.. autosummary::
    :toctree: generated
    :nosignatures:

    atleast_1d
    broadcast_tensors
    broadcast_to
    cat
    chunk
    column_stack
    dsplit
    flatten
    hsplit
    hstack
    kron
    meshgrid
    narrow
    ravel
    select
    split
    t
    transpose
    vsplit
    vstack
    Tensor.expand
    Tensor.expand_as
    Tensor.reshape
    Tensor.reshape_as
    Tensor.view

.. يجب توثيق هذه الوحدة. أضفها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.masked.maskedtensor.binary
.. py:module:: torch.masked.maskedtensor.core
.. py:module:: torch.masked.maskedtensor.creation
.. py:module:: torch.masked.maskedtensor.passthrough
.. py:module:: torch.masked.maskedtensor.reductions
.. py:module:: torch.masked.maskedtensor.unary