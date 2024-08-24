.. _torch-label:

**الإطار PyTorch**

`PyTorch <https://pytorch.org/>`_ هو إطار عمل مفتوح المصدر للتعلم الآلي، تم تطويره بشكل أساسي من قبل `Facebook's AI Research lab <https://ai.facebook.com/>`_ (FAIR). يوفر PyTorch واجهة تفاعل سهلة الاستخدام للتعلم الآلي العميق، مما يسمح للمطورين بتصميم وتدريب شبكات عصبية معقدة بفعالية.

أحد المزايا الرئيسية لـ PyTorch هو قدرته على استخدام التسارع عبر وحدات معالجة الرسوميات (GPU) أو وحدات معالجة الرسوميات التخصصية (TPU)، مما يجعله أداة قوية للتعامل مع نماذج التعلم العميق الكبيرة والمعقدة. كما يوفر PyTorch قدرات تفاضلية قوية، مما يجعله أداة مفضلة للعديد من الباحثين في مجال التعلم الآلي.

بالإضافة إلى ذلك، يتمتع PyTorch بدعم مجتمعي قوي، مع توفر العديد من المكتبات والموارد الإضافية التي تمكن المطورين من تسريع عملية تطويرهم. كما أن لديها نظامًا بيئيًا غنيًا من الأدوات والوحدات التي يمكن أن تساعد في مختلف جوانب تطوير التعلم الآلي، بما في ذلك ما قبل المعالجة، والتصور، ونشر النماذج.

يوفر PyTorch أيضًا PyTorch Hub، وهو مستودع عبر الإنترنت للنماذج المسبقة التدريب والنماذج التي تم إنشاؤها بواسطة المجتمع، مما يسهل على المطورين والباحثين الوصول إليها واستخدامها.

.. _torch-install-label:

**تثبيت PyTorch**

يمكن تثبيت PyTorch باستخدام مدير الحزم ``pip``::

   pip install torch torchvision

يرجى ملاحظة أن PyTorch يتطلب بعض المكتبات الإضافية، مثل ``numpy`` و``matplotlib``، والتي يمكن تثبيتها أيضًا باستخدام ``pip``.

.. _torch-resources-label:

**الموارد**

- `موقع PyTorch الرسمي <https://pytorch.org/>`_

- `وثائق PyTorch <https://pytorch.org/docs/stable/index.html>`_

- `PyTorch على GitHub <https://github.com/pytorch/pytorch>`_

- `دليل البدء السريع لـ PyTorch <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`_

- `مجموعة أدوات PyTorch <https://pytorch.org/tutorials>`_ - مجموعة من البرامج التعليمية والكتب الإلكترونية حول PyTorch.

- `منتدى مناقشة PyTorch <https://discuss.pytorch.org/>`_ - مكان رائع لطرح الأسئلة ومناقشة PyTorch مع المجتمع.

.. seealso::

   `إطار TensorFlow <tensorflow.html>`_
      تعرف على إطار TensorFlow للتعلم العميق.

.. |PyTorch| image:: ../_static/logos/PyTorch.svg
   :width: 300px
   :alt: PyTorch

.. _PyTorch: https://pytorch.org/

.. |Facebook| image:: ../_static/logos/facebook.svg
   :width: 200px
   :alt: Facebook

.. _Facebook: https://www.facebook.com/

.. |FAIR| replace:: مختبر FAIR

.. _FAIR: https://ai.facebook.com/

.. |GPU| replace:: وحدة معالجة الرسوميات (GPU)

.. |TPU| replace:: وحدة معالجة الرسوميات التخصصية (TPU)

.. |numpy| replace:: NumPy

.. _numpy: https://numpy.org/

.. |matplotlib| replace:: Matplotlib

.. _matplotlib: https://matplotlib.org/

.. |pip| replace:: Pip

.. _pip: https://pip.pypa.io/

.. |PyTorch Hub| raw:: html

   <a href="https://pytorch.org/hub/" target="_blank">PyTorch Hub</a>

.. |PyTorch Hub| replace:: PyTorch Hub

.. _PyTorch Hub: https://pytorch.org/hub/

.. |PyTorch tutorials| raw:: html

   <a href="https://pytorch.org/tutorials/" target="_blank">مجموعة أدوات PyTorch</a>

.. |PyTorch tutorials| replace:: مجموعة أدوات PyTorch

.. _PyTorch tutorials: https://pytorch.org/tutorials/

.. |PyTorch forum| raw:: html

   <a href="https://discuss.pytorch.org/" target="_blank">منتدى مناقشة PyTorch</a>

.. |PyTorch forum| replace:: منتدى مناقشة PyTorch

.. _PyTorch forum: https://discuss.pytorch.Multiplier: https://en.wikipedia.org/wiki/Multiprocessingorg/
.. automodule:: torch
.. currentmodule:: torch

التنسورات
-------
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_tensor
    is_storage
    is_complex
    is_conj
    is_floating_point
    is_nonzero
    set_default_dtype
    get_default_dtype
    set_default_device
    get_default_device
    set_default_tensor_type
    numel
    set_printoptions
    set_flush_denormal

.. _tensor-creation-ops:

عمليات الإنشاء
~~~~~~~~~~~~

.. note::
    يتم سرد عمليات إنشاء العينات العشوائية تحت :ref:`random-sampling` وتشمل:
    :func:`torch.rand`
    :func:`torch.rand_like`
    :func:`torch.randn`
    :func:`torch.randn_like`
    :func:`torch.randint`
    :func:`torch.randint_like`
    :func:`torch.randperm`
    يمكنك أيضًا استخدام :func:`torch.empty` مع :ref:`inplace-random-sampling`
    الأساليب لإنشاء :class:`torch.Tensor` s مع القيم التي تم أخذ عينات منها من مجموعة أوسع
    من التوزيعات.

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    sparse_coo_tensor
    sparse_csr_tensor
    sparse_csc_tensor
    sparse_bsr_tensor
    sparse_bsc_tensor
    asarray
    as_tensor
    as_strided
    from_file
    from_numpy
    from_dlpack
    frombuffer
    zeros
    zeros_like
    ones
    ones_like
    arange
    range
    linspace
    logspace
    eye
    empty
    empty_like
    empty_strided
    full
    full_like
    quantize_per_tensor
    quantize_per_channel
    dequantize
    complex
    polar
    heaviside

.. _indexing-slicing-joining:

الفهرسة والتقطيع والانضمام وعمليات الطفرة
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    adjoint
    argwhere
    cat
    concat
    concatenate
    conj
    chunk
    dsplit
    column_stack
    dstack
    gather
    hsplit
    hstack
    index_add
    index_copy
    index_reduce
    index_select
    masked_select
    movedim
    moveaxis
    narrow
    narrow_copy
    nonzero
    permute
    reshape
    row_stack
    select
    scatter
    diagonal_scatter
    select_scatter
    slice_scatter
    scatter_add
    scatter_reduce
    split
    squeeze
    stack
    swapaxes
    swapdims
    t
    take
    take_along_dim
    tensor_split
    tile
    transpose
    unbind
    unravel_index
    unsqueeze
    vsplit
    vstack
    where

.. _accelerators:

المعالجات
------------
في مستودع PyTorch، نُعرِّف "المعالج" على أنه :class:`torch.device` الذي يتم استخدامه
إلى جانب وحدة المعالجة المركزية لتسريع الحساب. تستخدم هذه الأجهزة مخطط تنفيذ غير متزامن،
باستخدام :class:`torch.Stream` و :class:`torch.Event` كطريقتها الرئيسية لأداء المزامنة.
نفترض أيضًا أنه لا يمكن إلا لمعالج تسريع واحد أن يكون متاحًا في نفس الوقت على مضيف معين. يسمح لنا هذا
باستخدام المعالج المسرع الافتراضي كجهاز افتراضي لمفاهيم ذات صلة مثل الذاكرة المثبتة،
نوع الجهاز Stream، FSDP، إلخ.

اعتبارًا من اليوم، أجهزة المعالجة المسرعة هي (بدون ترتيب معين) :doc:`"CUDA" <cuda>`، :doc:`"MTIA" <mtia>`،
:doc:`"XPU" <xpu>`، وPrivateUse1 (العديد من الأجهزة غير الموجودة في مستودع PyTorch نفسه).

.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    Event

.. _generators:

المولدات
-----------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Generator

.. _random-sampling:

النماذج العشوائية
------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    seed
    manual_seed
    initial_seed
    get_rng_state
    set_rng_state

.. autoattribute:: torch.default_generator
   :annotation:  يعيد مولد CPU الافتراضي

.. The following doesn't actually seem to exist.
   https://github.com/pytorch/pytorch/issues/27780
   .. autoattribute:: torch.cuda.default_generators
      :annotation:  إذا كان CUDA متاحًا، فإنه يعيد مجموعة من مولدات CUDA الافتراضية.
                    عدد مولدات CUDA التي تم إرجاعها يساوي عدد
                    وحدات معالجة الرسومات المتوفرة في النظام.
.. autosummary::
    :toctree: generated
    :nosignatures:

    bernoulli
    multinomial
    normal
    poisson
    rand
    rand_like
    randint
    randint_like
    randn
    randn_like
    randperm

.. _inplace-random-sampling:

النماذج العشوائية في المكان
~~~~~~~~~~~~~~~~~~~~

هناك عدد قليل من وظائف النماذج العشوائية في المكان المحدد على التنسورات أيضًا. انقر من خلال الرجوع إلى وثائقها:

- :func:`torch.Tensor.bernoulli_` - الإصدار في المكان لـ :func:`torch.bernoulli`
- :func:`torch.Tensor.cauchy_` - الأرقام التي تم رسمها من توزيع كوشي
- :func:`torch.Tensor.exponential_` - الأرقام التي تم رسمها من التوزيع الأسي
- :func:`torch.Tensor.geometric_` - العناصر التي تم رسمها من التوزيع الهندسي
- :func:`torch.Tensor.log_normal_` - العينات من التوزيع اللوغاريتمي الطبيعي
- :func:`torch.Tensor.normal_` - الإصدار في المكان لـ :func:`torch.normal`
- :func:`torch.Tensor.random_` - الأرقام التي تم أخذ عينات منها من التوزيع المنتظم المتقطع
- :func:`torch.Tensor.uniform_` - الأرقام التي تم أخذ عينات منها من التوزيع المنتظم المستمر

النماذج شبه العشوائية
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: sobolengine.rst

    quasirandom.SobolEngine

التوصيف
---------
.. autosummary::
    :toctree: generated
    :nosignatures:

    save
    load

التوازي
----------
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_num_threads
    set_num_threads
    get_num_interop_threads
    set_num_interop_threads

.. _torch-rst-local-disable-grad:

تعطيل حساب التدرج محليًا
------------------
تعد برامج إدارة السياق :func:`torch.no_grad`، و :func:`torch.enable_grad`، و
:func:`torch.set_grad_enabled` مفيدة لتعطيل حساب التدرج وتمكينه محليًا. راجع :ref:`locally-disable-grad` لمزيد من التفاصيل حول
استخدامها. تعد برامج إدارة السياق هذه خاصة بالخيوط، لذا فلن تعمل إذا أرسلت عملًا إلى خيط آخر باستخدام وحدة "الخيوط"، إلخ.

أمثلة::

  >>> x = torch.zeros(1, requires_grad=True)
  >>> with torch.no_grad():
  ...     y = x * 2
  >>> y.requires_grad
  False

  >>> is_train = False
  >>> with torch.set_grad_enabled(is_train):
  ...     y = x * 2
  >>> y.requires_grad
  False

  >>> torch.set_grad_enabled(True)  # يمكن أيضًا استخدام هذا كدالة
  >>> y = x * 2
  >>> y.requires_grad
  True

  >>> torch.set_grad_enabled(False)
  >>> y = x * 2
  >>> y.requires_grad
  False

.. autosummary::
    :toctree: generated
    :nosignatures:

    no_grad
    enable_grad
    autograd.grad_mode.set_grad_enabled
    is_grad_enabled
    autograd.grad_mode.inference_mode
    is_inference_mode_enabled

عمليات الرياضيات
------------

عمليات نقطة إلى نقطة
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    absolute
    acos
    arccos
    acosh
    arccosh
    add
    addcdiv
    addcmul
    angle
    asin
    arcsin
    asinh
    arcsinh
    atan
    arctan
    atanh
    arctanh
    atan2
    arctan2
    bitwise_not
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_left_shift
    bitwise_right_shift
    ceil
    clamp
    clip
    conj_physical
    copysign
    cos
    cosh
    deg2rad
    div
    divide
    digamma
    erf
    erfc
    erfinv
    exp
    exp2
    expm1
    fake_quantize_per_channel_affine
    fake_quantize_per_tensor_affine
    fix
    float_power
    floor
    floor_divide
    fmod
    frac
    frexp
    gradient
    imag
    ldexp
    lerp
    lgamma
    log
    log10
    log1p
    log2
    logaddexp
    logaddexp2
    logical_and
    logical_not
    logical_or
    logical_xor
    logit
    hypot
    i0
    igamma
    igammac
    mul
    multiply
    mvlgamma
    nan_to_num
    neg
    negative
    nextafter
    polygamma
    positive
    pow
    quantized_batch_norm
    quantized_max_pool1d
    quantized_max_pool2d
    rad2deg
    real
    reciprocal
    remainder
    round
    rsqrt
    sigmoid
    sign
    sgn
    signbit
    sin
    sinc
    sinh
    softmax
    sqrt
    square
    sub
    subtract
    tan
    tanh
    true_divide
    trunc
    xlogy

عمليات التخفيض
~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    argmax
    argmin
    amax
    amin
    aminmax
    all
    any
    max
    min
    dist
    logsumexp
    mean
    nanmean
    median
    nanmedian
    mode
    norm
    nansum
    prod
    quantile
    nanquantile
    std
    std_mean
    sum
    unique
    unique_consecutive
    var
    var_mean
    count_nonzero

عمليات المقارنة
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    allclose
    argsort
    eq
    equal
    ge
    greater_equal
    gt
    greater
    isclose
    isfinite
    isin
    isinf
    isposinf
    isneginf
    isnan
    isreal
    kthvalue
    le
    less_equal
    lt
    less
    maximum
    minimum
    fmax
    fmin
    ne
    not_equal
    sort
    topk
    msort


العمليات الطيفية
~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    stft
    istft
    bartlett_window
    blackman_window
    hamming_window
    hann_window
    kaiser_window


عمليات أخرى
~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    atleast_1d
    atleast_2d
    atleast_3d
    bincount
    block_diag
    broadcast_tensors
    broadcast_to
    broadcast_shapes
    bucketize
    cartesian_prod
    cdist
    clone
    combinations
    corrcoef
    cov
    cross
    cummax
    cummin
    cumprod
    cumsum
    diag
    diag_embed
    diagflat
    diagonal
    diff
    einsum
    flatten
    flip
    fliplr
    flipud
    kron
    rot90
    gcd
    histc
    histogram
    histogramdd
    meshgrid
    lcm
    logcumsumexp
    ravel
    renorm
    repeat_interleave
    roll
    searchsorted
    tensordot
    trace
    tril
    tril_indices
    triu
    triu_indices
    unflatten
    vander
    view_as_real
    view_as_complex
    resolve_conj
    resolve_neg


عمليات BLAS و LAPACK
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    addbmm
    addmm
    addmv
    addr
    baddbmm
    bmm
    chain_matmul
    cholesky
    cholesky_inverse
    cholesky_solve
    dot
    geqrf
    ger
    inner
    inverse
    det
    logdet
    slogdet
    lu
    lu_solve
    lu_unpack
    matmul
    matrix_power
    matrix_exp
    mm
    mv
    orgqr
    ormqr
    outer
    pinverse
    qr
    svd
    svd_lowrank
    pca_lowrank
    lobpcg
    trapz
    trapezoid
    cumulative_trapezoid
    triangular_solve
    vdot

عمليات ForEach
~~~~~~~~~~~~~~~~~~

.. warning::
    هذا API في مرحلة البيتا وقد يخضع لتغييرات في المستقبل.
    لا يتم دعم طريقة AD للأمام.

.. autosummary::
    :toctree: generated
    :nosignatures:

    _foreach_abs
    _foreach_abs_
    _foreach_acos
    _foreach_acos_
    _foreach_asin
    _foreach_asin_
    _foreach_atan
    _foreach_atan_
    _foreach_ceil
    _foreach_ceil_
    _foreach_cos
    _foreach_cos_
    _foreach_cosh
    _foreach_cosh_
    _foreach_erf
    _foreach_erf_
    _foreach_erfc
    _foreach_erfc_
    _foreach_exp
    _foreach_exp_
    _foreach_expm1
    _foreach_expm1_
    _foreach_floor
    _foreach_floor_
    _foreach_log
    _foreach_log_
    _foreach_log10
    _foreach_log10_
    _foreach_log1p
    _foreach_log1p_
    _foreach_log2
    _foreach_log2_
    _foreach_neg
    _foreach_neg_
    _foreach_tan
    _foreach_tan_
    _foreach_sin
    _foreach_sin_
    _foreach_sinh
    _foreach_sinh_
    _foreach_round
    _foreach_round_
    _foreach_sqrt
    _foreach_sqrt_
    _foreach_lgamma
    _foreach_lgamma_
    _foreach_frac
    _foreach_frac_
    _foreach_reciprocal
    _foreach_reciprocal_
    _foreach_sigmoid
    _foreach_sigmoid_
    _foreach_trunc
    _foreach_trunc_
    _foreach_zero_

مرافق
-----
.. autosummary::
    :toctree: generated
    :nosignatures:

    compiled_with_cxx11_abi
    result_type
    can_cast
    promote_types
    use_deterministic_algorithms
    are_deterministic_algorithms_enabled
    is_deterministic_algorithms_warn_only_enabled
    set_deterministic_debug_mode
    get_deterministic_debug_mode
    set_float32_matmul_precision
    get_float32_matmul_precision
    set_warn_always
    get_device_module
    is_warn_always_enabled
    vmap
    _assert

الأرقام الرمزية
----------------
.. autoclass:: SymInt
    :members:

.. autoclass:: SymFloat
    :members:

.. autoclass:: SymBool
    :members:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sym_float
    sym_int
    sym_max
    sym_min
    sym_not
    sym_ite

مسار التصدير
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

.. warning::
    هذه الميزة هي نموذج أولي وقد تخضع لتغييرات تتعارض مع التوافق في المستقبل.

    export
    generated/exportdb/index

تدفق التحكم
---------

.. warning::
    هذه الميزة هي نموذج أولي وقد تخضع لتغييرات تتعارض مع التوافق في المستقبل.

.. autosummary::
    :toctree: generated
    :nosignatures:

    cond

التحسين
-------
.. autosummary::
    :toctree: generated
    :nosignatures:

    compile

`وثائق torch.compile <https://pytorch.org/docs/main/torch.compiler.html>`__

علامات المشغل
---------------
.. autoclass:: Tag
    :members:

.. تم إضافة وحدات فرعية فارغة فقط لأغراض التتبع.
.. py:module:: torch.contrib
.. py:module:: torch.utils.backcompat

.. هذه الوحدة تستخدم داخليا فقط لبناء ROCm.
.. py:module:: torch.utils.hipify

.. تحتاج هذه الوحدة إلى توثيق. نقوم بإضافتها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.utils.model_dump
.. py:module:: torch.utils.viz
.. py:module:: torch.functional
.. py:module:: torch.quasirandom
.. py:module:: torch.return_types
.. py:module:: torch.serialization
.. py:module:: torch.signal.windows.windows
.. py:module:: torch.sparse.semi_structured
.. py:module:: torch.storage
.. py:module:: torch.torch_version
.. py:module:: torch.types
.. py:module:: torch.version