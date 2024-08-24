.. currentmodule:: torch

.. _tensor-doc:

torch.Tensor
====================

:class:`torch.Tensor` هو مصفوفة متعددة الأبعاد تحتوي على عناصر من نوع بيانات واحد.

أنواع البيانات
----------

يحدد تورتش أنواع التنسورات التالية مع أنواع البيانات التالية:

======================================= ===========================================
نوع البيانات                               dtype
======================================= ===========================================
نقطة عائمة 32 بت                   ``torch.float32`` أو ``torch.float``
نقطة عائمة 64 بت                   ``torch.float64`` أو ``torch.double``
نقطة عائمة 16 بت [1]_              ``torch.float16`` أو ``torch.half``
نقطة عائمة 16 بت [2]_              ``torch.bfloat16``
مركب 32 بت                          ``torch.complex32`` أو ``torch.chalf``
مركب 64 بت                          ``torch.complex64`` أو ``torch.cfloat``
مركب 128 بت                         ``torch.complex128`` أو ``torch.cdouble``
عدد صحيح 8 بت (غير موقع)                ``torch.uint8``
عدد صحيح 16 بت (غير موقع)               ``torch.uint16`` (دعم محدود) [4]_
عدد صحيح 32 بت (غير موقع)               ``torch.uint32`` (دعم محدود) [4]_
عدد صحيح 64 بت (غير موقع)               ``torch.uint64`` (دعم محدود) [4]_
عدد صحيح 8 بت (موقّع)                  ``torch.int8``
عدد صحيح 16 بت (موقّع)                 ``torch.int16`` أو ``torch.short``
عدد صحيح 32 بت (موقّع)                 ``torchMultiplier.int32`` أو ``torch.int``
عدد صحيح 64 بت (موقّع)                 ``torch.int64`` أو ``torch.long``
منطقي                                 ``torch.bool``
كمي 8 بت عدد صحيح (غير موقع)      ``torch.quint8``
كمي 8 بت عدد صحيح (موقّع)        ``torch.qint8``
كمي 32 بت عدد صحيح (موقّع)       ``torch.qint32``
كمي 4 بت عدد صحيح (غير موقع) [3]_ ``torch.quint4x2``
نقطة عائمة 8 بت، e4m3 [5]_         ``torch.float8_e4m3fn`` (دعم محدود)
نقطة عائمة 8 بت، e5m2 [5]_         ``torch.float8_e5m2`` (دعم محدود)
======================================= ===========================================

.. [1]
  يشار إليه أحيانًا باسم binary16: يستخدم 1 بت للإشارة، و5 للأس، و10
  بت للدلالة. مفيد عندما تكون الدقة مهمة على حساب النطاق.
.. [2]
  يشار إليه أحيانًا باسم Brain Floating Point: يستخدم 1 بت للإشارة، و8 للأس، و7
  بت للدلالة. مفيد عندما يكون النطاق مهمًا، حيث أن له نفس
  عدد بتات الأس مثل ``float32``
.. [3]
  يتم تخزين العدد الصحيح الكمي 4 بت على أنه عدد صحيح موقّع 8 بت. حاليًا، فهو مدعوم فقط في مشغل EmbeddingBag.
.. [4]
  من المخطط حاليًا أن يكون للأنواع غير الموقّعة بخلاف ``uint8`` دعم محدود فقط في الوضع المتلهف (فهي موجودة بشكل أساسي للمساعدة في الاستخدام مع
  torch.compile)؛ إذا كنت بحاجة إلى دعم متلهف والنطاق الإضافي غير مطلوب،
  نوصي باستخدام متغيراتها الموقّعة بدلاً من ذلك.  راجع
  https://github.com/pytorch/pytorch/issues/58734 لمزيد من التفاصيل.
.. [5]
  ``torch.float8_e4m3fn`` و ``torch.float8_e5m2`` ينفذان مواصفات أنواع النقطة العائمة 8 بت من https://arxiv.org/abs/2209.05433. دعم
  العملية محدود للغاية.

توافقًا مع الإصدارات السابقة، ندعم أسماء الفئات البديلة التالية
لهذه أنواع البيانات:

======================================= ============================= ================================
نوع البيانات                               التنسور CPU                    التنسور GPU
======================================= ============================= ================================
نقطة عائمة 32 بت                   :class:`torch.FloatTensor`    :class:`torch.cuda.FloatTensor`
نقطة عائمة 64 بت                   :class:`torch.DoubleTensor`   :class:`torch.cuda.DoubleTensor`
نقطة عائمة 16 بت                   :class:`torch.HalfTensor`     :class:`torch.cuda.HalfTensor`
نقطة عائمة 16 بت                   :class:`torch.BFloat16Tensor` :class:`torch.cuda.BFloat16Tensor`
عدد صحيح 8 بت (غير موقع)                :class:`torch.ByteTensor`     :class:`torch.cuda.ByteTensor`
عدد صحيح 8 بت (موقّع)                  :class:`torch.CharTensor`     :class:`torch.cuda.CharTensor`
عدد صحيح 16 بت (موقّع)                 :class:`torch.ShortTensor`    :class:`torch.cuda.ShortTensor`
عدد صحيح 32 بت (موقّع)                 :class:`torch.IntTensor`      :class:`torch.cuda.IntTensor`
عدد صحيح 64 بت (موقّع)                 :class:`torch.LongTensor`     :class:`torch.cuda.LongTensor`
منطقي                                 :class:`torch.BoolTensor`     :class:`torch.cuda.BoolTensor`
======================================= ============================= ================================

ومع ذلك، لبناء التنسورات، نوصي باستخدام وظائف المصنع مثل
:func:`torch.empty` مع وسيط ``dtype`` بدلاً من ذلك.  هو
:class:`torch.Tensor` الباني هو مرادف لنوع التنسور الافتراضي
(:class:`torch.FloatTensor`).

التهيئة والعمليات الأساسية
---------------------

يمكن إنشاء تنسور من قائمة Python أو تسلسل باستخدام
:func:`torch.tensor` الباني:

::

    >>> torch.tensor([[1., -1.], [1., -1.]])
    tensor([[ 1.0000, -1.0000],
            [ 1.0000, -1.0000]])
    >>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])

.. warning::

    :func:`torch.tensor` ينسخ دائمًا :attr:`data`. إذا كان لديك Tensor
    :attr:`data` وتريد فقط تغيير علم ``requires_grad`` الخاص به، استخدم
    :meth:`~torch.Tensor.requires_grad_` أو
    :meth:`~torch.Tensor.detach` لتجنب النسخ.
    إذا كان لديك مصفوفة نومبي وتريد تجنب النسخ، استخدم
    :func:`torch.as_tensor`.

يمكن إنشاء تنسور من نوع بيانات محدد عن طريق تمرير
:class:`torch.dtype` و/أو :class:`torch.device` إلى
باني أو عملية إنشاء تنسور:

::

    >>> torch.zeros([2, 4], dtype=torch.int32)
    tensor([[ 0,  0,  0,  0],
            [ 0,  0,  0,  0]], dtype=torch.int32)
    >>> cuda0 = torch.device('cuda:0')
    >>> torch.ones([2, 4], dtype=torch.float64, device=cuda0)
    tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')

للحصول على مزيد من المعلومات حول بناء التنسورات، راجع :ref:`tensor-creation-ops`


يمكن الوصول إلى محتويات التنسور وتعديلها باستخدام ترميز الفهرسة
والشرائح في Python:

::

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> print(x[1][2])
    tensor(6)
    >>> x[0][1] = 8
    >>> print(x)
    tensor([[ 1,  8,  3],
            [ 4,  5,  6]])

استخدم :meth:`torch.Tensor.item` للحصول على رقم Python من تنسور تحتوي على قيمة واحدة:

::

    >>> x = torch.tensor([[1]])
    >>> x
    tensor([[ 1]])
    >>> x.item()
    1
    >>> x = torch.tensor(2.5)
    >>> x
    tensor(2.5000)
    >>> x.item()
    2.5

للحصول على مزيد من المعلومات حول الفهرسة، راجع :ref:`indexing-slicing-joining`

يمكن إنشاء تنسور مع :attr:`requires_grad=True` بحيث
:mod:`torch.autograd` يسجل العمليات عليها للتفاضل التلقائي.

::

    >>> x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
    >>> out = x.pow(2).sum()
    >>> out.backward()
    >>> x.grad
    tensor([[ 2.0000, -2.0000],
            [ 2.0000,  2.0000]])

يرتبط بكل تنسور :class:`torch.Storage`، والذي يحتفظ ببياناته.
توفر فئة التنسور أيضًا طريقة عرض متعددة الأبعاد، `strided <https://en.wikipedia.org/wiki/Stride_of_an_array>`_
عرض التخزين، ويحدد العمليات الرقمية عليه.

.. note::
   للحصول على مزيد من المعلومات حول طرق عرض التنسورات، راجع :ref:`tensor-view-doc`.

.. note::
   للحصول على مزيد من المعلومات حول سمات :class:`torch.dtype`، و:class:`torch.device`، و
   :class:`torch.layout` من :class:`torch.Tensor`، راجع
   :ref:`tensor-attributes-doc`.

.. note::
   يتم تمييز الطرق التي تطفر تنسور بلاحقة شرطة سفلية.
   على سبيل المثال، :func:`torch.FloatTensor.abs_` يحسب القيمة المطلقة
   في المكان ويعيد التنسور المعدلة، في حين :func:`torch.FloatTensor.abs`
   يحسب النتيجة في تنسور جديدة.

.. note::
   لتغيير :class:`torch.device` و/أو :class:`torch.dtype` لتنسور موجودة، ضع في اعتبارك استخدام
   :meth:`~torch.Tensor.to` طريقة على التنسور.

.. warning::
   التنفيذ الحالي لـ :class:`torch.Tensor` يقدم ذاكرة إضافية،
   وبالتالي قد يؤدي إلى استخدام ذاكرة مرتفع بشكل غير متوقع في التطبيقات التي تحتوي على العديد من التنسورات الصغيرة.
   إذا كان هذا هو الحال لديك، فكر في استخدام بنية كبيرة واحدة.


مرجع فئة التنسور
-------------

.. class:: Tensor()

   هناك بضع طرق رئيسية لإنشاء Tensor، وذلك يعتمد على استخدامك.

   - لإنشاء Tensor ببيانات موجودة مسبقًا، استخدم :func:`torch.tensor`.
   - لإنشاء Tensor بحجم محدد، استخدم ``torch.*`` عمليات إنشاء Tensor
     (انظر :ref:`tensor-creation-ops`).
   - لإنشاء Tensor بنفس حجم (وأنواع مماثلة) مثل Tensor أخرى،
     استخدم ``torch.*_like`` عمليات إنشاء Tensor
     (انظر :ref:`tensor-creation-ops`).
   - لإنشاء Tensor بنوع مماثل ولكن بحجم مختلف عن Tensor أخرى،
     استخدم عمليات الإنشاء ``tensor.new_*``.
   - هناك بناء قديم ``torch.Tensor`` لا يُنصح باستخدامه.
     استخدم :func:`torch.tensor` بدلاً من ذلك.

.. method:: Tensor.__init__(self, data)

   هذا الباني (البناء) قديم، نوصي باستخدام :func:`torch.tensor` بدلاً من ذلك.
   يعتمد ما يفعله هذا الباني على نوع ``data``.

   * إذا كان ``data`` عبارة عن Tensor، فإنه يعيد مرجعًا إلى Tensor الأصلي. على عكس
     :func:`torch.tensor`، فإن هذا يتتبع autograd وينشر التدرجات إلى
     Tensor الأصلي. لا يتم دعم وسيط device لـ ``data`` هذا النوع.

   * إذا كان ``data`` تسلسلاً أو تسلسلاً متداخلاً، قم بإنشاء Tensor من النوع الافتراضي
     (عادة ``torch.float32``) تكون بياناته هي القيم الموجودة في التسلسلات،
     وإجراء الإكراهات إذا لزم الأمر. تجدر الإشارة إلى أن هذا يختلف عن
     :func:`torch.tensor` في أن هذا الباني سينشئ دائمًا Tensor ذات أعداد الفاصلة العائمة،
     حتى إذا كانت جميع المدخلات أعدادًا صحيحة.

   * إذا كان ``data`` عبارة عن :class:`torch.Size`، فإنه يعيد Tensor فارغة بذلك الحجم.

   لا يدعم هذا الباني تحديد ``dtype`` أو ``device`` لـ Tensor المعادة. نوصي باستخدام
     :func:`torch.tensor` الذي يوفر هذه الوظيفة.

   Args:
       data (array_like): Tensor المراد إنشاؤها منه.

   Keyword args:
       device (:class:`torch.device`, optional): الجهاز المطلوب لـ Tensor المعادة.
           الافتراضي: إذا كان None، نفس :class:`torch.device` مثل هذه Tensor.

.. autoattribute:: Tensor.T
.. autoattribute:: Tensor.H
.. autoattribute:: Tensor.mT
.. autoattribute:: Tensor.mH

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.new_tensor
    Tensor.new_full
    Tensor.new_empty
    Tensor.new_ones
    Tensor.new_zeros

    Tensor.is_cuda
    Tensor.is_quantized
    Tensor.is_meta
    Tensor.device
    Tensor.grad
    Tensor.ndim
    Tensor.real
    Tensor.imag
    Tensor.nbytes
    Tensor.itemsize
    
    Tensor.abs
    Tensor.abs_
    Tensor.absolute
    Tensor.absolute_
    Tensor.acos
    Tensor.acos_
    Tensor.arccos
    Tensor.arccos_
    Tensor.add
    Tensor.add_
    Tensor.addbmm
    Tensor.addbmm_
    Tensor.addcdiv
    Tensor.addcdiv_
    Tensor.addcmul
    Tensor.addcmul_
    Tensor.addmm
    Tensor.addmm_
    Tensor.sspaddmm
    Tensor.addmv
    Tensor.addmv_
    Tensor.addr
    Tensor.addr_
    Tensor.adjoint
    Tensor.allclose
    Tensor.amax
    Tensor.amin
    Tensor.aminmax
    Tensor.angle
    Tensor.apply_
    Tensor.argmax
    Tensor.argmin
    Tensor.argsort
    Tensor.argwhere
    Tensor.asin
    Tensor.asin_
    Tensor.arcsin
    Tensor.arcsin_
    Tensor.as_strided
    Tensor.atan
    Tensor.atan_
    Tensor.arctan
    Tensor.arctan_
    Tensor.atan2
    Tensor.atan2_
    Tensor.arctan2
    Tensor.arctan2_
    Tensor.all
    Tensor.any
    Tensor.backward
    Tensor.baddbmm
    Tensor.baddbmm_
    Tensor.bernoulli
    Tensor.bernoulli_
    Tensor.bfloat16
    Tensor.bincount
    Tensor.bitwise_not
    Tensor.bitwise_not_
    Tensor.bitwise_and
    Tensor.bitwise_and_
    Tensor.bitwise_or
    Tensor.bitwise_or_
    Tensor.bitwise_xor
    Tensor.bitwise_xor_
    Tensor.bitwise_left_shift
    Tensor.bitwise_left_shift_
    Tensor.bitwise_right_shift
    Tensor.bitwise_right_shift_
    Tensor.bmm
    Tensor.bool
    Tensor.byte
    Tensor.broadcast_to
    Tensor.cauchy_
    Tensor.ceil
    Tensor.ceil_
    Tensor.char
    Tensor.cholesky
    Tensor.cholesky_inverse
    Tensor.cholesky_solve
    Tensor.chunk
    Tensor.clamp
    Tensor.clamp_
    Tensor.clip
    Tensor.clip_
    Tensor.clone
    Tensor.contiguous
    Tensor.copy_
    Tensor.conj
    Tensor.conj_physical
    Tensor.conj_physical_
    Tensor.resolve_conj
    Tensor.resolve_neg
    Tensor.copysign
    Tensor.copysign_
    Tensor.cos
    Tensor.cos_
    Tensor.cosh
    Tensor.cosh_
    Tensor.corrcoef
    Tensor.count_nonzero
    Tensor.cov
    Tensor.acosh
    Tensor.acosh_
    Tensor.arccosh
    Tensor.arccosh_
    Tensor.cpu
    Tensor.cross
    Tensor.cuda
    Tensor.logcumsumexp
    Tensor.cummax
    Tensor.cummin
    Tensor.cumprod
    Tensor.cumprod_
    Tensor.cumsum
    Tensor.cumsum_
    Tensor.chalf
    Tensor.cfloat
    Tensor.cdouble
    Tensor.data_ptr
    Tensor.deg2rad
    Tensor.dequantize
    Tensor.det
    Tensor.dense_dim
    Tensor.detach
    Tensor.detach_
    Tensor.diag
    Tensor.diag_embed
    Tensor.diagflat
    Tensor.diagonal
    Tensor.diagonal_scatter
    Tensor.fill_diagonal_
    Tensor.fmax
    Tensor.fmin
    Tensor.diff
    Tensor.digamma
    Tensor.digamma_
    Tensor.dim
    Tensor.dim_order
    Tensor.dist
    Tensor.div
    Tensor.div_
    Tensor.divide
    Tensor.divide_
    Tensor.dot
    Tensor.double
    Tensor.dsplit
    Tensor.element_size
    Tensor.eq
    Tensor.eq_
    Tensor.equal
    Tensor.erf
    Tensor.erf_
    Tensor.erfc
    Tensor.erfc_
    Tensor.erfinv
    Tensor.erfinv_
    Tensor.exp
    Tensor.exp_
    Tensor.expm1
    Tensor.expm1_
    Tensor.expand
    Tensor.expand_as
    Tensor.exponential_
    Tensor.fix
    Tensor.fix_
    Tensor.fill_
    Tensor.flatten
    Tensor.flip
    Tensor.fliplr
    Tensor.flipud
    Tensor.float
    Tensor.float_power
    Tensor.float_power_
    Tensor.floor
    Tensor.floor_
    Tensor.floor_divide
    Tensor.floor_divide_
    Tensor.fmod
    Tensor.fmod_
    Tensor.frac
    Tensor.frac_
    Tensor.frexp
    Tensor.gather
    Tensor.gcd
    Tensor.gcd_
    Tensor.ge
    Tensor.ge_
    Tensor.greater_equal
    Tensor.greater_equal_
    Tensor.geometric_
    Tensor.geqrf
    Tensor.ger
    Tensor.get_device
    Tensor.gt
    Tensor.gt_
    Tensor.greater
    Tensor.greater_
    Tensor.half
    Tensor.hardshrink
    Tensor.heaviside
    Tensor.histc
    Tensor.histogram
    Tensor.hsplit
    Tensor.hypot
    Tensor.hypot_
    Tensor.i0
    Tensor.i0_
    Tensor.igamma
    Tensor.igamma_
    Tensor.igammac
    Tensor.igammac_
    Tensor.index_add_
    Tensor.index_add
    Tensor.index_copy_
    Tensor.index_copy
    Tensor.index_fill_
    Tensor.index_fill
    Tensor.index_put_
    Tensor.index_put
    Tensor.index_reduce_
    Tensor.index_reduce
    Tensor.index_select
    Tensor.indices
    Tensor.inner
    Tensor.int
    Tensor.int_repr
    Tensor.inverse
    Tensor.isclose
    Tensor.isfinite
    Tensor.isinf
    Tensor.isposinf
    Tensor.isneginf
    Tensor.isnan
    Tensor.is_contiguous
    Tensor.is_complex
    Tensor.is_conj
    Tensor.is_floating_point
    Tensor.is_inference
    Tensor.is_leaf
    Tensor.is_pinned
    Tensor.is_set_to
    Tensor.is_shared
    Tensor.is_signed
    Tensor.is_sparse
    Tensor.istft
    Tensor.isreal
    Tensor.item
    Tensor.kthvalue
    Tensor.lcm
    Tensor.lcm_
    Tensor.ldexp
    Tensor.ldexp_
    Tensor.le
    Tensor.le_
    Tensor.less_equal
    Tensor.less_equal_
    Tensor.lerp
    Tensor.lerp_
    Tensor.lgamma
    Tensor.lgamma_
    Tensor.log
    Tensor.log_
    Tensor.logdet
    Tensor.log10
    Tensor.log10_
    Tensor.log1p
    Tensor.log1p_
    Tensor.log2
    Tensor.log2_
    Tensor.log_normal_
    Tensor.logaddexp
    Tensor.logaddexp2
    Tensor.logsumexp
    Tensor.logical_and
    Tensor.logical_and_
    Tensor.logical_not
    Tensor.logical_not_
    Tensor.logical_or
    Tensor.logical_or_
    Tensor.logical_xor
    Tensor.logical_xor_
    Tensor.logit
    Tensor.logit_
    Tensor.long
    Tensor.lt
    Tensor.lt_
    Tensor.less
    Tensor.less_
    Tensor.lu
    Tensor.lu_solve
    Tensor.as_subclass
    Tensor.map_
    Tensor.masked_scatter_
    Tensor.masked_scatter
    Tensor.masked_fill_
    Tensor.masked_fill
    Tensor.masked_select
    Tensor.matmul
    Tensor.matrix_power
    Tensor.matrix_exp
    Tensor.max
    Tensor.maximum
    Tensor.mean
    Tensor.module_load
    Tensor.nanmean
    Tensor.median
    Tensor.nanmedian
    Tensor.min
    Tensor.minimum
    Tensor.mm
    Tensor.smm
    Tensor.mode
    Tensor.movedim
    Tensor.moveaxis
    Tensor.msort
    Tensor.mul
    Tensor.mul_
    Tensor.multiply
    Tensor.multiply_
    Tensor.multinomial
    Tensor.mv
    Tensor.mvlgamma
    Tensor.mvlgamma_
    Tensor.nansum
    Tensor.narrow
    Tensor.narrow_copy
    Tensor.ndimension
    Tensor.nan_to_num
    Tensor.nan_to_num_
    Tensor.ne
    Tensor.ne_
    Tensor.not_equal
    Tensor.not_equal_
    Tensor.neg
    Tensor.neg_
    Tensor.negative
    Tensor.negative_
    Tensor.nelement
    Tensor.nextafter
    Tensor.nextafter_
    Tensor.nonzero
    Tensor.norm
    Tensor.normal_
    Tensor.numel
    Tensor.numpy
    Tensor.orgqr
    Tensor.ormqr
    Tensor.outer
    Tensor.permute
    Tensor.pin_memory
    Tensor.pinverse
    Tensor.polygamma
    Tensor.polygamma_
    Tensor.positive
    Tensor.pow
    Tensor.pow_
    Tensor.prod
    Tensor.put_
    Tensor.qr
    Tensor.qscheme
    Tensor.quantile
    Tensor.nanquantile
    Tensor.q_scale
    Tensor.q_zero_point
    Tensor.q_per_channel_scales
    Tensor.q_per_channel_zero_points
    Tensor.q_per_channel_axis
    Tensor.rad2deg
    Tensor.random_
    Tensor.ravel
    Tensor.reciprocal
    Tensor.reciprocal_
    Tensor.record_stream
    Tensor.register_hook
    Tensor.register_post_accumulate_grad_hook
    Tensor.remainder
    Tensor.remainder_
    Tensor.renorm
    Tensor.renorm_
    Tensor.repeat
    Tensor.repeat_interleave
    Tensor.requires_grad
    Tensor.requires_grad_
    Tensor.reshape
    Tensor.reshape_as
    Tensor.resize_
    Tensor.resize_as_
    Tensor.retain_grad
    Tensor.retains_grad
    Tensor.roll
    Tensor.rot90
    Tensor.round
    Tensor.round_
    Tensor.rsqrt
    Tensor.rsqrt_
    Tensor.scatter
    Tensor.scatter_
    Tensor.scatter_add_
    Tensor.scatter_add
    Tensor.scatter_reduce_
    Tensor.scatter_reduce
    Tensor.select
    Tensor.select_scatter
    Tensor.set_
    Tensor.share_memory_
    Tensor.short
    Tensor.sigmoid
    Tensor.sigmoid_
    Tensor.sign
    Tensor.sign_
    Tensor.signbit
    Tensor.sgn
    Tensor.sgn_
    Tensor.sin
    Tensor.sin_
    Tensor.sinc
    Tensor.sinc_
    Tensor.sinh
    Tensor.sinh_
    Tensor.asinh
    Tensor.asinh_
    Tensor.arcsinh
    Tensor.arcsinh_
    Tensor.shape
    Tensor.size
    Tensor.slogdet
    Tensor.slice_scatter
    Tensor.softmax
    Tensor.sort
    Tensor.split
    Tensor.sparse_mask
    Tensor.sparse_dim
    Tensor.sqrt
    Tensor.sqrt_
    Tensor.square
    Tensor.square_
    Tensor.squeeze
    Tensor.squeeze_
    Tensor.std
    Tensor.stft
    Tensor.storage
    Tensor.untyped_storage
    Tensor.storage_offset
    Tensor.storage_type
    Tensor.stride
    Tensor.sub
    Tensor.sub_
    Tensor.subtract
    Tensor.subtract_
    Tensor.sum
    Tensor.sum_to_size
    Tensor.svd
    Tensor.swapaxes
    Tensor.swapdims
    Tensor.t
    Tensor.t_
    Tensor.tensor_split
    Tensor.tile
    Tensor.to
    Tensor.to_mkldnn
    Tensor.take
    Tensor.take_along_dim
    Tensor.tan
    Tensor.tan_
    Tensor.tanh
    Tensor.tanh_
    Tensor.atanh
    Tensor.atanh_
    Tensor.arctanh
    Tensor.arctanh_
    Tensor.tolist
    Tensor.topk
    Tensor.to_dense
    Tensor.to_sparse
    Tensor.to_sparse_csr
    Tensor.to_sparse_csc
    Tensor.to_sparse_bsr
    Tensor.to_sparse_bsc
    Tensor.trace
    Tensor.transpose
    Tensor.transpose_
    Tensor.triangular_solve
    Tensor.tril
    Tensor.tril_
    Tensor.triu
    Tensor.triu_
    Tensor.true_divide
    Tensor.true_divide_
    Tensor.trunc
    Tensor.trunc_
    Tensor.type
    Tensor.type_as
    Tensor.unbind
    Tensor.unflatten
    Tensor.unfold
    Tensor.uniform_
    Tensor.unique
    Tensor.unique_consecutive
    Tensor.unsqueeze
    Tensor.unsqueeze_
    Tensor.values
    Tensor.var
    Tensor.vdot
    Tensor.view
    Tensor.view_as
    Tensor.vsplit
    Tensor.where
    Tensor.xlogy
    Tensor.xlogy_
    Tensor.xpu
    Tensor.zero_