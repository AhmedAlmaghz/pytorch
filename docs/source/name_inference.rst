.. currentmodule:: torch

.. _name_inference_reference-doc:

تغطية مشغل Named Tensors 

فيما يلي قائمة بالعمليات التي تدعمها Named Tensors. إذا كانت هناك أي اختلافات في السلوك بين Named و Standard Tensors، فسيتم توضيحها.

العمليات العامة

============

.. currentmodule:: torch

.. autosummary::
   :nosignatures:

   ~Tensor.names
   ~Tensor.rename
   ~Tensor.rename_
   ~Tensor.align_to
   ~Tensor.align_as
   ~Tensor.refine_names
   ~Tensor.moveaxis
   ~Tensor.as_strided
   ~Tensor.split
   ~Tensor.unbind
   ~Tensor.meshgrid
   ~Tensor.stack
   ~Tensor.unbind
   ~Tensor.unsqueeze
   ~Tensor.squeeze
   ~Tensor.swapaxes
   ~Tensor.swapdims
   ~Tensor.t
   ~Tensor.permute
   ~Tensor.transpose
   ~Tensor.view
   ~Tensor.reshape
   ~Tensor.resize_
   ~Tensor.resize_as_
   ~Tensor.flatten
   ~Tensor.unflatten
   ~Tensor.flip
   ~Tensor.roll
   ~Tensor.rot90
   ~Tensor.moveaxis
   ~Tensor.expand
   ~Tensor.repeat
   ~Tensor.tile
   ~Tensor.gather
   ~Tensor.scatter
   ~Tensor.scatter_add
   ~Tensor.index_select
   ~Tensor.index_fill
   ~Tensor.take
   ~Tensor.put_
   ~Tensor.tril
   ~Tensor.triu
   ~Tensor.diagonal
   ~Tensor.diag_embed
   ~Tensor.diagflat
   ~Tensor.diag
   ~Tensor.trace
   ~Tensor.ne
   ~Tensor.eq
   ~Tensor.ge
   ~Tensor.le
   ~Tensor.gt
   ~Tensor.lt
   ~Tensor.abs
   ~Tensor.ceil
   ~Tensor.floor
   ~Tensor.round
   ~Tensor.trunc
   ~Tensor.fix
   ~Tensor.sign
   ~Tensor.relu
   ~Tensor.gelu
   ~Tensor.silu
   ~Tensor.mish
   ~Tensor.sigmoid
   ~Tensor.tanh
   ~Tensor.sin
   ~Tensor.cos
   ~Tensor.tan
   ~Tensor.asin
   ~Tensor.acos
   ~Tensor.atan
   ~Tensor.sinh
   ~Tensor.cosh
   ~Tensor.tanh
   ~Tensor.erf
   ~Tensor.erfc
   ~Tensor.exp
   ~Tensor.expm1
   ~Tensor.log
   ~Tensor.log10
   ~Tensor.log1p
   ~Tensor.log2
   ~Tensor.mm
   ~Tensor.bmm
   ~Tensor.addmm
   ~Tensor.addbmm
   ~Tensor.addmv
   ~Tensor.addr
   ~Tensor.mv
   ~Tensor.dot
   ~Tensor.add
   ~Tensor.sub
   ~Tensor.rsub
   ~Tensor.mul
   ~Tensor.div
   ~Tensor.true_divide
   ~Tensor.floordiv
   ~Tensor.pow
   ~Tensor.atan2
   ~Tensor.lerp
   ~Tensor.mean
   ~Tensor.sum
   ~Tensor.cumsum
   ~Tensor.std
   ~Tensor.var
   ~Tensor.median
   ~Tensor.mode
   ~Tensor.min
   ~Tensor.max
   ~Tensor.argmax
   ~Tensor.argmin
   ~Tensor.kthvalue
   ~Tensor.clone
   ~Tensor.zero_
   ~Tensor.copy_
   ~Tensor.normal_
   ~Tensor.fmod
   ~Tensor.remainder
   ~Tensor.clamp
   ~Tensor.clamp_min
   ~Tensor.clamp_max
   ~Tensor.cross
   ~Tensor.renorm
   ~Tensor.lerp
   ~Tensor.lerp_
   ~Tensor.histc
   ~Tensor.to
   ~Tensor.set_
   ~Tensor.get_device
   ~Tensor.type_as
   ~Tensor.to_dense
   ~Tensor.to_sparse
   ~Tensor.is_set_to
   ~Tensor.is_sparse
   ~Tensor.is_mkldnn
   ~Tensor.is_quantized
   ~Tensor.is_distributed
   ~Tensor.is_complex
   ~Tensor.is_floating_point
   ~Tensor.is_signed
   ~Tensor.is_inference
   ~Tensor.dim
   ~Tensor.nelement
   ~Tensor.numel
   ~Tensor.element_size
   ~Tensor.ndimension
   ~Tensor.numel
   ~Tensor.size
   ~Tensor.shape
   ~Tensor.setrequiresgrad
   ~Tensor.requires_grad
   ~Tensor.requires_grad_
   ~Tensor.grad
   ~Tensor.grad_fn
   ~Tensor.backward
   ~Tensor.backward
   ~Tensor.register_hook
   ~Tensor.register_backward_hook
   ~Tensor.register_forward_hook
   ~Tensor.register_forward_pre_hook
   ~Tensor.register_forward_hook
   ~Tensor.hooks
   ~Tensor.retain_grad
   ~Tensor.detach
   ~Tensor.detach_
   ~Tensor.detach_
   ~Tensor.detach_
   ~Tensor.detach_
   ~Tensor.detach_
   ~Tensor.item
   ~Tensor.cpu
   ~Tensor.cuda
   ~Tensor.to_mkldnn
   ~Tensor.to_sparse
   ~Tensor.to_dense
   ~Tensor.numpy
   ~Tensor.share_memory_
   ~Tensor.share_memory
   ~Tensor.pin_memory
   ~Tensor.pin_memory
   ~Tensor.record_stream
   ~Tensor.record_stream
   ~Tensor.record_stream
   ~Tensor.to_here
   ~Tensor.to_device
   ~Tensor.to_dense
   ~Tensor.to_mkldnn
   ~Tensor.to_sparse
   ~Tensor.to_param
   ~Tensor.to_parameter
   ~Tensor.to_dtype
   ~Tensor.to_device
   ~Tensor.to_sparse
   ~Tensor.to_dense
   ~Tensor.to_mkldnn
   ~Tensor.to_meta
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.to_other
   ~Tensor.toMultiplier
   ~Tensor.toType
   ~Tensor.toBackend
   ~Tensor.toScalar
   ~Tensor.toComplex
   ~Tensor.toReal
   ~Tensor.toHalf
   ~Tensor.toFloat
   ~Tensor.toDouble
   ~Tensor.toCDouble
   ~Tensor.toQInt8
   ~Tensor.toQUInt8
   ~Tensor.toQInt32
   ~Tensor.toBFloat16
   ~Tensor.dequantize
   ~Tensor.q_scale
   ~Tensor.q_zero_point
   ~Tensor.int_repr
   ~Tensor.qscheme
   ~Tensor.q_per_channel_scales
   ~Tensor.q_per_channel_zero_points
   ~Tensor.q_per_channel_axis
   ~Tensor.is_quantized
   ~Tensor.qscheme
   ~Tensor.q_scale
   ~TensorMultiplier
   ~TensorOptions
   ~dtype
   ~device
   ~layout
   ~requires_grad
   ~pin_memory
   ~memory_format
   ~is_complex
   ~is_floating_point
   ~is_signed
   ~is_inference
   ~scalar_type
   ~has_names
   ~names
   ~rename
   ~rename_
   ~align_to
   ~align_as
   ~refine_names
   ~dim
   ~nelement
   ~numel
   ~element_size
   ~ndimension
   ~numel
   ~size
   ~shape
   ~set_
   ~get_device
   ~type_as
   ~to_dense
   ~to_sparse
   ~is_set_to
   ~is_sparse
   ~is_mkldnn
   ~is_quantized
   ~is_distributed
   ~is_complex
   ~is_floating_point
   ~is_signed
   ~is_inference
   ~scalar_type
   ~has_names
   ~names
   ~rename
   ~rename_
   ~align_to
   ~align_as
   ~refine_names
   ~dim
   ~nelement
   ~numel
   ~element_

العمليات المنطقية

=================

.. currentmodule:: torch

.. autosummary::
   :nosignatures:

   ~Tensor.__lshift__
   ~Tensor.__rlshift__
   ~Tensor.logical_shift
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
   ~Tensor.logical_shift_
===============================

يرجى قراءة :ref:`named_tensors-doc` أولاً للتعرف على المقدمة عن tensors المسماة.

هذه الوثيقة هي مرجع لـ *استنتاج الاسم*، وهي عملية تحدد كيف
تستخدم المنسوجات المسماة الأسماء للتحقق من صحة وقت التشغيل التلقائي الإضافي:

1. توفر عمليات فحص صحة وقت التشغيل التلقائي الإضافية باستخدام الأسماء
2. تنشر الأسماء من المنسوجات المدخلة إلى المنسوجات الناتجة

فيما يلي قائمة بجميع العمليات المدعومة باستخدام المنسوجات المسماة
قواعد استنتاج الاسم المرتبطة بها.

إذا لم تشاهد عملية مدرجة هنا، ولكنها ستساعد في حالتك الاستخدامية، يرجى
«البحث عما إذا كان قد تم تقديم مشكلة بالفعل <https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22>`_ وإذا لم يكن الأمر كذلك، `قدم واحدة <https://github.com/pytorch/pytorch/issues/new/choose>`_.

.. تحذير ::
    واجهة برمجة تطبيقات المنسوجات المسماة تجريبية وقد تتغير.

.. جدول csv :: العمليات المدعومة
   : header: API، قاعدة استدلال الاسم
   : عرض: 20، 20
":meth:`Tensor.abs`, :func:`torch.abs`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.abs_`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.acos`, :func:`torch.acos`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.acos_`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.add`, :func:`torch.add`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:meth:`Tensor.add_`,:ref:`يوحد_الأسماء_من_الإدخالات-doc`
":meth:`Tensor.addmm`, :func:`torch.addmm`",:ref:`يحذف_الأبعاد-doc`
:meth:`Tensor.addmm_`,:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.addmv`, :func:`torch.addmv`",:ref:`يحذف_الأبعاد-doc`
:meth:`Tensor.addmv_`,:ref:`يحذف_الأبعاد-doc`
:meth:`Tensor.align_as`, راجع التوثيق
:meth:`Tensor.align_to`, راجع التوثيق
":meth:`Tensor.all`, :func:`torch.all`",None
":meth:`Tensor.any`, :func:`torch.any`",None
":meth:`Tensor.asin`, :func:`torch.asin`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.asin_`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.atan`, :func:`torch.atan`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.atan2`, :func:`torch.atan2`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:meth:`Tensor.atan2_`,:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:meth:`Tensor.atan_`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.bernoulli`, :func:`torch.bernoulli`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.bernoulli_`,None
:meth:`Tensor.bfloat16`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.bitwise_not`, :func:`torch.bitwise_not`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.bitwise_not_`,None
":meth:`Tensor.bmm`, :func:`torch.bmm`",:ref:`يحذف_الأبعاد-doc`
:meth:`Tensor.bool`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.byte`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
:func:`torch.cat`,:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:meth:`Tensor.cauchy_`,None
":meth:`Tensor.ceil`, :func:`torch.ceil`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.ceil_`,None
:meth:`Tensor.char`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.chunk`, :func:`torch.chunk`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.clamp`, :func:`torch.clamp`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.clamp_`,None
:meth:`Tensor.copy_`,:ref:`دلالية_وظيفة_الإخراج-doc`
":meth:`Tensor.cos`, :func:`torch.cos`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.cos_`,None
":meth:`Tensor.cosh`, :func:`torch.cosh`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.cosh_`,None
":meth:`Tensor.acosh`, :func:`torch.acosh`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.acosh_`,None
:meth:`Tensor.cpu`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.cuda`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.cumprod`, :func:`torch.cumprod`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.cumsum`, :func:`torch.cumsum`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.data_ptr`,None
":meth:`Tensor.deg2rad`, :func:`torch.deg2rad`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.deg2rad_`,None
":meth:`Tensor.detach`, :func:`torch.detach`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.detach_`,None
":attr:`Tensor.device`, :func:`torch.device`",None
":meth:`Tensor.digamma`, :func:`torch.digamma`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.digamma_`,None
:meth:`Tensor.dim`,None
":meth:`Tensor.div`, :func:`torch.div`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:meth:`Tensor.div_`,:ref:`يوحد_الأسماء_من_الإدخالات-doc`
":meth:`Tensor.dot`, :func:`torch.dot`",None
:meth:`Tensor.double`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.element_size`,None
:func:`torch.empty`,:ref:`مصنع-doc`
:func:`torch.empty_like`,:ref:`مصنع-doc`
":meth:`Tensor.eq`, :func:`torch.eq`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
":meth:`Tensor.erf`, :func:`torch.erf`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.erf_`,None
":meth:`Tensor.erfc`, :func:`torch.erfc`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.erfc_`,None
":meth:`Tensor.erfinv`, :func:`torch.erfinv`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.erfinv_`,None
":meth:`Tensor.exp`, :func:`torch.exp`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.exp_`,None
:meth:`Tensor.expand`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.expm1`, :func:`torch.expm1`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.expm1_`,None
:meth:`Tensor.exponential_`,None
:meth:`Tensor.fill_`,None
":meth:`Tensor.flatten`, :func:`torch.flatten`", راجع التوثيق
:meth:`Tensor.float`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.floor`, :func:`torch.floor`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.floor_`,None
":meth:`Tensor.frac`, :func:`torch.frac`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.frac_`,None
":meth:`Tensor.ge`, :func:`torch.ge`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
":meth:`Tensor.get_device`, :func:`torch.get_device`",None
:attr:`Tensor.grad`,None
":meth:`Tensor.gt`, :func:`torch.gt`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:meth:`Tensor.half`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.has_names`, راجع التوثيق
":meth:`Tensor.index_fill`, :func:`torch.index_fill`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.index_fill_`,None
:meth:`Tensor.int`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.is_contiguous`,None
:attr:`Tensor.is_cuda`,None
":meth:`Tensor.is_floating_point`, :func:`torch.is_floating_point`",None
:attr:`Tensor.is_leaf`,None
:meth:`Tensor.is_pinned`,None
:meth:`Tensor.is_shared`,None
":meth:`Tensor.is_signed`, :func:`torch.is_signed`",None
:attr:`Tensor.is_sparse`,None
:attr:`Tensor.is_sparse_csr`,None
:func:`torch.is_tensor`,None
:meth:`Tensor.item`,None
:attr:`Tensor.itemsize`,None
":meth:`Tensor.kthvalue`, :func:`torch.kthvalue`",:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.le`, :func:`torch.le`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
":meth:`Tensor.log`, :func:`torch.log`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.log10`, :func:`torch.log10`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.log10_`,None
":meth:`Tensor.log1p`, :func:`torch.log1p`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.log1p_`,None
":meth:`Tensor.log2`, :func:`torch.log2`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.log2_`,None
:meth:`Tensor.log_`,None
:meth:`Tensor.log_normal_`,None
":meth:`Tensor.logical_not`, :func:`torch.logical_not`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.logical_not_`,None
":meth:`Tensor.logsumexp`, :func:`torch.logsumexp`",:ref:`يحذف_الأبعاد-doc`
:meth:`Tensor.long`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.lt`, :func:`torch.lt`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:func:`torch.manual_seed`,None
":meth:`Tensor.masked_fill`, :func:`torch.masked_fill`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.masked_fill_`,None
":meth:`Tensor.masked_select`, :func:`torch.masked_select`", يوائم القناع مع الإدخال ثم يوحد_الأسماء_من_إدخالات_التنسور
":meth:`Tensor.matmul`, :func:`torch.matmul`",:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.mean`, :func:`torch.mean`",:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.median`, :func:`torch.median`",:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.nanmedian`, :func:`torch.nanmedian`",:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.mm`, :func:`torch.mm`",:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.mode`, :func:`torch.mode`",:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.mul`, :func:`torch.mul`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:meth:`Tensor.mul_`,:ref:`يوحد_الأسماء_من_الإدخالات-doc`
":meth:`Tensor.mv`, :func:`torch.mv`",:ref:`يحذف_الأبعاد-doc`
:attr:`Tensor.names`, راجع التوثيق
":meth:`Tensor.narrow`, :func:`torch.narrow`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:attr:`Tensor.nbytes`,None
:attr:`Tensor.ndim`,None
:meth:`Tensor.ndimension`,None
":meth:`Tensor.ne`, :func:`torch.ne`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
":meth:`Tensor.neg`, :func:`torch.neg`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.neg_`,None
:func:`torch.normal`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.normal_`,None
":meth:`Tensor.numel`, :func:`torch.numel`",None
:func:`torch.ones`,:ref:`مصنع-doc`
":meth:`Tensor.pow`, :func:`torch.pow`",:ref:`يوحد_الأسماء_من_الإدخالات-doc`
:meth:`Tensor.pow_`,None
":meth:`Tensor.prod`, :func:`torch.prod`",:ref:`يحذف_الأبعاد-doc`
":meth:`Tensor.rad2deg`, :func:`torch.rad2deg`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.rad2deg_`,None
:func:`torch.rand`,:ref:`مصنع-doc`
:func:`torch.rand`,:ref:`مصنع-doc`
:func:`torch.randn`,:ref:`مصنع-doc`
:func:`torch.randn`,:ref:`مصنع-doc`
:meth:`Tensor.random_`,None
":meth:`Tensor.reciprocal`, :func:`torch.reciproc.al`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.reciprocal_`,None
:meth:`Tensor.refine_names`, راجع التوثيق
:meth:`Tensor.register_hook`,None
:meth:`Tensor.register_post_accumulate_grad_hook`,None
:meth:`Tensor.rename`, راجع التوثيق
:meth:`Tensor.rename_`, راجع التوثيق
:attr:`Tensor.requires_grad`,None
:meth:`Tensor.requires_grad_`,None
:meth:`Tensor.resize_`, يسمح فقط بعمليات تغيير الحجم التي لا تغير الشكل
:meth:`Tensor.resize_as_`, يسمح فقط بعمليات تغيير الحجم التي لا تغير الشكل
":meth:`Tensor.round`, :func:`torch.round`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.round_`,None
":meth:`Tensor.rsqrt`, :func:`torch.rsqrt`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.rsqrt_`,None
":meth:`Tensor.select`, :func:`torch.select`",:ref:`يحذف_الأبعاد-doc`
:meth:`Tensor.short`,:ref:`يحتفظ_بأسماء_الإدخال-doc`
":meth:`Tensor.sigmoid`, :func:`torch.sigmoid`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.sigmoid_`,None
":meth:`Tensor.sign`, :func:`torch.sign`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.sign_`,None
":meth:`Tensor.sgn`, :func:`torch.sgn`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.sgn_`,None
":meth:`Tensor.sin`, :func:`torch.sin`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.sin_`,None
":meth:`Tensor.sinh`, :func:`torch.sinh`",:ref:`يحتفظ_بأسماء_الإدخال-doc`
:meth:`Tensor.sinh_`,None
":meth:`Tensor.asinh`, :func:`torch.as
هذا هو النص المترجم إلى اللغة العربية بتنسيق ReStructuredText:

::

    >>> x = torch.randn(3, 3, names=('N', 'C'))
    >>> x.abs().names
    ('N', 'C')

.. _removes_dimensions-doc:

يحذف الأبعاد
^^^^^^^^^^^^^^^^^^

تقوم جميع عمليات التخفيض مثل :meth:`~Tensor.sum` بحذف الأبعاد عن طريق التخفيض
على الأبعاد المرغوبة. هناك عمليات أخرى مثل :meth:`~Tensor.select` و
:meth:`~Tensor.squeeze` تقوم بحذف الأبعاد.

في أي مكان يمكن فيه تمرير مؤشر بُعد صحيح إلى مشغل، يمكن أيضًا تمرير
اسم البعد. يمكن للوظائف التي تأخذ قوائم من مؤشرات الأبعاد أن تأخذ أيضًا
قائمة من أسماء الأبعاد.

- تحقق من الأسماء: إذا تم تمرير :attr:`dim` أو :attr:`dims` كقائمة من الأسماء،
  تحقق من وجود هذه الأسماء في :attr:`self`.
- قم بإنشاء الأسماء: إذا كانت أبعاد مصفوفة الإدخال المحددة بواسطة :attr:`dim`
  أو :attr:`dims` غير موجودة في مصفوفة الإخراج، فإن الأسماء المقابلة لتلك الأبعاد
  لا تظهر في ``output.names``.

::

    >>> x = torch.randn(1, 3, 3, 3, names=('N', 'C', 'H', 'W'))
    >>> x.squeeze('N').names
    ('C', 'H', 'W')

    >>> x = torch.randn(3, 3, 3, 3, names=('N', 'C', 'H', 'W'))
    >>> x.sum(['N', 'C']).names
    ('H', 'W')

    # لا تقوم عمليات التخفيض مع keepdim=True بحذف الأبعاد فعليًا.
    >>> x = torch.randn(3, 3, 3, 3, names=('N', 'C', 'H', 'W'))
    >>> x.sum(['N', 'C'], keepdim=True).names
    ('N', 'C', 'H', 'W')


.. _unifies_names_from_inputs-doc:

توحيد الأسماء من الإدخالات
^^^^^^^^^^^^^^^^^^^^^^^^^

تتبع جميع العمليات الحسابية الثنائية هذه القاعدة. لا تزال العمليات التي يتم بثها تبث موضعيًا من اليمين للحفاظ على التوافق مع المصفوفات غير المسماة. لأداء البث الصريح بالأسماء، استخدم :meth:`Tensor.align_as`.

- تحقق من الأسماء: يجب أن تتطابق جميع الأسماء موضعيًا من اليمين. أي، في
  ``tensor + other``، يجب أن يكون ``match(tensor.names[i], other.names[i])`` صحيحًا لجميع
  ``i`` في ``(-min(tensor.dim(), other.dim()) + 1, -1]``.
- تحقق من الأسماء: علاوة على ذلك، يجب محاذاة جميع الأبعاد المسماة من اليمين.
  أثناء المطابقة، إذا قمنا بمطابقة بعد مسمى ``A`` مع بعد غير مسمى
  ``None``، فيجب ألا يظهر ``A`` في المصفوفة مع البعد غير المسمى.
- قم بإنشاء الأسماء: قم بتوحيد أزواج الأسماء من اليمين من كلا المصفوفتين لإنتاج أسماء الإخراج.

على سبيل المثال،

::

    # tensor: Tensor[   N, None]
    # other:  Tensor[None,    C]
    >>> tensor = torch.randn(3, 3, names=('N', None))
    >>> other = torch.randn(3, 3, names=(None, 'C'))
    >>> (tensor + other).names
    ('N', 'C')

تحقق من الأسماء:

- ``match(tensor.names[-1], other.names[-1])`` هو ``True``
- ``match(tensor.names[-2], tensor.names[-2])`` هو ``True``
- لأننا قمنا بمطابقة ``None`` في :attr:`tensor` مع ``'C'``،
  تحقق للتأكد من أن ``'C'`` لا يوجد في :attr:`tensor` (لا يوجد).
- تحقق للتأكد من أن ``'N'`` لا يوجد في :attr:`other` (لا يوجد).

أخيرًا، يتم حساب أسماء الإخراج باستخدام
``[unify('N', None), unify(None, 'C')] = ['N', 'C']``

أمثلة أخرى::

    # لا تتطابق الأبعاد من اليمين:
    # tensor: Tensor[N, C]
    # other:  Tensor[   N]
    >>> tensor = torch.randn(3, 3, names=('N', 'C'))
    >>> other = torch.randn(3, names=('N',))
    >>> (tensor + other).names
    RuntimeError: Error when attempting to broadcast dims ['N', 'C'] and dims
    ['N']: dim 'C' and dim 'N' are at the same position from the right but do
    not match.

    # الأبعاد غير محاذاة عند مطابقة tensor.names[-1] و other.names[-1]:
    # tensor: Tensor[N, None]
    # other:  Tensor[      N]
    >>> tensor = torch.randn(3, 3, names=('N', None))
    >>> other = torch.randn(3, names=('N',))
    >>> (tensor + other).names
    RuntimeError: Misaligned dims when attempting to broadcast dims ['N'] and
    dims ['N', None]: dim 'N' appears in a different position from the right
    across both lists.

.. note::

    في كلا المثالين الأخيرين، من الممكن محاذاة المصفوفات بالأسماء
    ثم إجراء الإضافة. استخدم :meth:`Tensor.align_as` لمحاذاة
    المصفوفات حسب الاسم أو :meth:`Tensor.align_to` لمحاذاة المصفوفات إلى ترتيب أبعاد مخصص.

.. _permutes_dimensions-doc:

يبدل الأبعاد
^^^^^^^^^^^^^^^^^^^

تقوم بعض العمليات، مثل :meth:`Tensor.t()`، بتبديل ترتيب الأبعاد. ترتبط أسماء الأبعاد بالأبعاد الفردية لذا يتم تبديلها أيضًا.

إذا كان المشغل يأخذ مؤشر موضعي :attr:`dim`، فيمكنه أيضًا أخذ اسم البعد كـ :attr:`dim`.

- تحقق من الأسماء: إذا تم تمرير :attr:`dim` كاسم، تحقق من وجوده في المصفوفة.
- قم بإنشاء الأسماء: قم بتبديل أسماء الأبعاد بنفس طريقة تبديل الأبعاد التي يتم
  تبديلها.

::

    >>> x = torch.randn(3, 3, names=('N', 'C'))
    >>> x.transpose('N', 'C').names
    ('C', 'N')

.. _contracts_away_dims-doc:

يتعاقد بعيدا الأبعاد
^^^^^^^^^^^^^^^^^^^

تتبع وظائف الضرب المصفوفي بعض المتغيرات من هذا. دعنا نمر عبر
:func:`torch.mm` أولاً ثم نقوم بتعميم القاعدة لضرب المصفوفة الدفعية.

بالنسبة لـ ``torch.mm(tensor، other)``:

- تحقق من الأسماء: لا شيء
- قم بإنشاء الأسماء: تكون أسماء النتائج هي ``(tensor.names[-2]، other.names[-1])``.

::

    >>> x = torch.randn(3, 3, names=('N', 'D'))
    >>> y = torch.randn(3, 3, names=('in', 'out'))
    >>> x.mm(y).names
    ('N', 'out')

بشكل جوهري، يؤدي ضرب المصفوفة إلى تنفيذ جداء نقطي عبر بعدين،
مما يؤدي إلى انهيارهما. عندما يتم ضرب مصفوفتين، تختفي الأبعاد المتعاقدة ولا تظهر في مصفوفة الإخراج.

:func:`torch.mv`، :func:`torch.dot` تعمل بطريقة مماثلة: لا يتحقق الاستدلال الاسمي من أسماء الإدخال ويزيل الأبعاد المشاركة في الجداء النقطي:

::

    >>> x = torch.randn(3, 3, names=('N', 'D'))
    >>> y = torch.randn(3, names=('something',))
    >>> x.mv(y).names
    ('N',)

الآن، دعنا نلقي نظرة على ``torch.matmul(tensor، other)``. افترض أن ``tensor.dim() >= 2``
و ``other.dim() >= 2``.

- تحقق من الأسماء: تحقق من أن أبعاد الدفعات للمُدخلات محاذاة وقابلة للبث.
  راجع :ref:`unifies_names_from_inputs-doc` لمعرفة ما يعنيه أن تكون المدخلات محاذاة.
- قم بإنشاء الأسماء: يتم الحصول على أسماء النتائج عن طريق توحيد أبعاد الدفعات وإزالة
  الأبعاد المتعاقدة:
  ``unify(tensor.names[:-2]، other.names[:-2]) + (tensor.names[-2]، other.names[-1])``.

أمثلة::

    # ضرب المصفوفة الدفعي للمصفوفات Tensor['C'، 'D'] و Tensor['E'، 'F'].
    # 'A'، 'B' هي أبعاد الدفعات.
    >>> x = torch.randn(3, 3, 3, 3, names=('A'، 'B'، 'C'، 'D'))
    >>> y = torch.randn(3، 3، 3، names=('B'، 'E'، 'F'))
    >>> torch.matmul(x، y).names
    ('A'، 'B'، 'C'، 'F')


أخيرًا، هناك إصدارات "إضافة" مدمجة للعديد من وظائف ضرب المصفوفة. أي: func:`addmm`
و: func:`addmv`. يتم التعامل مع هذه الإصدارات على أنها تُشكل استدلال الاسم لـ أي: func:`mm` واستدلال الاسم لـ: func:`add`.

.. _factory-doc:

وظائف المصنع
^^^^^^^^^^^^^^^^^


تأخذ وظائف المصنع الآن وسيطًا جديدًا :attr:`names` يربط اسمًا
مع كل بُعد.

::

    >>> torch.zeros(2, 3, names=('N', 'C'))
    tensor([[0.، 0.، 0.]،
            [0.، 0.، 0.]], names=('N'، 'C'))

.. _out_function_semantics-doc:

وظيفة out والإصدارات داخل المكان
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يكون للمصفوفة المحددة كمصفوفة "out=" السلوك التالي:

- إذا لم يكن لديه أبعاد مسماة، فسيتم نشر الأسماء المحسوبة من العملية
  إليه.
- إذا كان لديه أي أبعاد مسماة، فيجب أن تكون الأسماء المحسوبة من العملية
  متطابقة تمامًا مع الأسماء الموجودة. وإلا، فإن العملية تخطئ.

تقوم جميع الطرق داخل المكان بتعديل المدخلات بحيث يكون لها أسماء متساوية مع الأسماء المحسوبة
من الاستدلال الاسمي. على سبيل المثال:

::

    >>> x = torch.randn(3, 3)
    >>> y = torch.randn(3, 3, names=('N', 'C'))
    >>> x.names
    (None, None)

    >>> x += y
    >>> x.names
    ('N', 'C')