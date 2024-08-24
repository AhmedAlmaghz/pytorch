.. role:: hidden
    :class: hidden-section

حزمة الدقة المختلطة التلقائية - torch.amp
=============================

.. كلا الموديولين التاليين يفتقران إلى إدخال التوثيق. نقوم بإضافتهما هنا مؤقتًا.
.. هذا لا يضيف أي شيء إلى الصفحة المعروضة
.. py:module:: torch.cpu.amp
.. py:module:: torch.cuda.amp

.. automodule:: torch.amp
.. currentmodule:: torch.amp

:class:`torch.amp` يوفر طرقًا مريحة للتدقيق المختلط،
حيث تستخدم بعض العمليات نوع بيانات "torch.float32" (float) وتستخدم عمليات أخرى
نوع بيانات النقطة العائمة ذات الدقة المنخفضة (lower_precision_fp): "torch.float16" (half) أو "torch.bfloat16". بعض العمليات، مثل الطبقات الخطية والعمليات الترافقية،
أسرع بكثير في "lower_precision_fp". تحتاج عمليات أخرى، مثل التخفيضات، غالبًا إلى النطاق الديناميكي لـ "float32". يحاول التدقيق المختلط مطابقة كل عملية مع نوع البيانات المناسب لها.

عادةً، يستخدم "التدريب بالتدقيق المختلط التلقائي" بنوع بيانات "torch.float16" كلاً من: class:`torch.autocast` و
:class:`torch.amp.GradScaler` معًا، كما هو موضح في الأمثلة على: ref:`التدقيق المختلط التلقائي<amp-examples>`
وصفة "التدقيق المختلط التلقائي" <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html> _.
ومع ذلك، فإن: class:`torch.autocast` و: class:`torch.GradScaler` قابلة للتبديل، ويمكن استخدامها بشكل منفصل إذا رغبت في ذلك.
كما هو موضح في قسم مثال وحدة المعالجة المركزية من: class:`torch.autocast`، "التدريب/الاستدلال بالتدقيق المختلط التلقائي" على وحدة المعالجة المركزية بنوع بيانات "torch.bfloat16" يستخدم فقط: class:`torch.autocast`.

.. تحذير ::
    "torch.cuda.amp.autocast(args...)" و "torch.cpu.amp.autocast(args...)" سيتم إهمالهما. يرجى استخدام "torch.autocast("cuda"، args...)" أو "torch.autocast("cpu"، args...)" بدلاً من ذلك.
    "torch.cuda.amp.GradScaler(args...)" و "torch.cpu.amp.GradScaler(args...)" سيتم إهمالهما. يرجى استخدام "torch.GradScaler("cuda"، args...)" أو "torch.GradScaler("cpu"، args...)" بدلاً من ذلك.

:class:`torch.autocast` و: class:`torch.cpu.amp.autocast` جديدان في الإصدار 1.10.

.. محتويات:: :local:

.. _autocasting:

Autocasting
^^^^^^^^^^^
.. currentmodule:: torch.amp.autocast_mode

.. autofunction::  is_autocast_available

.. currentmodule:: torch

.. autoclass:: autocast
    :members:

.. currentmodule:: torch.amp

.. autofunction::  custom_fwd

.. autofunction::  custom_bwd

.. currentmodule:: torch.cuda.amp

.. autoclass:: autocast
    :members:

.. autofunction::  custom_fwd

.. autofunction::  custom_bwd

.. currentmodule:: torch.cpu.amp

.. autoclass:: autocast
    :members:

.. _gradient-scaling:

Gradient Scaling
^^^^^^^^^^^^^^^^

إذا كانت عملية التقديم لشبكة معينة تحتوي على إدخالات "float16"، فإن عملية التراجع لتلك
العملية ستنتج تدرجات "float16".
قد لا تكون قيم التدرجات ذات المقادير الصغيرة قابلة للتمثيل في "float16".
سيتم مسح هذه القيم إلى الصفر ("تحت التدفق")، لذا سيتم فقدان التحديث المقابل للمعلمات.

لمنع التدفق السفلي، تضرب "مقياس التدرج" خسارة (خسائر) الشبكة بعامل مقياس وتستدعي
عملية تراجع على الخسائر المُدرجة. ثم يتم ضبط التدرجات المتدفقة للخلف عبر الشبكة
بواسطة عامل المقياس نفسه. وبعبارة أخرى، فإن قيم التدرج لها حجم أكبر،
لذلك لا يتم مسحها إلى الصفر.

يجب إلغاء ضبط تدرج كل معلمة (سمة ".grad") قبل أن يقوم المحسن بتحديث المعلمات،
لذلك لا يتداخل عامل المقياس مع معدل التعلم.

.. ملاحظة ::

  قد لا تعمل AMP/fp16 لكل نموذج! على سبيل المثال، معظم النماذج المُدربة مسبقًا على bf16 لا يمكنها العمل في
  النطاق العددي fp16 بحد أقصى 65504 وستتسبب في فيض التدرجات بدلاً من التدفق السفلي. في
  هذه الحالة، قد ينخفض عامل المقياس إلى أقل من 1 لمحاولة جلب التدرجات إلى رقم
  قابل للتمثيل في النطاق الديناميكي fp16. في حين قد يتوقع المرء أن المقياس دائمًا أعلى من 1، فإن
  GradScaler لا يضمن ذلك للحفاظ على الأداء. إذا صادفت NaN في خسارتك أو تدرجاتك عند
  العمل مع AMP/fp16، تحقق من توافق نموذجك.

.. currentmodule:: torch.cuda.amp

.. autoclass:: GradScaler
    :members:

.. _autocast-op-reference:

Autocast Op Reference
^^^^^^^^^^^^^^^^^^^^^

.. _autocast-eligibility:

Op Eligibility
--------------
العمليات التي تعمل في "float64" أو أنواع البيانات غير العائمة غير مؤهلة، وسوف
تشغيل هذه الأنواع سواء تم تمكين التدقيق التلقائي أم لا.

العمليات غير الموضعية وطرق Tensor فقط مؤهلة.
يُسمح بالمتغيرات الموضعية والمكالمات التي توفر صراحةً "out=..." Tensor
في المناطق الممكنة للتدقيق التلقائي، ولكنها لن تمر عبر التدقيق التلقائي.
على سبيل المثال، في منطقة ممكنة للتدقيق التلقائي، يمكن أن يكون "a.addmm(b، c)" تدقيقًا تلقائيًا،
لكن "a.addmm_(b، c)" و "a.addmm(b، c، out=d)" لا يمكنهما ذلك.
من أجل الأداء والاستقرار الأفضل، يُفضل العمليات غير الموضعية في المناطق الممكّنة للتدقيق التلقائي.

العمليات التي يتم استدعاؤها باستخدام وسيط "dtype=..." صريح غير مؤهلة،
وسينتج عنها إخراج يحترم وسيط "dtype".

.. _autocast-cuda-op-reference:

CUDA Op-Specific Behavior
-------------------------
تصف القوائم التالية سلوك العمليات المؤهلة في المناطق الممكّنة للتدقيق التلقائي.
هذه العمليات تمر دائمًا عبر التدقيق التلقائي سواء تم استدعاؤها كجزء من: class:`torch.nn.Module`،
كدالة، أو كطريقة: class:`torch.Tensor`. إذا تم عرض الوظائف في مساحات أسماء متعددة،
فإنها تمر عبر التدقيق التلقائي بغض النظر عن مساحة الاسم.

العمليات غير المدرجة أدناه لا تمر عبر التدقيق التلقائي. إنهم يعملون في النوع
تحدده إدخالاتها. ومع ذلك، قد يغير التدقيق التلقائي النوع
الذي تعمل فيه العمليات غير المدرجة إذا كانت أسفل العمليات الممكّنة للتدقيق التلقائي.

إذا كانت العملية غير مدرجة، فإننا نفترض أنها مستقرة عدديًا في "float16".
إذا كنت تعتقد أن عملية غير مدرجة غير مستقرة عدديًا في "float16"،
يرجى تقديم مشكلة.

عمليات CUDA التي يمكنها التدقيق التلقائي إلى "float16"
""""""""""""""""""""""""""""""""""""""

``__matmul__``،
``addbmm``،
``addmm``،
``addmv``،
``addr``،
``baddbmm``،
``bmm``،
``chain_matmul``،
``multi_dot``،
``conv1d``،
``conv2d``،
``conv3d``،
``conv_transpose1d``،
``conv_transpose2d``،
``conv_transpose3d``،
``GRUCell``،
``linear``،
``LSTMCell``،
``matmul``،
``mm``،
``mv``،
``prelu``،
``RNNCell``

عمليات CUDA التي يمكنها التدقيق التلقائي إلى "float32"
""""""""""""""""""""""""""""""""""""""

``__pow__``،
``__rdiv__``،
``__rpow__``،
``__rtruediv__``،
``acos``،
``asin``،
``binary_cross_entropy_with_logits``،
``cosh``،
``cosine_embedding_loss``،
``cdist``،
``cosine_similarity``،
``cross_entropy``،
``cumprod``،
``cumsum``،
``dist``،
``erfinv``،
``exp``،
``expm1``،
``group_norm``،
``hinge_embedding_loss``،
``kl_div``،
``l1_loss``،
``layer_norm``،
``log``،
``log_softmax``،
``log10``،
``log1p``،
``log2``،
``margin_ranking_loss``،
``mse_loss``،
``multilabel_margin_loss``،
``multi_margin_loss``،
``nll_loss``،
``norm``،
``normalize``،
``pdist``،
``poisson_nll_loss``،
``pow``،
``prod``،
``reciprocal``،
``rsqrt``،
``sinh``،
``smooth_l1_loss``،
``soft_margin_loss``،
``softmax``،
``softmin``،
``softplus``،
``sum``،
``renorm``،
``tan``،
``triplet_margin_loss``

عمليات CUDA التي تنتقل إلى أوسع نوع إدخال
"""""""""""""""""""""""""""""
هذه العمليات لا تتطلب نوع بيانات معين للاستقرار، ولكنها تأخذ إدخالات متعددة
وتتطلب أن تتطابق أنواع بيانات الإدخال. إذا كانت جميع الإدخالات
"float16"، تعمل العملية في "float16". إذا كان أي من الإدخالات "float32"،
يقوم التدقيق التلقائي بتحويل جميع الإدخالات إلى "float32" وتشغيل العملية في "float32".

``addcdiv``،
``addcmul``،
``atan2``،
``bilinear``،
``cross``،
``dot``،
``grid_sample``،
``index_put``،
``scatter_add``،
``tensordot``

بعض العمليات غير المدرجة هنا (على سبيل المثال، العمليات الثنائية مثل "add") تقوم أصلاً بترقية
الإدخالات دون تدخل التدقيق التلقائي. إذا كانت الإدخالات مزيجًا من "float16"
و "float32"، تعمل هذه العمليات في "float32" وتنتج إخراج "float32"،
بغض النظر عما إذا كان التدقيق التلقائي ممكّنًا أم لا.

يفضل "binary_cross_entropy_with_logits" على "binary_cross_entropy"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
يمكن لعمليات التراجع الخلفي لـ: func:`torch.nn.functional.binary_cross_entropy` (و: mod:`torch.nn.BCELoss`، الذي يغلّفه)
إنتاج تدرجات غير قابلة للتمثيل في "float16". في المناطق الممكّنة للتدقيق التلقائي، قد يكون الإدخال الأمامي
"float16"، مما يعني أن تدرج التراجع الخلفي يجب أن يكون قابلًا للتمثيل في "float16" (تحويل "float16" الأمامي
إلى "float32" لا يساعد، لأنه يجب عكس هذا التحويل في الخلف).
لذلك، تثير "binary_cross_entropy" و "BCELoss" خطأ في المناطق الممكّنة للتدقيق التلقائي.

تستخدم العديد من النماذج طبقة سيجمويد مباشرة قبل طبقة الخسارة الثنائية.
في هذه الحالة، قم بدمج الطبقتين باستخدام: func:`torch.nn.functional.binary_cross_entropy_with_logits`
أو: mod:`torch.nn.BCEWithLogitsLoss`. "binary_cross_entropy_with_logits" و "BCEWithLogits"
آمنة للتدقيق التلقائي.

.. _autocast-xpu-op-reference:

سلوك XPU المحدد للعملية (تجريبي)
-----------------------
تصف القوائم التالية سلوك العمليات المؤهلة في المناطق الممكّنة للتدقيق التلقائي.
هذه العمليات تمر دائمًا عبر التدقيق التلقائي سواء تم استدعاؤها كجزء من: class:`torch.nn.Module`،
كدالة، أو كطريقة: class:`torch.Tensor`. إذا تم عرض الوظائف في مساحات أسماء متعددة،
فإنها تمر عبر التدقيق التلقائي بغض النظر عن مساحة الاسم.

العمليات غير المدرجة أدناه لا تمر عبر التدقيق التلقائي. إنهم يعملون في النوع
تحدده إدخالاتها. ومع ذلك، قد يغير التدقيق التلقائي النوع
الذي تعمل فيه العمليات غير المدرجة إذا كانت أسفل العمليات الممكّنة للتدقيق التلقائي.

إذا كانت العملية غير مدرجة، فإننا نفترض أنها مستقرة عدديًا في "float16".
إذا كنت تعتقد أن عملية غير مدرجة غير مستقرة عدديًا في "float16"،
يرجى تقديم مشكلة.

عمليات XPU التي يمكنها التدقيق التلقائي إلى "float16"
"""""""""""""""""""""""""""""""""""""

``addbmm``،
``addmm``،
``addmv``،
``addr``،
``baddbmm``،
``bmm``،
``chain_matmul``،
``multi_dot``،
``conv1d``،
``conv2d``،
``conv3d``،
``conv_transpose1d``،
``conv_transpose2d``،
``conv_transpose3d``،
``GRUCell``،
``linear``،
``LSTMCell``،
``matmul``،
``mm``،
``mv``،
``RNNCell``

عمليات XPU التي يمكنها التدقيق التلقائي إلى "float32"
"""""""""""""""""""""""""""""""""""""

``__pow__``،
``__rdiv__``،
``__rpow__``،
``__rtruediv__``،
``binary_cross_entropy_with_logits``،
``cosine_embedding_loss``،
``cosine_similarity``،
``cumsum``،
``dist``،
``exp``،
``group_norm``،
``hinge_embedding_loss``،
``kl_div``،
``l1_loss``،
``layer_norm``،
``log``،
``log_softmax``،
``margin_ranking_loss``،
``nll_loss``،
``normalize``،
``poisson_nll_loss``،
``pow``،
``reciprocal``،
``rsqrt``،
``soft_margin_loss``،
``softmax``،
``softmin``،
``sum``،
``triplet_margin_loss``

عمليات XPU التي تنتقل إلى أوسع نوع إدخال
""""""""""""""""""""""""""""
هذه العمليات لا تتطلب نوع بيانات معين للاستقرار، ولكنها تأخذ إدخالات متعددة
وتتطلب أن تتطابق أنواع بيانات الإدخال. إذا كانت جميع الإدخالات
"float16"، تعمل العملية في "float16". إذا كان أي من الإدخالات "float32"،
يقوم التدقيق التلقائي بتحويل جميع الإدخالات إلى "float32" وتشغيل العملية في "float32".

``bilinear``،
``cross``،
``grid_sample``،
``index_put``،
``scatter_add``،
``tensordot``

بعض العمليات غير المدرجة هنا (على سبيل المثال، العمليات الثنائية مثل "add") تقوم أصلاً بترقية

الإدخالات دون تدخل
-------------
تصف القوائم التالية سلوك العمليات المؤهلة في المناطق التي تدعم التحويل التلقائي للنوع (autocast-enabled regions).
هذه العمليات تخضع دائمًا للتحويل التلقائي للنوع بغض النظر عما إذا كانت مستدعاة كجزء من :class:`torch.nn.Module`،
أو كدالة، أو كطريقة لـ :class:`torch.Tensor`. إذا كانت الدوال متاحة في مساحات أسماء متعددة،
فإنها تخضع للتحويل التلقائي للنوع بغض النظر عن مساحة الأسماء.

العمليات غير المدرجة أدناه لا تخضع للتحويل التلقائي للنوع. فهي تعمل وفق النوع
الذي تحدده مدخلاتها. ومع ذلك، قد يؤدي التحويل التلقائي للنوع إلى تغيير النوع
الذي تعمل فيه العمليات غير المدرجة إذا كانت تقع في اتجاه تدفق البيانات بعد العمليات التي تخضع للتحويل التلقائي للنوع.

إذا كانت عملية ما غير مدرجة، فإننا نفترض أنها مستقرة من الناحية العددية في ``bfloat16``.
إذا كنت تعتقد أن هناك عملية غير مدرجة غير مستقرة من الناحية العددية في ``bfloat16``،
يرجى إرسال تقرير عن المشكلة.

عمليات وحدة المعالجة المركزية (CPU Ops) التي يمكن تحويلها تلقائيًا إلى ``bfloat16``
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

``conv1d``،
``conv2d``،
``conv3d``،
``bmm``،
``mm``،
``baddbmm``،
``addmm``،
``addbmm``،
``linear``،
``matmul``،
``_convolution``

عمليات وحدة المعالجة المركزية (CPU Ops) التي يمكن تحويلها تلقائيًا إلى ``float32``
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

``conv_transpose1d``،
``conv_transpose2d``،
``conv_transpose3d``،
``avg_pool3d``،
``binary_cross_entropy``،
``grid_sampler``،
``grid_sampler_2d``،
``_grid_sampler_2d_cpu_fallback``،
``grid_sampler_3d``،
``polar``،
``prod``،
``quantile``،
``nanquantile``،
``stft``،
``cdist``،
``trace``،
``view_as_complex``،
``cholesky``،
``cholesky_inverse``،
``cholesky_solve``،
``inverse``،
``lu_solve``،
``orgqr``،
``inverse``،
``ormqr``،
``pinverse``،
``max_pool3d``،
``max_unpool2d``،
``max_unpool3d``،
``adaptive_avg_pool3d``،
``reflection_pad1d``،
``reflection_pad2d``،
``replication_pad1d``،
``replication_pad2d``،
``replication_pad3d``،
``mse_loss``،
``ctc_loss``،
``kl_div``،
``multilabel_margin_loss``،
``fft_fft``،
``fft_ifft``،
``fft_fft2``،
``fft_ifft2``،
``fft_fftn``،
``fft_ifftn``،
``fft_rfft``،
``fft_irfft``،
``fft_rfft2``،
``fft_irfft2``،
``fft_rfftn``،
``fft_irfftn``،
``fft_hfft``،
``fft_ihfft``،
``linalg_matrix_norm``،
``linalg_cond``،
``linalg_matrix_rank``،
``linalg_solve``،
``linalg_cholesky``،
``linalg_svdvals``،
``linalg_eigvals``،
``linalg_eigvalsh``،
``linalg_inv``،
``linalg_householder_product``،
``linalg_tensorinv``،
``linalg_tensorsolve``،
``fake_quantize_per_tensor_affine``،
``eig``،
``geqrf``،
``lstsq``،
``_lu_with_info``،
``qr``،
``solve``،
``svd``،
``symeig``،
``triangular_solve``،
``fractional_max_pool2d``،
``fractional_max_pool3d``،
``adaptive_max_pool3d``،
``multilabel_margin_loss_forward``،
``linalg_qr``،
``linalg_cholesky_ex``،
``linalg_svd``،
``linalg_eig``،
``linalg_eigh``،
``linalg_lstsq``،
``linalg_inv_ex``

عمليات وحدة المعالجة المركزية (CPU Ops) التي ترفع إلى النوع الأوسع للمدخلات (widest input type)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
هذه العمليات لا تتطلب نوع بيانات (dtype) محدد للاستقرار، ولكنها تأخذ عدة مدخلات
وتتطلب مطابقة أنواع بيانات المدخلات. إذا كانت جميع المدخلات من النوع
``bfloat16``، فإن العملية تعمل في ``bfloat16``. إذا كان أي من المدخلات من النوع ``float32``،
يقوم التحويل التلقائي للنوع بتحويل جميع المدخلات إلى ``float32`` وتشغيل العملية في ``float32``.

``cat``،
``stack``،
``index_copy``

بعض العمليات غير المدرجة هنا (مثل العمليات الثنائية مثل ``add``) تقوم بشكل أصلي برفع
المدخلات بدون تدخل التحويل التلقائي للنوع. إذا كانت المدخلات مزيجًا من ``bfloat16``
و ``float32``، فإن هذه العمليات تعمل في ``float32`` وتنتج مخرجات من النوع ``float32``،
بغض النظر عما إذا كان التحويل التلقائي للنوع مفعلا أم لا.


.. تحتاج هذه الوحدة إلى توثيق. نضيفها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.amp.autocast_mode
.. py:module:: torch.cpu.amp.autocast_mode
.. py:module:: torch.cuda.amp.autocast_mode
.. py:module:: torch.cuda.amp.common
.. py:module:: torch.amp.grad_scaler
.. py:module:: torch.cpu.amp.grad_scaler
.. py:module:: torch.cuda.amp.grad_scaler