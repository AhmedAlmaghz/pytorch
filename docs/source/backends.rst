.. role:: hidden
    :class: hidden-section

torch.backends
==============
.. automodule:: torch.backends

تحكم ``torch.backends`` في سلوك العديد من الواجهات الخلفية التي تدعمها PyTorch.

تشمل هذه الواجهات الخلفية ما يلي:

- ``torch.backends.cpu``
- ``torch.backends.cuda``
- ``torch.backends.cudnn``
- ``torch.backends.cusparselt``
- ``torch.backends.mha``
- ``torch.backends.mps``
- ``torch.backends.mkl``
- ``torch.backends.mkldnn``
- ``torch.backends.nnpack``
- ``torch.backends.openmp``
- ``torch.backends.opt_einsum``
- ``torch.backends.xeon``

torch.backends.cpu
^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cpu

.. autofunction::  torch.backends.cpu.get_cpu_capability

torch.backends.cuda
^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cuda

.. autofunction::  torch.backends.cuda.is_built

.. currentmodule:: torch.backends.cuda.matmul
.. attribute::  allow_tf32

    قيمة منطقية ``bool`` تتحكم فيما إذا كان يمكن استخدام TensorFloat-32 tensor cores في عمليات الضرب
    المصفوفي على معالجات Ampere الرسومية أو الأحدث. راجع: :ref:`tf32_on_ampere`.

.. attribute::  allow_fp16_reduced_precision_reduction

    قيمة منطقية ``bool`` تتحكم فيما إذا كان مسموحًا بالتخفيضات ذات الدقة المنخفضة (على سبيل المثال، مع نوع
    التراكم fp16) مع عمليات الضرب المصفوفي fp16.

.. attribute::  allow_bf16_reduced_precision_reduction

    قيمة منطقية ``bool`` تتحكم فيما إذا كان مسموحًا بالتخفيضات ذات الدقة المنخفضة مع عمليات الضرب المصفوفي
    bf16.

.. currentmodule:: torch.backends.cuda
.. attribute::  cufft_plan_cache

    تحتوي ``cufft_plan_cache`` على ذاكرة التخزين المؤقت لخطة cuFFT لكل جهاز CUDA. يمكنك الاستعلام عن ذاكرة
    التخزين المؤقت لجهاز محدد 'i' عبر ``torch.backends.cuda.cufft_plan_cache[i]``.

    .. currentmodule:: torch.backends.cuda.cufft_plan_cache
    .. attribute::  size

        عدد صحيح ``int`` للقراءة فقط يظهر عدد الخطط حاليًا في ذاكرة التخزين المؤقت لـ cuFFT.

    .. attribute::  max_size

        عدد صحيح ``int`` يتحكم في سعة ذاكرة التخزين المؤقت لـ cuFFT.

    .. method::  clear()

        مسح ذاكرة التخزين المؤقت لـ cuFFT.

.. autofunction:: torch.backends.cuda.preferred_blas_library

.. autofunction:: torch.backends.cuda.preferred_linalg_library

.. autoclass:: torch.backends.cuda.SDPAParams

.. autofunction:: torch.backends.cuda.flash_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_mem_efficient_sdp

.. autofunction:: torch.backends.cuda.mem_efficient_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_flash_sdp

.. autofunction:: torch.backends.cuda.math_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_math_sdp

.. autofunction:: torch.backends.cuda.cudnn_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_cudnn_sdp

.. autofunction:: torch.backends.cuda.is_flash_attention_available

.. autofunction:: torch.backends.cuda.can_use_flash_attention

.. autofunction:: torch.backends.cuda.can_use_efficient_attention

.. autofunction:: torch.backends.cuda.can_use_cudnn_attention

.. autofunction:: torch.backends.cuda.sdp_kernel

torch.backends.cudnn
^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cudnn

.. autofunction:: torch.backends.cudnn.version

.. autofunction:: torch.backends.cudnn.is_available

.. attribute::  enabled

    قيمة منطقية ``bool`` تتحكم فيما إذا كان cuDNN ممكّنًا.

.. attribute::  allow_tf32

    قيمة منطقية ``bool`` تتحكم فيما إذا كان يمكن استخدام TensorFloat-32 tensor cores في عمليات الضرب
    المصفوفي لـ cuDNN على معالجات Ampere الرسومية أو الأحدث. راجع: :ref:`tf32_on_ampere`.

.. attribute::  deterministic

    قيمة منطقية ``bool``، إذا كانت ``True``، ستتسبب في استخدام cuDNN لخوارزميات الضرب المصفوفي
    المحددة فقط. راجع أيضًا: ``torch.are_deterministic_algorithms_enabled`` و
    ``torch.use_deterministic_algorithms``.

.. attribute::  benchmark

    قيمة منطقية ``bool``، إذا كانت ``True``، ستتسبب في قيام cuDNN باختبار خوارزميات الضرب المصفوفي
    المتعددة واختيار الأسرع.

.. attribute::  benchmark_limit

    عدد صحيح ``int`` يحدد الحد الأقصى لعدد خوارزميات الضرب المصفوفي لـ cuDNN لمحاولة عندما يكون
    ``torch.backends.cudnn.benchmark`` هو ``True``. قم بتعيين ``benchmark_limit`` إلى الصفر لتجربة
    كل خوارزمية متاحة. لاحظ أن هذا الإعداد يؤثر فقط على عمليات الضرب المصفوفي التي يتم إرسالها
    عبر واجهة برمجة التطبيقات cuDNN v8.

.. py:module:: torch.backends.cudnn.rnn

torch.backends.cusparselt
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cusparselt

.. autofunction:: torch.backends.cusparselt.version

.. autofunction:: torch.backends.cusparselt.is_available

torch.backends.mha
^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.mha

.. autofunction::  torch.backends.mha.get_fastpath_enabled
.. autofunction::  torch.backends.mha.set_fastpath_enabled


torch.backends.mps
^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.mps

.. autofunction::  torch.backends.mps.is_available

.. autofunction::  torch.backends.mps.is_built


torch.backends.mkl
^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.mkl

.. autofunction::  torch.backends.mkl.is_available

.. autoclass::  torch.backends.mkl.verbose


torch.backends.mkldnn
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.mkldnn

.. autofunction::  torch.backends.mkldnn.is_available

.. autoclass::  torch.backends.mkldnn.verbose

torch.backends.nnpack
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.nnpack

.. autofunction::  torch.backends.nnpack.is_available

.. autofunction::  torch.backends.nnpack.flags

.. autofunction::  torch.backends.nnpack.set_flags

torch.backends.openmp
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.openmp

.. autofunction::  torch.backends.openmp.is_available

.. يجب إضافة وثائق الواجهات الخلفية الأخرى هنا.
.. يتم تضمين الوحدات النمطية التلقائية فقط للتأكد من تشغيل الفحوصات ولكنها لا تضيف
.. أي شيء إلى الصفحة المقدمة حاليًا.
.. py:module:: torch.backends.quantized
.. py:module:: torch.backends.xnnpack


torch.backends.opt_einsum
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.opt_einsum

.. autofunction:: torch.backends.opt_einsum.is_available

.. autofunction:: torch.backends.opt_einsum.get_opt_einsum

.. attribute::  enabled

    قيمة منطقية ``bool`` تتحكم فيما إذا كان opt_einsum ممكّنًا (``True`` بشكل افتراضي). إذا كان
    الأمر كذلك، فسيستخدم ``torch.einsum`` opt_einsum
    (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html) إذا كان متاحًا لحساب
    مسار تعاقدي أمثل للأداء الأسرع.

    إذا لم يكن opt_einsum متاحًا، فسيستخدم ``torch.einsum`` مسار التعاقد الافتراضي من اليسار إلى
    اليمين.

.. attribute::  strategy

    سلسلة ``str`` تحدد الاستراتيجيات التي يجب تجربتها عندما يكون
    ``torch.backends.opt_einsum.enabled`` هو ``True``. بشكل افتراضي، سيحاول ``torch.einsum``
    استراتيجية "auto"، ولكن يتم أيضًا دعم استراتيجيتي "greedy" و "optimal". لاحظ أن الاستراتيجية
    "optimal" تكون فئوية على عدد الإدخالات حيث أنها تجرب جميع المسارات الممكنة. راجع المزيد من
    التفاصيل في وثائق opt_einsum
    (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).


torch.backends.xeon
^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.xeon
.. py:module:: torch.backends.xeon.run_cpu