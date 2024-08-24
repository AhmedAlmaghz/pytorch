.. meta::
   :description: دليل حول torch.cuda، وهو وحدة في PyTorch لتنفيذ عمليات CUDA
   :keywords: إدارة الذاكرة، PYTORCH_CUDA_ALLOC_CONF، تحسين PyTorch، CUDA

.. _cuda-semantics:

دلالات CUDA
==============

تُستخدم وحدة :mod:`torch.cuda` لإعداد وتشغيل العمليات المتعلقة بـ CUDA. حيث تقوم بتتبع وحدة GPU المحددة حالياً، وجميع مصفوفات CUDA التي تقوم بتخصيصها يتم إنشاؤها افتراضياً على تلك الوحدة. ويمكن تغيير الوحدة المحددة باستخدام مدير السياق :any:`torch.cuda.device`.

ومع ذلك، بمجرد تخصيص مصفوفة، يمكنك إجراء العمليات عليها بغض النظر عن الوحدة المحددة، وستكون النتائج دائماً على نفس الوحدة التي توجد عليها المصفوفة.

العمليات عبر وحدات GPU غير مسموح بها افتراضياً، باستثناء :meth:`~torch.Tensor.copy_` والطرق الأخرى ذات الوظائف المشابهة للنسخ مثل :meth:`~torch.Tensor.to` و :meth:`~torch.Tensor.cuda`. ما لم تقم بتمكين الوصول المباشر إلى الذاكرة بين الأقران، فإن أي محاولات لتشغيل العمليات على المصفوفات المنتشرة عبر وحدات مختلفة ستؤدي إلى حدوث خطأ.

فيما يلي مثال صغير يوضح ذلك::

    cuda = torch.device('cuda')     # وحدة CUDA الافتراضية
    cuda0 = torch.device('cuda:0')
    cuda2 = torch.device('cuda:2')  # وحدة المعالجة الرسومية 2 (ذات ترقيم من نوع صفر)

    x = torch.tensor([1., 2.], device=cuda0)
    # x.device هي device(type='cuda', index=0)
    y = torch.tensor([1., 2.]).cuda()
    # y.device هي device(type='cuda', index=0)

    مع torch.cuda.device(1):
        # يقوم بتخصيص مصفوفة على وحدة GPU 1
        a = torch.tensor([1., 2.], device=cuda)

        # ينقل مصفوفة من وحدة المعالجة المركزية إلى وحدة GPU 1
        b = torch.tensor([1., 2.]).cuda()
        # a.device و b.device هما device(type='cuda', index=1)

        # يمكنك أيضاً استخدام ``Tensor.to`` لنقل مصفوفة:
        b2 = torch.tensor([1., 2.]).to(device=cuda)
        # b.device و b2.device هما device(type='cuda', index=1)

        c = a + b
        # c.device هي device(type='cuda', index=1)

        z = x + y
        # z.device هي device(type='cuda', index=0)

        # حتى ضمن سياق، يمكنك تحديد الوحدة
        # (أو إعطاء مؤشر وحدة GPU إلى مكالمة .cuda)
        d = torch.randn(2, device=cuda2)
        e = torch.randn(2).to(cuda2)
        f = torch.randn(2).cuda(cuda2)
        # d.device و e.device و f.device جميعها هي device(type='cuda', index=2)

.. _tf32_on_ampere:

TensorFloat-32 (TF32) على أجهزة Ampere (والأجهزة الأحدث)
---------------------------------------------------

بدءاً من PyTorch 1.7، هناك علم جديد يسمى `allow_tf32`. هذا العلم
افتراضي إلى True في PyTorch 1.7 إلى PyTorch 1.11، وإلى False في PyTorch 1.12 والإصدارات الأحدث.
يتحكم هذا العلم فيما إذا كان مسموحاً لـ PyTorch باستخدام TensorFloat32 (TF32) tensor cores،
المتوفرة على وحدات معالجة الرسوميات NVIDIA منذ Ampere، داخلياً لحساب عمليات الضرب في المصفوفة (عمليات الضرب في المصفوفة
وضرب المصفوفة المعززة) والعمليات التحويلية.

تم تصميم TensorFloat32 tensor cores لتحقيق أداء أفضل في عمليات الضرب في المصفوفة والعمليات التحويلية على
مصفوفات `torch.float32` عن طريق تقريب بيانات الإدخال لتكون 10 بتات من الفاصلة العائمة، وتراكم
النتائج بدقة FP32، مع الحفاظ على نطاق FP32 الديناميكي.

يتم التحكم في عمليات الضرب في المصفوفة والعمليات التحويلية بشكل منفصل، ويمكن الوصول إلى الأعلام المقابلة لها على النحو التالي:

.. code:: python

  # يتحكم العلم أدناه فيما إذا كان مسموحاً باستخدام TF32 في عمليات الضرب في المصفوفة. هذا العلم افتراضي إلى False
  # في PyTorch 1.12 والإصدارات الأحدث.
  torch.backends.cuda.matmul.allow_tf32 = True

  # يتحكم العلم أدناه فيما إذا كان مسموحاً باستخدام TF32 في cuDNN. هذا العلم افتراضي إلى True.
  torch.backends.cudnn.allow_tf32 = True

يمكن أيضاً تعيين دقة عمليات الضرب في المصفوفة بشكل أكثر عمومية (ليس فقط على CUDA) عبر :meth:`~torch.set_float_32_matmul_precision`.
لاحظ أنه بالإضافة إلى عمليات الضرب في المصفوفة والعمليات التحويلية نفسها، فإن الدوال ووحدات nn التي تستخدم داخلياً
عمليات الضرب في المصفوفة أو العمليات التحويلية تتأثر أيضاً. وتشمل هذه `nn.Linear`، `nn.Conv*`، cdist، tensordot،
affine grid و grid sample، adaptive log softmax، GRU و LSTM.

للحصول على فكرة عن الدقة والسرعة، راجع كود المثال وبيانات المعيار المرجعي (على A100) أدناه:

.. code:: python

  a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
  b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
  ab_full = a_full @ b_full
  mean = ab_full.abs().mean()  # 80.7277

  a = a_full.float()
  b = b_full.float()

  # قم بعملية ضرب المصفوفة في وضع TF32.
  torch.backends.cuda.matmul.allow_tf32 = True
  ab_tf32 = a @ b  # يستغرق 0.016 ثانية على GA100
  error = (ab_tf32 - ab_full).abs().max()  # 0.1747
  relative_error = error / mean  # 0.0022

  # قم بعملية ضرب المصفوفة مع تعطيل TF32.
  torch.backends.cuda.matmul.allow_tf32 = False
  ab_fp32 = a @ b  # يستغرق 0.11 ثانية على GA100
  error = (ab_fp32 - ab_full).abs().max()  # 0.0031
  relative_error = error / mean  # 0.000039

من المثال أعلاه، يمكننا أن نرى أنه مع تمكين TF32، تكون السرعة أسرع بحوالي 7 مرات على A100، وأن
الخطأ النسبي مقارنة بالدقة المزدوجة أكبر بحوالي مرتبتين من حيث الحجم. لاحظ أن
النسبة الدقيقة لـ TF32 إلى سرعة الدقة الفردية تعتمد على جيل الأجهزة، حيث قد تختلف الخصائص
مثل نسبة عرض النطاق الترددي للذاكرة إلى الحوسبة، وكذلك نسبة الإنتاجية TF32 إلى FP32 في عمليات الضرب في المصفوفة
قد تختلف من جيل إلى جيل أو من نموذج إلى نموذج.
إذا كانت الدقة FP32 الكاملة مطلوبة، يمكن للمستخدمين تعطيل TF32 عن طريق:

.. code:: python

  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False

للتبديل بين أعلام TF32 في C++، يمكنك القيام بما يلي:

.. code:: C++

  at::globalContext().setAllowTF32CuBLAS(false)؛
  at::globalContext().setAllowTF32CuDNN(false)؛

للحصول على مزيد من المعلومات حول TF32، راجع:

- `TensorFloat-32`_
- `CUDA 11`_
- `Ampere architecture`_

.. _TensorFloat-32: https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/
.. _CUDA 11: https://devblogs.nvidia.com/cuda-11-features-revealed/
.. _Ampere architecture: https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/

.. _fp16reducedprecision:

خفض الدقة في عمليات الضرب في المصفوفة FP16
-----------------------------------------

قد يتم إجراء عمليات الضرب في المصفوفة FP16 ببعض التخفيضات في الدقة الوسيطة (على سبيل المثال، في FP16 بدلاً من FP32). يمكن أن تسمح هذه التخفيضات الانتقائية في الدقة بتحقيق أداء أعلى في بعض أعباء العمل (خاصة تلك التي تحتوي على بعد "k" كبير) وبنى وحدات معالجة الرسوميات بتكلفة الدقة العددية واحتمال حدوث فيض.

فيما يلي بعض بيانات المعيار المرجعي على V100:

.. code::

  [--------------------------- bench_gemm_transformer --------------------------]
        [  m ,  k  ,  n  ]    |  allow_fp16_reduc=True  |  allow_fp16_reduc=False
  1 threads: --------------------------------------------------------------------
        [4096, 4048, 4096]    |           1634.6        |           1639.8
        [4096, 4056, 4096]    |           1670.8        |           1661.9
        [4096, 4080, 4096]    |           1664.2        |           1658.3
        [4096, 4096, 4096]    |           1639.4        |           1651.0
        [4096, 4104, 4096]    |           1677.4        |           1674.9
        [4096, 4128, 4096]    |           1655.7        |           1646.0
        [4096, 4144, 4096]    |           1796.8        |           2519.6
        [4096, 5096, 4096]    |           2094.6        |           3190.0
        [4096, 5104, 4096]    |           2144.0        |           2663.5
        [4096, 5112, 4096]    |           2149.1        |           2766.9
        [4096, 5120, 4096]    |           2142.8        |           2631.0
        [4096, 9728, 4096]    |           3875.1        |           5779.8
        [4096, 16384, 4096]   |           6182.9        |           9656.5
  (الأوقات بالميكروثانية).

إذا كانت التخفيضات في الدقة الكاملة مطلوبة، يمكن للمستخدمين تعطيل التخفيضات في الدقة الوسيطة في عمليات الضرب في المصفوفة FP16 مع:

.. code:: python

  torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

للتبديل بين أعلام التخفيض في الدقة في C++، يمكنك القيام بما يلي:

.. code:: C++

  at::globalContext().setAllowFP16ReductionCuBLAS(false)؛

.. _bf16reducedprecision:

خفض الدقة في عمليات الضرب في المصفوفة BF16
-------------------------------------

يوجد علم مماثل (كما هو موضح أعلاه) لعمليات الضرب في المصفوفة BFloat16.
لاحظ أن هذا المفتاح مضبوط على `True` افتراضياً لـ BF16، إذا لاحظت
عدم استقرار عددي في عبء العمل الخاص بك، فقد ترغب في تعيينه على `False`.

إذا لم تكن التخفيضات في الدقة المرغوبة، يمكن للمستخدمين تعطيل التخفيضات
في الدقة في عمليات الضرب في المصفوفة bf16 مع:

.. code:: python

  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

للتبديل بين أعلام التخفيض في الدقة في C++، يمكنك القيام بما يلي:

.. code:: C++

  at::globalContext().setAllowBF16ReductionCuBLAS(true)؛

التنفيذ غير المتزامن
----------------------

بشكل افتراضي، تكون عمليات GPU غير متزامنة. فعندما تستدعي دالة تستخدم وحدة معالجة الرسوميات (GPU)، يتم *وضع* العمليات في قائمة الانتظار على الجهاز المحدد، ولكنها لا تنفذ بالضرورة حتى وقت لاحق. يسمح لنا ذلك بتنفيذ المزيد من الحسابات بشكل متوازي، بما في ذلك العمليات على وحدة المعالجة المركزية (CPU) أو وحدات معالجة الرسوميات الأخرى.

وبشكل عام، يكون تأثير الحساب غير المتزامن غير مرئي للمستدعي، وذلك لأن (1) كل جهاز ينفذ العمليات بالترتيب الذي يتم وضعها في قائمة الانتظار، و (2) PyTorch يقوم تلقائيًا بمزامنة ضرورية عند نسخ البيانات بين وحدة المعالجة المركزية ووحدة معالجة الرسوميات أو بين وحدتي معالجة رسوميات. وبالتالي، ستستمر عملية الحساب كما لو أن كل عملية تم تنفيذها بشكل متزامن.

يمكنك فرض الحساب المتزامن عن طريق تعيين متغير البيئة ``CUDA_LAUNCH_BLOCKING=1``. قد يكون هذا مفيدًا عندما يحدث خطأ في وحدة معالجة الرسوميات. (مع التنفيذ غير المتزامر، لا يتم الإبلاغ عن مثل هذا الخطأ حتى يتم تنفيذ العملية بالفعل، لذا لا يظهر تتبع المكدس المكان الذي تم طلبها منه.)

ومن نتائج الحساب غير المتزامن أن قياسات الوقت دون مزامنات غير دقيقة. للحصول على قياسات دقيقة، يجب على المرء إما استدعاء :func:`torch.cuda.synchronize()` قبل القياس، أو استخدام :class:`torch.cuda.Event` لتسجيل الأوقات كما يلي::

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # قم بتشغيل بعض الأشياء هنا

    end_event.record()
    torch.cuda.synchronize()  # انتظر حتى يتم تسجيل الأحداث!
    elapsed_time_ms = start_event.elapsed_time(end_event)

وكاستثناء، هناك العديد من الدوال مثل :meth:`~torch.Tensor.to` و :meth:`~torch.Tensor.copy_` التي تقبل وسيط :attr:`non_blocking` صريح، مما يسمح للمستدعي بتجاوز المزامنة عندما لا تكون ضرورية.

وهناك استثناء آخر هو تيارات CUDA، الموضحة أدناه.

تيارات CUDA
^^^^^^^^^^

تيار CUDA هو تسلسل خطي للتنفيذ ينتمي إلى جهاز محدد. عادةً لا تحتاج إلى إنشاء واحد بشكل صريح: بشكل افتراضي، يستخدم كل جهاز تيار "افتراضي" الخاص به.

يتم تسلسل العمليات داخل كل تيار بالترتيب الذي يتم إنشاؤها به، ولكن يمكن أن تُنفذ العمليات من تيارات مختلفة بشكل متزامن بأي ترتيب نسبي، ما لم يتم استخدام دوال المزامنة الصريحة (مثل :meth:`~torch.cuda.synchronize` أو :meth:`~torch.cuda.Stream.wait_stream`). على سبيل المثال، الكود التالي غير صحيح::

    cuda = torch.device('cuda')
    s = torch.cuda.Stream()  # إنشاء تيار جديد.
    A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
    with torch.cuda.stream(s):
        # قد تبدأ الدالة sum() التنفيذ قبل انتهاء normal_()!
        B = torch.sum(A)

عندما يكون التيار "الحالي" هو التيار الافتراضي، يقوم PyTorch تلقائيًا بمزامنة ضرورية عند نقل البيانات، كما هو موضح أعلاه. ومع ذلك، عند استخدام تيارات غير افتراضية، يكون المستخدم مسؤولاً عن ضمان المزامنة الصحيحة. النسخة المصححة من هذا المثال هي::

    cuda = torch.device('cuda')
    s = torch.cuda.Stream()  # إنشاء تيار جديد.
    A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
    s.wait_stream(torch.cuda.default_stream(cuda))  # جديد!
    with torch.cuda.stream(s):
        B = torch.sum(A)
    A.record_stream(s)  # جديد!

هناك إضافتان جديدتان. تضمن مكالمة :meth:`torch.cuda.Stream.wait_stream` أن تنفيذ ``normal_()`` قد انتهى قبل أن نبدأ تشغيل ``sum(A)`` على تيار جانبي. تضمن :meth:`torch.Tensor.record_stream` (راجع التفاصيل للحصول على مزيد من التفاصيل) أننا لا نقوم بإلغاء تخصيص A قبل اكتمال ``sum(A)``. يمكنك أيضًا الانتظار يدويًا على التيار في وقت لاحق باستخدام ``torch.cuda.default_stream(cuda).wait_stream(s)`` (لاحظ أنه من غير المجدي الانتظار فورًا، حيث أن ذلك سيمنع تنفيذ التيار من العمل بشكل متوازٍ مع الأعمال الأخرى على التيار الافتراضي.) راجع وثائق :meth:`torch.Tensor.record_stream` للحصول على مزيد من التفاصيل حول متى يجب استخدام أحدهما أو الآخر.

لاحظ أن هذه المزامنة ضرورية حتى في حالة عدم وجود تبعية قراءة، كما هو موضح في هذا المثال::

    cuda = torch.device('cuda')
    s = torch.cuda.Stream()  # إنشاء تيار جديد.
    A = torch.empty((100, 100), device=cuda)
    s.wait_stream(torch.cuda.default_stream(cuda))  # لا يزال مطلوبًا!
    with torch.cuda.stream(s):
        A.normal_(0.0, 1.0)
        A.record_stream(s)

على الرغم من أن الحساب على التيار "s" لا يقرأ محتويات "A" ولا توجد استخدامات أخرى لـ "A"، إلا أنه لا يزال من الضروري المزامنة، لأن "A" قد تتوافق مع الذاكرة التي أعاد تخصيصها مخصص ذاكرة التخزين المؤقت CUDA، مع وجود عمليات معلقة من الذاكرة القديمة (الملغاة).

.. _bwd-cuda-stream-semantics:

دلالة التيار لعمليات التقهقر
^^^^^^^^^^^^^^^^^^^

يتم تشغيل كل عملية تقهقر CUDA على نفس التيار الذي تم استخدامه لعملية التقدير المقابلة.
إذا قمت بتشغيل عمليات مستقلة بشكل متوازٍ على تيارات مختلفة في التقدير،
فإن هذا يساعد عملية التقهقر على استغلال نفس الموازاة.

إن دلالة التيار لمكالمة تقهقر فيما يتعلق بالعمليات المحيطة هي نفسها
كما هو الحال مع أي مكالمة أخرى. تقوم عملية التقهقر بإدراج مزامنات داخلية لضمان ذلك حتى عند
تشغيل عمليات التقهقر على تيارات متعددة كما هو موضح في الفقرة السابقة.
وبشكل أكثر تحديدًا، عند استدعاء
:func:`autograd.backward<torch.autograd.backward>`،
:func:`autograd.grad<torch.autograd.grad>`، أو
:meth:`tensor.backward<torch.Tensor.backward>`،
واختياريًا توفير Tensor(s) CUDA كـ gradient(s) الأولي (على سبيل المثال،
:func:`autograd.backward(..., grad_tensors=initial_grads)<torch.autograd.backward>`،
:func:`autograd.grad(..., grad_outputs=initial_grads)<torch.autograd.grad>`، أو
:meth:`tensor.backward(..., gradient=initial_grad)<torch.Tensor.backward>`)،
تتمثل أفعال

1. اختياريًا، ملء تدرجات أولية،
2. استدعاء عملية التقهقر، و
3. استخدام التدرجات

في نفس علاقة دلالة التيار مثل أي مجموعة من العمليات::

    s = torch.cuda.Stream()

    # آمن، يتم استخدام التدرجات في نفس سياق التيار مثل backward()
    with torch.cuda.stream(s):
        loss.backward()
        استخدم التدرجات

    # غير آمن
    with torch.cuda.stream(s):
        loss.backward()
    استخدم التدرجات

    # آمن، مع المزامنة
    with torch.cuda.stream(s):
        loss.backward()
    torch.cuda.current_stream().wait_stream(s)
    استخدم التدرجات

    # آمن، يتم ملء التدرج الأولي واستدعاء عملية التقهقر في نفس سياق التيار
    with torch.cuda.stream(s):
        loss.backward(gradient=torch.ones_like(loss))

    # غير آمن، يتم ملء التدرج الأولي واستدعاء عملية التقهقر في سياقات تيارات مختلفة،
    # بدون مزامنة
    initial_grad = torch.ones_like(loss)
    with torch.cuda.stream(s):
        loss.backward(gradient=initial_grad)

    # آمن، مع المزامنة
    initial_grad = torch.ones_like(loss)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        initial_grad.record_stream(s)
        loss.backward(gradient=initial_grad)

ملاحظة BC: استخدام التدرجات على التيار الافتراضي
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

في الإصدارات السابقة من PyTorch (1.9 والإصدارات الأقدم)، قامت محرك autograd دائمًا بمزامنة
التيار الافتراضي مع جميع عمليات التقهقر، لذا فإن النمط التالي::

    with torch.cuda.stream(s):
        loss.backward()
    استخدم التدرجات

كان آمنًا طالما حدث "استخدم التدرجات" على التيار الافتراضي.
في PyTorch الحالي، لم يعد هذا النمط آمنًا. إذا كان "backward()"
و "استخدم التدرجات" في سياقات تيارات مختلفة، فيجب عليك مزامنة التيارات::

    with torch.cuda.stream(s):
        loss.backward()
    torch.cuda.current_stream().wait_stream(s)
    استخدم التدرجات

حتى إذا كان "استخدم التدرجات" على التيار الافتراضي.

.. _CUDA stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams

.. _cuda-memory-management:

إدارة الذاكرة
هذا النص بتنسيق ReStructuredText:

---------------

يستخدم PyTorch مخصص ذاكرة مؤقت للتسريع عمليات تخصيص الذاكرة. يسمح ذلك بالإفراج السريع عن الذاكرة دون مزامنات للجهاز. ومع ذلك، فإن الذاكرة غير المستخدمة التي يديرها المخصص ستظهر كما لو كانت مستخدمة في "nvidia-smi". يمكنك استخدام "torch.cuda.memory_allocated" و "torch.cuda.max_memory_allocated" لمراقبة الذاكرة التي تشغلها المصفوفات، واستخدام "torch.cuda.memory_reserved" و "torch.cuda.max_memory_reserved" لمراقبة إجمالي حجم الذاكرة التي يديرها المخصص المؤقت. يؤدي استدعاء "torch.cuda.empty_cache" إلى تحرير كل الذاكرة المؤقتة غير المستخدمة من PyTorch بحيث يمكن استخدامها من قبل تطبيقات GPU الأخرى. ومع ذلك، فإن ذاكرة GPU التي تشغلها المصفوفات لن يتم تحريرها، لذا لا يمكنها زيادة حجم ذاكرة GPU المتاحة لـ PyTorch.

لفهم أفضل لكيفية استخدام ذاكرة CUDA بمرور الوقت، يوضح القسم المرجعي "torch_cuda_memory" الأدوات الخاصة بالتقاط مخططات استخدام الذاكرة وتصويرها.

بالنسبة للمستخدمين المتقدمين، نقدم معايير أكثر شمولاً لذاكرة GPU من خلال "torch.cuda.memory_stats". كما نقدم القدرة على التقاط لقطة كاملة لحالة مخصص الذاكرة من خلال "torch.cuda.memory_snapshot"، والتي يمكن أن تساعدك على فهم أنماط التخصيص الأساسية التي ينتجها رمزك.

.. _cuda-memory-envvars:

تحسين استخدام الذاكرة باستخدام "PYTORCH_CUDA_ALLOC_CONF"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يمكن أن يتعارض استخدام مخصص مؤقت مع أدوات فحص الذاكرة مثل "cuda-memcheck". ولتصحيح أخطاء الذاكرة باستخدام "cuda-memcheck"، قم بتعيين "PYTORCH_NO_CUDA_MEMORY_CACHING=1" في بيئتك لإيقاف التشغيل المؤقت.

يمكن التحكم في سلوك المخصص المؤقت عبر المتغير البيئي "PYTORCH_CUDA_ALLOC_CONF". التنسيق هو "PYTORCH_CUDA_ALLOC_CONF=<option>:<value>,<option2>:<value2>...". الخيارات المتاحة:

* "backend" يسمح باختيار تنفيذ مخصص الأساسي. حاليًا، الخيارات الصالحة هي "native"، والتي تستخدم التنفيذ الأصلي لـ PyTorch، و "cudaMallocAsync"، والتي تستخدم مخصص CUDA الأساسي غير المتزامن. يتطلب "cudaMallocAsync" CUDA 11.4 أو أحدث. الافتراضي هو "native". ينطبق "backend" على جميع الأجهزة التي تستخدمها العملية، ولا يمكن تحديدها لكل جهاز على حدة.

* "max_split_size_mb" يمنع المخصص الأصلي من تقسيم الكتل الأكبر من هذا الحجم (بالميغابايت). يمكن أن يقلل ذلك من التجزئة وقد يسمح لبعض أحمال العمل الحدية بالاكتمال دون نفاد الذاكرة. يمكن أن تتراوح التكلفة من "صفر" إلى "كبيرة" اعتمادًا على أنماط التخصيص. القيمة الافتراضية غير محدودة، أي يمكن تقسيم جميع الكتل. تعد طرق "torch.cuda.memory_stats" و "torch.cuda.memory_summary" مفيدة للضبط الدقيق. يجب استخدام هذا الخيار كملاذ أخير لحمل العمل الذي يتم إنهاؤه بسبب "نفاد الذاكرة" ويظهر عددًا كبيرًا من الكتل غير النشطة المقسمة. "max_split_size_mb" له معنى فقط مع "backend:native". مع "backend:cudaMallocAsync"، يتم تجاهل "max_split_size_mb".

* "roundup_power2_divisions" يساعد في تقريب حجم التخصيص المطلوب إلى أقرب تقسيم لقوة 2 والاستفادة بشكل أفضل من الكتل. في المخصص الأصلي لـ CUDA، يتم تقريب الأحجام إلى أعلى بمضاعفات حجم الكتلة 512، لذا فإن هذا يعمل بشكل جيد للأحجام الصغيرة. ومع ذلك، يمكن أن يكون هذا غير فعال لتخصيصات كبيرة قريبة حيث سيذهب كل منها إلى حجم مختلف من الكتل ويتم تقليل إعادة استخدام تلك الكتل. قد يؤدي هذا إلى إنشاء العديد من الكتل غير المستخدمة وإهدار سعة ذاكرة GPU. يمكّن هذا الخيار تقريب حجم التخصيص إلى أقرب تقسيم لقوة 2. على سبيل المثال، إذا كنا بحاجة إلى تقريب حجم 1200 وإذا كان عدد التقسيمات 4، فإن الحجم 1200 يقع بين 1024 و 2048 وإذا قمنا بـ 4 تقسيمات بينهما، فإن القيم هي 1024 و 1280 و 1536 و 1792. لذلك، سيتم تقريب حجم التخصيص 1200 إلى 1280 كأقرب تقسيم لسقف قوة 2. حدد قيمة واحدة لتطبيقها على جميع أحجام التخصيص أو حدد مصفوفة من أزواج القيم الرئيسية والقيم لضبط قوة 2 بشكل فردي لكل فاصل طاقة اثنين. على سبيل المثال، لتعيين 1 تقسيم لجميع التخصيصات أقل من 256 ميجابايت، و 2 تقسيم للتخصيصات بين 256 ميجابايت و 512 ميجابايت، و 4 تقسيمات للتخصيصات بين 512 ميجابايت و 1 جيجابايت و 8 تقسيمات لأي تخصيصات أكبر، قم بتعيين قيمة المفتاح إلى: [256:1،512:2،1024:4،>:8]. "roundup_power2_divisions" له معنى فقط مع "backend:native". مع "backend:cudaMallocAsync"، يتم تجاهل "roundup_power2_divisions".

* "garbage_collection_threshold" يساعد في الاستعادة النشطة لذاكرة GPU غير المستخدمة لتجنب تشغيل عملية sync-and-reclaim-all الباهظة (release_cached_blocks)، والتي قد تكون غير مواتية لتطبيقات GPU الحساسة للاتصال (مثل الخوادم). عند تعيين هذا العتبة (على سبيل المثال، 0.8)، سيبدأ المخصص في استعادة كتل ذاكرة GPU إذا تجاوز استخدام سعة ذاكرة GPU العتبة (أي 80% من إجمالي الذاكرة المخصصة لتطبيق GPU). تفضل الخوارزمية تحرير الكتل القديمة وغير المستخدمة أولاً لتجنب تحرير الكتل التي يجري إعادة استخدامها بشكل نشط. يجب أن تكون قيمة العتبة أكبر من 0.0 وأقل من 1.0. "garbage_collection_threshold" له معنى فقط مع "backend:native". مع "backend:cudaMallocAsync"، يتم تجاهل "garbage_collection_threshold".

* "expandable_segments" (تجريبي، الافتراضي: "False") إذا تم تعيينه على "True"، فإن هذا الإعداد يوجه المخصص إلى إنشاء تخصيصات CUDA يمكن توسيعها لاحقًا للتعامل بشكل أفضل مع الحالات التي تتغير فيها أحجام التخصيص بشكل متكرر، مثل وجود حجم دفعة متغير. عادةً ما يقوم المخصص، بالنسبة للتخصيصات الكبيرة (>2 ميجابايت)، باستدعاء "cudaMalloc" للحصول على تخصيصات بنفس حجم ما يطلبه المستخدم. في المستقبل، يمكن إعادة استخدام أجزاء من هذه التخصيصات لطلبات أخرى إذا كانت مجانية. يعمل هذا بشكل جيد عندما يقوم البرنامج بالعديد من الطلبات بنفس الحجم تمامًا أو بأحجام تكون مضاعفات لذلك الحجم. يتبع العديد من نماذج التعلم العميق هذا السلوك. ومع ذلك، فإن أحد الاستثناءات الشائعة هو عندما يتغير حجم الدفعة قليلاً من تكرار إلى آخر، على سبيل المثال في الاستدلال بالدفعات. عندما يعمل البرنامج في البداية بحجم دفعة "N"، فإنه سيقوم بعمليات تخصيص مناسبة لذلك الحجم. إذا قام في المستقبل بتشغيله بحجم "N - 1"، فستظل التخصيصات الموجودة كبيرة بما يكفي. ومع ذلك، إذا تم تشغيله بحجم "N + 1"، فسيتعين عليه إجراء تخصيصات جديدة أكبر قليلاً. ليست جميع المصفوفات بنفس الحجم. قد يكون البعض "(N + 1) * A" والبعض الآخر "(N + 1) * A * B" حيث "A" و "B" هما بعض الأبعاد غير الدفعية في النموذج. نظرًا لأن المخصص يعيد استخدام التخصيصات الموجودة عندما تكون كبيرة بما يكفي، فإن بعض عدد المصفوفات من النوع "(N + 1) * A" ستتناسب بالفعل مع المقاطع الموجودة مسبقًا "N * B * A"، على الرغم من عدم كمالها. مع تشغيل النموذج، فإنه سيملأ جزئيًا جميع هذه المقاطع تاركًا شرائح ذاكرة غير مستخدمة في نهاية هذه المقاطع. في مرحلة ما، سيتعين على المخصص استدعاء "cudaMalloc" لمقطع جديد من النوع "(N + 1) * A * B". إذا لم تكن هناك ذاكرة كافية، فلن يكون هناك الآن طريقة لاسترداد شرائح الذاكرة الحرة في نهاية المقاطع الموجودة. مع النماذج التي يزيد عمقها عن 50 طبقة، قد يتكرر هذا النمط 50 مرة أو أكثر مما يؤدي إلى إنشاء العديد من الشرائح.

  يسمح "expandable_segments" للمخصص بإنشاء مقطع في البداية ثم توسيعه لاحقًا عند الحاجة إلى المزيد من الذاكرة. بدلاً من إجراء تخصيص واحد لكل تخصيص، فإنه يحاول إجراء تخصيص واحد (لكل تدفق) ينمو حسب الضرورة. الآن عندما يتم تشغيل حالة "N + 1"، ستتبلط المخصصات بشكل جميل في المقطع الكبير الواحد حتى تمتلئ. ثم يتم طلب المزيد من الذاكرة وإلحاقها بنهاية المقطع. لا تخلق هذه العملية العديد من شرائح الذاكرة غير المستخدمة، لذا فمن المرجح أن تنجح في العثور على هذه الذاكرة.

  "pinned_use_cuda_host_register" هو علم منطقي يحدد ما إذا كان سيتم استخدام دالة "cudaHostRegister" من واجهة برمجة تطبيقات CUDA لتخصيص الذاكرة المثبتة بدلاً من "cudaHostAlloc" الافتراضية. عند تعيينه إلى "True"، يتم تخصيص الذاكرة باستخدام "malloc" العادي، ثم يتم تعيين الصفحات إلى الذاكرة قبل استدعاء "cudaHostRegister". تساعد عملية تعيين الصفحات المسبقة هذه في تقليل وقت القفل أثناء تنفيذ "cudaHostRegister".

  "pinned_num_register_threads" صالح فقط عندما يكون "pinned_use_cuda_host_register" تعيينه على "True". افتراضيًا، يتم استخدام خيط واحد لتعيين الصفحات. يسمح هذا الخيار باستخدام المزيد من الخيوط لتوازي عمليات تعيين الصفحة لتقليل وقت التخصيص الإجمالي للذاكرة المثبتة. وفقًا لنتائج المعايرة، فإن القيمة الجيدة لهذا الخيار هي 8.

.. note::

    بعض الإحصاءات التي يبلغ عنها واجهة برمجة تطبيقات إدارة ذاكرة CUDA محددة لـ "backend:native"، وليس لها معنى مع "backend:cudaMallocAsync".
    راجع توثيق الدالة للحصول على التفاصيل.

.. _CUDA's built-in asynchronous allocator:
    https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/

.. _cuda-memory-custom-allocator:

استخدام مخصصات ذاكرة CUDA مخصصة
---------------------------------

من الممكن تحديد مخصصات كدالات بسيطة في C/C++ وتجميعها كمكتبة مشتركة، ويوضح الكود أدناه مخصصًا أساسيًا يقوم فقط بتتبع جميع عمليات الذاكرة.

.. code:: C++

   #include <sys/types.h>
   #include <cuda_runtime_api.h>
   #include <iostream>
   // Compile with g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
   extern "C" {
   void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
      void *ptr;
      cudaMalloc(&ptr, size);
      std::cout<<"alloc "<<ptr<<size<<std::endl;
      return ptr;
   }

   void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
      std::cout<<"free "<<ptr<< " "<<stream<<std::endl;
      cudaFree(ptr);
   }
   }


يمكن استخدام هذا في Python من خلال "torch.cuda.memory.CUDAPluggableAllocator". يتحمل المستخدم مسؤولية توفير المسار إلى ملف ".so" وأسماء دالات "alloc" و "free" التي تتطابق مع التواقيع المحددة أعلاه.

.. code:: python

   import torch

   # Load the allocator
   new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
       'alloc.so', 'my_malloc', 'my_free')
   # Swap the current allocator
   torch.cuda.memory.change_current_allocator(new_alloc)
   # This will allocate memory on the device using the new allocator
   b = torch.zeros(10, device='cuda')


.. code:: python

   import torch

   # Do an initial memory allocator
   b = torch.zeros(10, device='cuda')
   # Load the allocator
   new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
       'alloc.so', 'my_malloc', 'my_free')
   # This will error since the current allocator was already instantiated
   torch.cuda.memory.change_current_allocator(new_alloc)

.. cublas-workspaces:

مساحات عمل cuBLAS
-----------------

بالنسبة لكل مجموعة من مقبض cuBLAS وتدفق CUDA، سيتم تخصيص مساحة عمل cuBLAS إذا نفذت تلك المجموعة من المقبض والتدفق نواة cuBLAS التي تتطلب مساحة عمل. لتجنب تخصيص مساحات العمل بشكل متكرر، لا يتم إلغاء تخصيص مساحات العمل هذه إلا إذا تم استدعاء "torch._C._cuda_clearCublasWorkspaces()". يمكن تحديد حجم مساحة العمل لكل تخصيص عبر المتغير البيئي "CUBLAS_WORKSPACE_CONFIG" بالتنسيق ":[SIZE]:[COUNT]". على سبيل المثال، حجم مساحة العمل الافتراضي لكل تخصيص هو "CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8" والذي يحدد حجمًا إجماليًا يبلغ "2 * 4096 + 8 * 16 KiB". لإجبار cuBLAS على تجنب استخدام مساحات العمل، قم بتعيين "CUBLAS_WORKSPACE_CONFIG=:0:0".

.. _cufft-plan-cache:

ذاكرة التخزين المؤقت لخطة cuFFT
بالنسبة لكل جهاز CUDA، يتم استخدام ذاكرة تخزين مؤقت LRU لخطط cuFFT لتسريع تشغيل طرق FFT المتكررة (مثل: func:`torch.fft.fft`) على مصفوفات CUDA من نفس الهندسة بنفس التكوين. نظرًا لأن بعض خطط cuFFT قد تخصص ذاكرة GPU، فإن ذاكرات التخزين المؤقت هذه لها سعة قصوى.

يمكنك التحكم في خصائص ذاكرة التخزين المؤقت للجهاز الحالي والاستعلام عنها باستخدام واجهات برمجة التطبيقات التالية:

* ``torch.backends.cuda.cufft_plan_cache.max_size`` يعطي سعة ذاكرة التخزين المؤقت (افتراضي هو 4096 في CUDA 10 والإصدارات الأحدث، و 1023 في الإصدارات الأقدم من CUDA). يؤدي تعيين هذه القيمة مباشرة إلى تعديل السعة.

* ``torch.backends.cuda.cufft_plan_cache.size`` يعطي عدد الخطط الموجودة حاليًا في ذاكرة التخزين المؤقت.

* ``torch.backends.cuda.cufft_plan_cache.clear()`` يقوم بمسح ذاكرة التخزين المؤقت.

لمراقبة ذاكرات التخزين المؤقت للخطة واستعلامها لجهاز غير افتراضي، يمكنك فهرسة كائن "torch.backends.cuda.cufft_plan_cache" إما باستخدام كائن "جهاز" أو فهرس جهاز، والوصول إلى أحد السمات المذكورة أعلاه. على سبيل المثال، لتعيين سعة ذاكرة التخزين المؤقت للجهاز "1"، يمكنك كتابة "torch.backends.cuda.cufft_plan_cache[1].max_size = 10".

.. _cuda-just-in-time-compilation:

الترجمة الآنية
----------

يقوم PyTorch بالترجمة الآنية لبعض العمليات، مثل torch.special.zeta، عند تنفيذها على مصفوفات CUDA. يمكن أن تكون هذه الترجمة مكلفة من حيث الوقت (حتى بضع ثوانٍ اعتمادًا على الأجهزة والبرامج الخاصة بك) وقد تحدث عدة مرات لمشغل واحد نظرًا لأن العديد من مشغلات PyTorch تقوم بالفعل باختيار مجموعة متنوعة من النواة، ويجب تجميع كل منها مرة واحدة، اعتمادًا على مدخلاتها. تحدث هذه الترجمة مرة واحدة لكل عملية، أو مرة واحدة فقط إذا تم استخدام ذاكرة تخزين مؤقت للنواة.

بشكل افتراضي، يقوم PyTorch بإنشاء ذاكرة تخزين مؤقت للنواة في $XDG_CACHE_HOME/torch/kernels إذا تم تعريف XDG_CACHE_HOME وفي $HOME/.cache/torch/kernels إذا لم يكن كذلك (باستثناء Windows، حيث لا يتم دعم ذاكرة التخزين المؤقت للنواة بعد). يمكن التحكم في سلوك التخزين المؤقت مباشرة باستخدام متغيرين من متغيرات البيئة. إذا تم تعيين USE_PYTORCH_KERNEL_CACHE على 0، فلن يتم استخدام أي ذاكرة تخزين مؤقت، وإذا تم تعيين PYTORCH_KERNEL_CACHE_PATH، فسيتم استخدام هذا المسار كذاكرة تخزين مؤقت للنواة بدلاً من الموقع الافتراضي.

أفضل الممارسات
--------------

رمز غير مرتبط بالجهاز
^^^^^^^^^^^^^^^^^^^^

بسبب بنية PyTorch، قد تحتاج إلى كتابة رمز غير مرتبط بالجهاز (CPU أو GPU) صراحةً؛ قد يكون أحد الأمثلة هو إنشاء مصفوفة جديدة كحالة مخفية أولية لشبكة عصبية متكررة.

الخطوة الأولى هي تحديد ما إذا كان يجب استخدام GPU أم لا. نمط شائع هو استخدام وحدة "argparse" في Python لقراءة الحجج المقدمة من المستخدم، وامتلاك علم يمكن استخدامه لتعطيل CUDA، بالاقتران مع: meth:`~torch.cuda.is_available`. في ما يلي، يؤدي "args.device" إلى كائن "جهاز" يمكن استخدامه لنقل المصفوفات إلى CPU أو CUDA.

::

    import argparse
    import torch

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

.. note::

    عند تقييم توفر CUDA في بيئة معينة (:meth:`~torch.cuda.is_available`)، يكون السلوك الافتراضي لـ PyTorch هو استدعاء طريقة CUDA Runtime API `cudaGetDeviceCount`_. نظرًا لأن هذه المكالمة تقوم بدورها بتinitializing CUDA Driver API (عبر `cuInit`_) إذا لم يتم initializingها بالفعل، فإن الفروع اللاحقة لعملية تم تشغيلها: meth:`~torch.cuda.is_available` ستفشل مع خطأ تهيئة CUDA.

    يمكنك تعيين "PYTORCH_NVML_BASED_CUDA_CHECK=1" في بيئتك قبل استيراد وحدات PyTorch التي تنفذ: meth:`~torch.cuda.is_available` (أو قبل تنفيذها مباشرةً) لتوجيه: meth:`~torch.cuda.is_available` لمحاولة تقييم قائم على NVML (`nvmlDeviceGetCount_v2`_). إذا نجح التقييم القائم على NVML (أي اكتشاف/تهيئة NVML لا يفشل)، فلن تسمم مكالمات: meth:`~torch.cuda.is_available` الفروع اللاحقة.

    إذا فشل اكتشاف/تهيئة NVML، فسيتم استخدام: meth:`~torch.cuda.is_available` للتراجع إلى تقييم CUDA Runtime API الافتراضي وسينطبق قيد التفرع المذكور أعلاه.

    لاحظ أن فحص توفر CUDA القائم على NVML أعلاه يوفر ضمانًا أضعف من نهج CUDA Runtime API الافتراضي (الذي يتطلب نجاح تهيئة CUDA). في بعض الظروف، قد ينجح الفحص القائم على NVML بينما تفشل تهيئة CUDA لاحقًا.

الآن بعد أن أصبح لدينا "args.device"، يمكننا استخدامه لإنشاء مصفوفة على الجهاز المطلوب.

::

    x = torch.empty((8, 42), device=args.device)
    net = Network().to(device=args.device)

يمكن استخدام هذا في عدد من الحالات لإنتاج رمز غير مرتبط بالجهاز. فيما يلي مثال عند استخدام برنامج تحميل البيانات:

::

    cuda0 = torch.device('cuda:0')  # CUDA GPU 0
    for i, x in enumerate(train_loader):
        x = x.to(cuda0)

عند العمل باستخدام وحدات معالجة الرسومات المتعددة على نظام، يمكنك استخدام علم البيئة "CUDA_VISIBLE_DEVICES" لإدارة وحدات معالجة الرسومات المتوفرة لـ PyTorch. كما ذُكر أعلاه، للتحكم يدويًا في وحدة معالجة الرسومات التي يتم إنشاء مصفوفة عليها، فإن أفضل ممارسة هي استخدام سياق "جهاز" "torch.cuda.device".

::

    print("Outside device is 0")  # On device 0 (default in most scenarios)
    with torch.cuda.device(1):
        print("Inside device is 1")  # On device 1
    print("Outside device is still 0")  # On device 0

إذا كان لديك مصفوفة وترغب في إنشاء مصفوفة جديدة من نفس النوع على نفس الجهاز، فيمكنك استخدام طريقة "torch.Tensor.new_*" (راجع: class:`torch.Tensor`).
في حين أن وظائف "torch.*" المذكورة أعلاه تعتمد على سياق GPU الحالي وحجج السمات التي تقوم بتمريرها، فإن طرق "torch.Tensor.new_*" تحافظ على الجهاز والسمات الأخرى للمصفوفة.

هذه هي الممارسة الموصى بها عند إنشاء وحدات يتم فيها إنشاء مصفوفات جديدة داخليًا أثناء التمرير للأمام.

::

    cuda = torch.device('cuda')
    x_cpu = torch.empty(2)
    x_gpu = torch.empty(2, device=cuda)
    x_cpu_long = torch.empty(2, dtype=torch.int64)

    y_cpu = x_cpu.new_full([3, 2], fill_value=0.3)
    print(y_cpu)

        tensor([[ 0.3000,  0.3000],
                [ 0.3000,  0.3000],
                [ 0.3000,  0.3000]])

    y_gpu = x_gpu.new_full([3, 2], fill_value=-5)
    print(y_gpu)

        tensor([[-5.0000, -5.0000],
                [-5.0000, -5.0000],
                [-5.0000, -5.0000]], device='cuda:0')

    y_cpu_long = x_cpu_long.new_tensor([[1, 2, 3]])
    print(y_cpu_long)

        tensor([[ 1,  2,  3]])


إذا كنت تريد إنشاء مصفوفة من نفس النوع والحجم لمصفوفة أخرى، وملؤها إما بواحد أو صفر، فيتم توفير: meth:`~torch.ones_like` أو: meth:`~torch.zeros_like` كدالات مساعدة مريحة (والتي تحافظ أيضًا على: class:`torch.device` و: class:`torch.dtype` لمصفوفة).

::

    x_cpu = torch.empty(2, 3)
    x_gpu = torch.empty(2, 3)

    y_cpu = torch.ones_like(x_cpu)
    y_gpu = torch.zeros_like(x_gpu)


.. _cuda-memory-pinning:

استخدم مخازن الذاكرة المؤقتة للذاكرة المثبتة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

    هذه نصيحة متقدمة. إذا قمت بالإفراط في استخدام الذاكرة المثبتة، فقد يتسبب ذلك في مشكلات خطيرة عند انخفاض RAM، ويجب أن تكون على دراية بأن التثبيت غالبًا ما يكون عملية مكلفة.

تكون النسخ من المضيف إلى GPU أسرع بكثير عندما تنشأ من ذاكرة مثبتة (صفحة مقفلة). تعرض المصفوفات وتخزينات CPU طريقة: meth:`~torch.Tensor.pin_memory`، والتي تعيد نسخة من الكائن، مع وضع البيانات في منطقة مثبتة.

أيضًا، بمجرد تثبيت مصفوفة أو تخزين، يمكنك استخدام نسخ GPU غير المتزامن. ما عليك سوى تمرير حجة "non_blocking=True" إضافية إلى مكالمة: meth:`~torch.Tensor.to` أو: meth:`~torch.Tensor.cuda`. يمكن استخدام هذا لتشغيل عمليات النقل بالبيانات مع الحساب.

يمكنك جعل "DataLoader" يعيد الدفعات الموجودة في ذاكرة التخزين المؤقت المثبتة عن طريق تمرير "pin_memory=True" إلى الباني الخاص به.

.. _cuda-nn-ddp-instead:

استخدم nn.parallel.DistributedDataParallel بدلاً من multiprocessing أو nn.DataParallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يجب أن تكون معظم حالات الاستخدام التي تتضمن إدخالات مجمعة وعدة وحدات معالجة الرسومات الافتراضية لاستخدام: class:`~torch.nn.parallel.DistributedDataParallel` لاستخدام أكثر من وحدة معالجة الرسومات.

هناك تحذيرات كبيرة لاستخدام نماذج CUDA مع: mod:`~torch.multiprocessing`؛ ما لم يتم توخي الحذر لتلبية متطلبات التعامل مع البيانات بالضبط، فمن المحتمل أن يكون لبرنامجك سلوك غير صحيح أو غير محدد.

يوصى باستخدام: class:`~torch.nn.parallel.DistributedDataParallel`، بدلاً من: class:`~torch.nn.DataParallel` للقيام بالتدريب متعدد وحدات معالجة الرسومات، حتى إذا كان هناك عقدة واحدة فقط.

الفرق بين: class:`~torch.nn.parallel.DistributedDataParallel` و: class:`~torch.nn.DataParallel` هو: يستخدم: class:`~torch.nn.parallel.DistributedDataParallel` multiprocessing حيث يتم إنشاء عملية لكل وحدة معالجة الرسومات، بينما يستخدم: class:`~torch.nn.DataParallel` multithreading. من خلال استخدام multiprocessing، يكون لكل وحدة معالجة الرسومات عملية مخصصة، مما يتجنب التكاليف العامة للأداء التي يسببها GIL لمفسر Python.

إذا كنت تستخدم: class:`~torch.nn.parallel.DistributedDataParallel`، فيمكنك استخدام أداة "torch.distributed.launch" لبدء تشغيل برنامجك، راجع: ref:`distributed-launch`.

.. _cudaGetDeviceCount:
    https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f

.. _cuInit:
    https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3

.. _nvmlDeviceGetCount_v2:
    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1ga93623b195bff04bbe3490ca33c8a42d

.. _cuda-graph-semantics:

مخططات CUDA
-----------

تسجيل CUDA هو سجل للعمل (معظمه نوى وحججها) الذي يؤديه تيار CUDA والتدفقات التابعة له.
لمبادئ وتفاصيل عامة حول واجهة برمجة التطبيقات CUDA الأساسية، راجع
`البدء مع رسومات CUDA`_ و
قسم الرسومات`_ من دليل برمجة CUDA C.

يدعم PyTorch إنشاء رسومات CUDA باستخدام `التقاط التدفق`_، والذي يضع
تيار CUDA في *وضع التقاط*. لا يتم تشغيل العمل CUDA الصادر إلى تيار التقاط فعليًا
على وحدة معالجة الرسومات. بدلاً من ذلك، يتم تسجيل العمل في رسم بياني.

بعد الالتقاط، يمكن *إطلاق* الرسم البياني لتشغيل عمل وحدة معالجة الرسومات (GPU) حسب الحاجة.
يقوم كل إعادة تشغيل بتشغيل نفس النواة بنفس الحجج. بالنسبة للحجج المؤشر،
هذا يعني استخدام نفس عناوين الذاكرة.
من خلال ملء ذاكرة الإدخال ببيانات جديدة (على سبيل المثال، من دفعة جديدة) قبل كل إعادة تشغيل،
يمكنك إعادة تشغيل نفس العمل على بيانات جديدة.

لماذا الرسوم البيانية CUDA؟
^^^^^^^^^^^^^^^^^^^^

يعوض إعادة تشغيل الرسم التضحية بالمرونة الديناميكية للتنفيذ الحريص المعتاد مقابل
**انخفاض كبير في عبء العمل على وحدة المعالجة المركزية**. يتم تثبيت حجج الرسم البياني والنواة،
لذلك تخطي إعادة تشغيل الرسم البياني جميع طبقات إعداد الحجة ونشر النواة، بما في ذلك
الأحمال الزائدة لـ Python وC++ وCUDA. تحت الغطاء، يقدم إعادة التشغيل عمل الرسم البياني بالكامل
إلى وحدة معالجة الرسومات (GPU) بمكالمة واحدة إلى `cudaGraphLaunch`_. أيضًا،
تنفذ النواة في إعادة التشغيل بشكل أسرع قليلاً على وحدة معالجة الرسومات (GPU)،
ولكن التخلص من عبء العمل على وحدة المعالجة المركزية (CPU) هو الفائدة الرئيسية.

يجب عليك تجربة الرسوم البيانية CUDA إذا كانت شبكتك بأكملها أو جزء منها آمنة للرسوم البيانية
(غالبًا ما يعني ذلك أشكالًا ثابتة وتدفق تحكم ثابتًا، ولكن راجع
القيود الأخرى: ref: <capture-constraints>)
وتشتبه في أن وقت تشغيلها محدود على الأقل إلى حد ما بوحدة المعالجة المركزية.

.. _Getting Started with CUDA Graphs:
   https://developer.nvidia.com/blog/cuda-graphs/
.. _Graphs section:
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
.. _stream capture:
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture
.. _cudaGraphLaunch:
   https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597

واجهة برمجة التطبيقات PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   هذه الواجهة في الإصدار التجريبي وقد تتغير في الإصدارات المستقبلية.

يكشف PyTorch عن الرسوم البيانية عبر فئة :class: `torch.cuda.CUDAGraph` الخام
وملصقين مريحين،
:class: `torch.cuda.graph` و
:class: `torch.cuda.make_graphed_callables`.

:class: `torch.cuda.graph` هو مدير سياق بسيط ومتعدد الاستخدامات يقوم بالتقاط عمل CUDA في سياقه.
قبل الالتقاط، قم بتسخين عبء العمل الذي سيتم التقاطه عن طريق تشغيل
بضع تكرارات حريصة. يجب أن يحدث التسخين في تيار جانبي.
نظرًا لأن الرسم البياني يقرأ من عناوين الذاكرة نفسها ويكتب إليها في كل
إعادة تشغيل، يجب عليك الاحتفاظ بالإشارات طويلة الأمد إلى المنسوجات التي تحتفظ
ببيانات الإدخال والإخراج أثناء الالتقاط.
لتشغيل الرسم البياني على بيانات الإدخال الجديدة، قم بنسخ بيانات جديدة إلى منسوجات الإدخال الخاصة بالالتقاط،
قم بإعادة تشغيل الرسم البياني، ثم اقرأ الإخراج الجديد من منسوجات الإخراج الخاصة بالالتقاط.
مثال::

   ز = تورتش.كودا.كوداجراف ()

   # الإدخال الوهمي المستخدم للالتقاط
   static_input = torch.empty ((5،)، device = "cuda")

   # التسخين قبل الالتقاط
   s = torch.cuda.Stream ()
   s.wait_stream (torch.cuda.current_stream ())
   مع تورتش.كودا.ستريم (s):
       ل _ في النطاق (3):
           static_output = static_input * 2
   torch.cuda.current_stream (). wait_stream (s)

   # يلتقط الرسم البياني
   # للسماح بالالتقاط، قم تلقائيًا بتعيين تيار جانبي كتيار حالي في السياق
   مع تورتش.كودا.جراف (ز):
       static_output = static_input * 2

   # تملأ ذاكرة الرسم البياني ببيانات جديدة لحسابها
   static_input.copy_ (torch.full ((5،)، 3، device = "cuda"))
   ز.إعادة التشغيل ()
   # يحتوي static_output على النتائج
   طباعة (static_output) # مليئة 3 * 2 = 6

   # تملأ ذاكرة الرسم البياني بالمزيد من البيانات لحسابها
   static_input.copy_ (torch.full ((5،)، 4، device = "cuda"))
   ز.إعادة التشغيل ()
   طباعة (static_output) # مليئة 4 * 2 = 8

راجع
: ref: <whole-network-capture>`،
: ref: <usage-with-amp>`، و
: ref: <multistream-capture>`
لأنماط واقعية ومتقدمة.

:class: `torch.cuda.make_graphed_callables` أكثر تطوراً.
يقبل :class: `torch.cuda.make_graphed_callables` الدوال Python و
:class: `torch.nn.Module` \ s. لكل دالة أو وحدة نمطية يتم تمريرها،
فهو يقوم بإنشاء رسوم بيانية منفصلة لعمل التمرير الأمامي وعمل التمرير الخلفي. راجع
: ref: <partial-network-capture>`.

.. _capture-constraints:

القيود
~~~~~~~~

مجموعة من العمليات *قابلة للالتقاط* إذا لم تنتهك أيًا من القيود التالية.

تنطبق القيود على جميع الأعمال في
:class: `torch.cuda.graph` السياق وجميع الأعمال في التمريرات الأمامية والخلفية
من أي دالة قابلة للاستدعاء تقوم بتمريرها إلى :func: `torch.cuda.make_graphed_callables`.

من المحتمل أن يؤدي انتهاك أي من هذه القيود إلى حدوث خطأ في وقت التشغيل:

* يجب أن يحدث الالتقاط في تيار غير افتراضي. (هذا مصدر قلق فقط إذا كنت تستخدم
  :meth: `CUDAGraph.capture_begin <torch.cuda.CUDAGraph.capture_begin>` الخام و
  :meth: `CUDAGraph.capture_end <torch.cuda.CUDAGraph.capture_end>` المكالمات.
  :class: `torch.cuda.graph` و
  :func: `torch.cuda.make_graphed_callables` قم بتعيين تيار جانبي لك.)
* العمليات التي تزامن وحدة المعالجة المركزية (CPU) مع وحدة معالجة الرسومات (GPU) (مثل مكالمات `.item ()`) محظورة.
* يُسمح بعمليات مولد الأرقام العشوائية (RNG) لـ CUDA، وعند استخدام عدة حالات :class: `torch.Generator` داخل رسم بياني،
  يجب تسجيلها باستخدام :meth: `CUDAGraph.register_generator_state <torch.cuda.CUDAGraph.register_generator_state>` قبل التقاط الرسم البياني.
  تجنب استخدام :meth: `Generator.get_state <torch.get_state>` و :meth: `Generator.set_state <torch.set_state>` أثناء الالتقاط؛
  بدلاً من ذلك، استخدم :meth: `Generator.graphsafe_set_state <torch.Generator.graphsafe_set_state>` و :meth: `Generator.graphsafe_get_state <torch.Generator.graphsafe_get_state>`
  لإدارة حالات المولد بأمان داخل سياق الرسم البياني. يضمن ذلك التشغيل الصحيح لمولد الأرقام العشوائية وإدارة المولدات داخل رسوم CUDA البيانية.


من المحتمل أن يؤدي انتهاك أي من هذه القيود إلى حدوث أخطاء رقمية صامتة أو سلوك غير محدد:

* داخل العملية، لا يمكن إلا لالتقاط واحد أن يكون جاريًا في أي وقت.
* لا يجوز تشغيل أي عمل CUDA غير مُلتقط في هذه العملية (على أي مؤشر ترابط) أثناء الالتقاط.
* لا يتم التقاط عمل وحدة المعالجة المركزية (CPU). إذا تضمنت العمليات المُلتقطة عمل وحدة المعالجة المركزية (CPU)،
  فسيتم تجاهل هذا العمل أثناء إعادة التشغيل.
* تقرأ كل إعادة تشغيل من عناوين الذاكرة (الافتراضية) نفسها وتكتب إليها.
* يُحظر التدفق الديناميكي للتحكم (القائم على بيانات وحدة المعالجة المركزية (CPU) أو وحدة معالجة الرسومات (GPU)).
* تُحظر الأشكال الديناميكية. يفترض الرسم البياني أن كل منسوج في تسلسل op المُلتقط
  له نفس الحجم والتخطيط في كل إعادة تشغيل.
* يُسمح باستخدام تيارات متعددة في عملية الالتقاط، ولكن هناك :ref: <multistream-capture>`.

غير قيود
~~~~~~~~~~

* بمجرد التقاطه، يمكن إعادة تشغيل الرسم البياني على أي تيار.

.. _whole-network-capture:

التقاط الشبكة بالكامل
^^^^^^^^^^^^^^

إذا كانت شبكتك بأكملها قابلة للالتقاط، فيمكنك التقاط وإعادة تشغيل تكرار كامل::

   ن، د_في، ح، د_خروج = 640، 4096، 2048، 1024
   النموذج = تورتش.نن.سيكونتيال (تورتش.نن.لينير (د_في، ح)،
                                     تورتش.نن.دروبووت (ص = 0.2)،
                                     تورتش.نن.لينير (ح، د_خروج)،
                                     تورتش.نن.دروبووت (ص = 0.1)). كودا ()
   فقدان_دالة = تورتش.نن.مسيلوس ()
   المُحَسِّن = تورتش.أوبتيم.سجد (نموذج.باراميتيرس ()، لر = 0.1)

   # الوهمية المستخدمة للالتقاط
   static_input = torch.randn (N، D_in، device = 'cuda')
   static_target = torch.randn (N، D_out، device = 'cuda')

   # التسخين
   # يستخدم static_input و static_target هنا للراحة،
   # ولكن في إعداد حقيقي، لأن التسخين يتضمن optimizer.step ()
   # يجب عليك استخدام بضع دفعات من البيانات الحقيقية.
   s = torch.cuda.Stream ()
   s.wait_stream (torch.cuda.current_stream ())
   مع تورتش.كودا.ستريم (s):
       ل ط في النطاق (3):
           المُحَسِّن.زيرو_غراد (مجموعة_إلى_لا شيء = صحيح)
           y_pred = model (static_input)
           فقدان = فقدان_دالة (y_pred، static_target)
           فقدان.الخلفي ()
           المُحَسِّن.خطوة ()
   torch.cuda.current_stream (). wait_stream (s)

   # الالتقاط
   ز = تورتش.كودا.كوداجراف ()
   # يحدد grads إلى None قبل الالتقاط، لذا فإن backward () سيقوم بإنشاء
   # سمات .grad من مخصصات من بركة خاصة بالرسم البياني
   المُحَسِّن.زيرو_غراد (مجموعة_إلى_لا شيء = صحيح)
   مع تورتش.كودا.جراف (ز):
       static_y_pred = model (static_input)
       static_loss = فقدان_دالة (static_y_pred، static_target)
       static_loss.backward ()
       المُحَسِّن.خطوة ()

   المدخلات الحقيقية = [تورتش.راندلايك (static_input) ل _ في النطاق (10)]
   الأهداف الحقيقية = [تورتش.راندنو (N، D_out، device = "cuda") ل _ في النطاق (10)]

   لبيانات، الهدف في الرمز البريدي (المدخلات الحقيقية، الأهداف الحقيقية):
       # تملأ ذاكرة الرسم البياني ببيانات جديدة لحسابها
       static_input.copy_ (data)
       static_target.copy_ (target)
       # تتضمن إعادة التشغيل () التمرير الأمامي والخلفي والخطوة.
       # لا تحتاج حتى إلى استدعاء optimizer.zero_grad () بين التكرارات
       # لأن التمرير الخلفي المُلتقط يملأ المنسوجات .grad الثابتة في مكانها.
       ز.إعادة التشغيل ()
       # تم تحديث المعلمات. تحتوي static_y_pred و static_loss و .grad
       # السمات القيم من الحساب على بيانات هذه الدورة.

.. _partial-network-capture:

التقاط الشبكة الجزئي
^^^^^^^^^^^^^^^^^^^^^^^^^

إذا كان جزء من شبكتك غير آمن للالتقاط (على سبيل المثال، بسبب تدفق التحكم الديناميكي،
الأشكال الديناميكية، أو عمليات المزامنة بين وحدة المعالجة المركزية (CPU) أو منطق جانب وحدة المعالجة المركزية (CPU) الضروري)،
يمكنك تشغيل الجزء غير الآمن (الأجزاء) بحماس واستخدام :func: `torch.cuda.make_graphed_callables`
لإنشاء رسم بياني فقط للجزء (الأجزاء) الآمنة للالتقاط.

بشكل افتراضي، تكون الدوال التي تم إرجاعها بواسطة :func: `torch.cuda.make_graphed_callables`
مدركة لـ autograd، ويمكن استخدامها في حلقة التدريب كبدائل مباشرة
للوظائف أو الوحدات النمطية :class: `nn.Module <torch.nn.Module>` التي قمت بتمريرها.

ينشئ :func: `torch.cuda.make_graphed_callables` داخليًا كائنات :class: `torch.cuda.CUDAGraph`،
ويقوم بتشغيل تكرارات التسخين، ويحافظ على الإدخالات والإخراجات الثابتة حسب الحاجة.
لذلك (على عكس :class: `torch.cuda.graph`) لا تحتاج إلى التعامل معها يدويًا.

في المثال التالي، يعني تدفق التحكم المعتمد على البيانات أن
الشبكة غير قابلة للالتقاط من البداية إلى النهاية، ولكن
:func: `torch.cuda.make_graphed_callables`
يتيح لنا التقاط وتشغيل الأقسام الآمنة للرسم البياني كرسوم بيانية بغض النظر::

   ن، د_في، ح، د_خروج = 640، 4096، 2048، 1024

   الوحدة النمطية 1 = تورتش.نن.لينير (د_في، ح). كودا ()
   الوحدة النمطية 2 = تورتش.نن.لينير (ح، د_خروج). كودا ()
   الوحدة النمطية 3 = تورتش.نن.لينير (ح، د_خروج). كودا ()

   فقدان_دالة = تورتش.نن.مسيلوس ()
   المُحَسِّن = تورتش.أوبتيم.سجد (سلسلة (معلمات الوحدة النمطية 1،
                                     معلمات الوحدة النمطية 2،
                                     معلمات الوحدة النمطية 3)،
                               لر = 0.1)

   # عينات الإدخال المستخدمة للالتقاط
   # يجب أن تتطابق حالة requires_grad لعينات الإدخال
   # تتطابق حالة requires_grad للإدخالات الفعلية التي سيشاهدها كل استدعاء قابل للاستدعاء.
   x = torch.randn (N، D_in، device = 'cuda')
   ح = تورتش.راندنو (ن، ح، الجهاز = "كودا"، يتطلب_غراد = صحيح)

   الوحدة النمطية 1 = تورتش.كودا.ماكي_جرافيد_كالابلز (الوحدة النمطية 1، (س،))
   الوحدة النمطية 2 = تورتش.كودا.
هذا هو النص المترجم إلى اللغة العربية بتنسيق ReStructuredText:

بالنسبة للمحسنات النموذجية، :meth:`GradScaler.step<torch.cuda.amp.GradScaler.step>` تزامن
الوحدة المركزية مع وحدة معالجة الرسومات، وهو ما يُمنع أثناء الالتقاط. لتجنب الأخطاء، يمكنك إما استخدام
:ref:`partial-network capture<partial-network-capture>`، أو (إذا كانت العمليات forward و loss
و backward آمنة للالتقاط) قم بالتقاط العمليات forward و loss و backward ولكن ليس
خطوة المحسن::

    # warmup
    # في إعداد حقيقي، استخدم عدة دفعات من البيانات الحقيقية.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                y_pred = model(static_input)
                loss = loss_fn(y_pred, static_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        with torch.cuda.amp.autocast():
            static_y_pred = model(static_input)
            static_loss = loss_fn(static_y_pred, static_target)
        scaler.scale(static_loss).backward()
        # لا تلتقط scaler.step(optimizer) أو scaler.update()

    real_inputs = [torch.rand_like(static_input) for _ in range(10)]
    real_targets = [torch.rand_like(static_target) for _ in range(10)]

    for data, target in zip(real_inputs, real_targets):
        static_input.copy_(data)
        static_target.copy_(target)
        g.replay()
        # تشغيل scaler.step و scaler.update بشكل متلهف
        scaler.step(optimizer)
        scaler.update()

.. _multistream-capture:

الاستخدام مع عدة دفقات
^^^^^^^^^^^^^^^^^^^^^^^^^^^

ينتشر وضع الالتقاط تلقائيًا إلى أي دفقات تتزامن مع دفق التقاط.
ضمن الالتقاط، يمكنك عرض الموازية من خلال إصدار مكالمات إلى دفقات مختلفة،
ولكن يجب أن تتفرع شجرة الاعتماد الإجمالية للدفق من
الدفق الأولي الذي يلتقط بعد بدء الالتقاط وينضم إلى الدفق الأولي
قبل انتهاء الالتقاط::

    with torch.cuda.graph(g):
        # في مدخل سياق المدير، torch.cuda.current_stream()
        # هو دفق الالتقاط الأولي

        # غير صحيح (لا يتفرع من الدفق الأولي أو ينضم إليه)
        with torch.cuda.stream(s):
            cuda_work()

        # صحيح:
        # يتفرع من الدفق الأولي
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            cuda_work()
        # ينضم إلى الدفق الأولي قبل انتهاء الالتقاط
        torch.cuda.current_stream().wait_stream(s)

.. note::

    لتجنب الارتباك للمستخدمين المتقدمين الذين يبحثون عن عمليات إعادة التشغيل في nsight systems أو nvprof:
    على عكس التنفيذ المتلهف، يفسر الرسم البياني شجرة DAG غير تافهة للدفق في الالتقاط
    كإشارة، وليس أمرًا. أثناء إعادة التشغيل، قد يعيد الرسم البياني تنظيم العمليات المستقلة
    إلى دفقات مختلفة أو وضعها في قائمة انتظار حسب ترتيب مختلف (مع احترام التبعيات الإجمالية لـ DAG الأصلي).

الاستخدام مع DistributedDataParallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NCCL < 2.9.6
~~~~~~~~~~~~

لا تسمح إصدارات NCCL الأقدم من 2.9.6 بالتقاط الجماعات.
يجب عليك استخدام :ref:`partial-network capture<partial-network-capture>`،
الذي يؤجل حدوث allreduces خارج الأقسام المخططة للخلف.

اتصل بـ :func:`~torch.cuda.make_graphed_callables` على أقسام الشبكة القابلة للرسم
*قبل* لف الشبكة في DDP.

NCCL >= 2.9.6
~~~~~~~~~~~~~

تسمح إصدارات NCCL 2.9.6 أو الأحدث بالجماعية في الرسم البياني.
النهج التي تلتقط :ref:`pass backward بالكامل<whole-network-capture>`
هي خيار قابل للتطبيق، ولكنها تحتاج إلى ثلاث خطوات إعداد.

1. تعطيل التعامل مع الأخطاء غير المتزامن الداخلي لـ DDP::

    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    torch.distributed.init_process_group(...)

2. قبل التقاط full-backward، يجب إنشاء DDP في سياق دفق جانبي::

    with torch.cuda.stream(s):
        model = DistributedDataParallel(model)

3. يجب أن يعمل التسخين الخاص بك على تشغيل 11 دورة على الأقل من التكرارات المتلهفة لـ DDP قبل الالتقاط.

.. _graph-memory-management:

إدارة ذاكرة الرسم البياني
^^^^^^^^^^^^^^^^^^^^^^^

يعمل الرسم البياني الملتقط على نفس العناوين الافتراضية في كل مرة يتم فيها إعادة تشغيله.
إذا حررت PyTorch الذاكرة، فقد يؤدي إعادة التشغيل اللاحق إلى حدوث خطأ في الوصول إلى الذاكرة غير القانونية.
إذا قامت PyTorch بإعادة تعيين الذاكرة لمصفوفات جديدة، فقد يؤدي إعادة التشغيل إلى إتلاف القيم
التي تشاهدها تلك المصفوفات. لذلك، يجب حجز عناوين افتراضية يستخدمها الرسم البياني للرسم البياني عبر عمليات إعادة التشغيل. يحقق مخصص التخزين المؤقت لـ PyTorch ذلك
من خلال اكتشاف ما إذا كان الالتقاط جاريًا وتلبية تخصيصات الالتقاط
من بركة ذاكرة خاصة بالرسم البياني. تظل البركة الخاصة نشطة حتى
يتم إخراج كائن :class:`~torch.cuda.CUDAGraph` الخاص به وجميع المصفوفات التي تم إنشاؤها أثناء الالتقاط
عن النطاق.

يتم الحفاظ على البرك الخاصة تلقائيًا. بشكل افتراضي، يقوم المخصص بإنشاء
بركة خاصة منفصلة لكل عملية التقاط. إذا قمت بالتقاط عدة رسوم بيانية،
تضمن هذه الطريقة المحافظة عدم إتلاف عمليات إعادة تشغيل الرسم البياني لقيم بعضها البعض مطلقًا،
ولكنه أحيانًا يهدر الذاكرة بلا داع.

مشاركة الذاكرة عبر عمليات الالتقاط
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

لتوفير الذاكرة المخزنة في البرك الخاصة، :class:`torch.cuda.graph`
و :func:`torch.cuda.make_graphed_callables` تسمح بشكل اختياري لعمليات الالتقاط المختلفة
لمشاركة نفس البركة الخاصة.
من الآمن لمجموعة من الرسوم البيانية مشاركة بركة خاصة إذا كنت تعلم أنها ستتم دائمًا
يتم إعادة تشغيلها بنفس الترتيب الذي تم التقاطها به،
ولن يتم إعادة تشغيلها أبدًا في نفس الوقت.

حجة "المسبح" :class:`torch.cuda.graph` هي تلميح لاستخدام بركة خاصة معينة،
ويمكن استخدامه لمشاركة الذاكرة عبر الرسوم البيانية على النحو الموضح::

    g1 = torch.cuda.CUDAGraph()
    g2 = torch.cuda.CUDAGraph()

    # (إنشاء إدخالات ثابتة لـ g1 و g2، تشغيل عمليات الإحماء لأحمال العمل الخاصة بها...)

    # يلتقط g1
    with torch.cuda.graph(g1):
        static_out_1 = g1_workload(static_in_1)

    # يلتقط g2، مما يوحي بأن g2 قد تشارك بركة ذاكرة مع g1
    with torch.cuda.graph(g2, pool=g1.pool()):
        static_out_2 = g2_workload(static_in_2)

    static_in_1.copy_(real_data_1)
    static_in_2.copy_(real_data_2)
    g1.replay()
    g2.replay()

مع :func:`torch.cuda.make_graphed_callables`، إذا كنت تريد رسم عدة
دالات قابلة للاستدعاء وتعرف أنها ستعمل دائمًا بنفس الترتيب (ولن تعمل أبدًا في نفس الوقت)
مررها كزوج بنفس الترتيب الذي ستعمل به في عبء العمل المباشر، و
:func:`~torch.cuda.make_graphed_callables` سيتم التقاط رسوماتها باستخدام بركة خاصة مشتركة.

إذا كان سيتم تشغيل الدالات القابلة للاستدعاء في عبء العمل المباشر بترتيب يتغير أحيانًا،
أو إذا كانوا يعملون في نفس الوقت، فلا يُسمح بتمريرها كزوج إلى استدعاء واحد لـ
:func:`~torch.cuda.make_graphed_callables`. بدلاً من ذلك، يجب عليك استدعاء
:func:`~torch.cuda.make_graphed_callables` بشكل منفصل لكل منها.