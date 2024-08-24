.. _hip-semantics:

معاني HIP (ROCm)
====================

ROCm\ |trade| هي منصة البرمجيات مفتوحة المصدر من AMD للحوسبة عالية الأداء والتعلم الآلي المعجل بواسطة GPU. تم تصميم HIP، وهي لهجة C++ الخاصة بـ ROCm، لتسهيل تحويل تطبيقات CUDA إلى كود C++ محمول. يتم استخدام HIP عند تحويل تطبيقات CUDA الحالية مثل PyTorch إلى C++ محمول وللمشاريع الجديدة التي تتطلب قابلية للتشغيل بين AMD و NVIDIA.

.. _hip_as_cuda:

واجهات HIP تعيد استخدام واجهات CUDA
-------------------------------

تستخدم PyTorch لـ HIP عن قصد واجهات :mod:`torch.cuda` الحالية. يساعد ذلك في تسريع نقل كود PyTorch ونماذجه الحالية لأن التغييرات في الكود، إن وجدت، قليلة جدًا.

سيتم تشغيل المثال من :ref:`cuda-semantics` بنفس الطريقة تمامًا لـ HIP::

    cuda = torch.device('cuda')     # جهاز HIP الافتراضي
    cuda0 = torch.device('cuda:0')  # 'rocm' أو 'hip' غير صالحين، استخدم 'cuda'
    cuda2 = torch.device('cuda:2')  # المعالج الرسومي 2 (فهي مفهرسة من 0)

    x = torch.tensor([1., 2.], device=cuda0)
    # x.device هي device(type='cuda', index=0)
    y = torch.tensor([1., 2.]).cuda()
    # y.device هي device(type='cuda', index=0)

    with torch.cuda.device(1):
        # يقوم بتخصيص مصفوفة على المعالج الرسومي 1
        a = torch.tensor([1., 2.], device=cuda)

        # ينقل مصفوفة من وحدة المعالجة المركزية إلى المعالج الرسومي 1
        b = torch.tensor([1., 2.]).cuda()
        # a.device و b.device هما device(type='cuda', index=1)

        # يمكنك أيضًا استخدام ``Tensor.to`` لنقل مصفوفة:
        b2 = torch.tensor([1., 2.]).to(device=cuda)
        # b.device و b2.device هما device(type='cuda', index=1)

        c = a + b
        # c.device هي device(type='cuda', index=1)

        z = x + y
        # z.device هي device(type='cuda', index=0)

        # حتى ضمن سياق، يمكنك تحديد الجهاز
        # (أو إعطاء فهرس GPU لمكالمة .cuda)
        d = torch.randn(2, device=cuda2)
        e = torch.randn(2).to(cuda2)
        f = torch.randn(2).cuda(cuda2)
        # d.device و e.device و f.device جميعها device(type='cuda', index=2)

.. _checking_for_hip:

التحقق من وجود HIP
----------------

سواء كنت تستخدم PyTorch لـ CUDA أو HIP، ستكون نتيجة استدعاء :meth:`~torch.cuda.is_available` هي نفسها. إذا كنت تستخدم PyTorch الذي تم بناؤه مع دعم GPU، فسيتم إرجاع `True`. إذا كنت بحاجة إلى التحقق من إصدار PyTorch الذي تستخدمه، فيمكنك الرجوع إلى هذا المثال أدناه::

    if torch.cuda.is_available() and torch.version.hip:
        # قم بعمل شيء محدد لـ HIP
    elif torch.cuda.is_available() and torch.version.cuda:
        # قم بعمل شيء محدد لـ CUDA

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:

.. _tf32_on_rocm:

TensorFloat-32(TF32) على ROCm
----------------------------

TF32 غير مدعوم في ROCm.

.. _rocm-memory-management:

إدارة الذاكرة
-----------------

يستخدم PyTorch مخصص ذاكرة التخزين المؤقت لتسريع تخصيصات الذاكرة. يسمح ذلك بالإلغاء السريع لتخصيص الذاكرة دون عمليات مزامنة للجهاز. ومع ذلك، ستظل الذاكرة غير المستخدمة التي يديرها المخصص تظهر كما لو كانت مستخدمة في ``rocm-smi``. يمكنك استخدام :meth:`~torch.cuda.memory_allocated` و :meth:`~torch.cuda.max_memory_allocated` لمراقبة الذاكرة التي تشغلها المصفوفات، واستخدام :meth:`~torch.cuda.memory_reserved` و :meth:`~torch.cuda.max_memory_reserved` لمراقبة إجمالي مقدار الذاكرة التي يديرها مخصص التخزين المؤقت. يؤدي استدعاء :meth:`~torch.cuda.empty_cache` إلى تحرير كل الذاكرة **غير المستخدمة** المخزنة مؤقتًا من PyTorch بحيث يمكن استخدامها بواسطة تطبيقات GPU الأخرى. ومع ذلك، لن يتم تحرير ذاكرة GPU التي تشغلها المصفوفات، لذا لا يمكنها زيادة مقدار ذاكرة GPU المتاحة لـ PyTorch.

بالنسبة للمستخدمين المتقدمين، نقدم معايير أكثر شمولاً لذاكرة التخزين المؤقت عبر :meth:`~torch.cuda.memory_stats`. كما نقدم القدرة على التقاط لقطة كاملة لحالة مخصص الذاكرة عبر :meth:`~torch.cuda.memory_snapshot`، والتي يمكن أن تساعدك في فهم أنماط التخصيص الأساسية التي ينتجها كودك.

لتصحيح أخطاء الذاكرة، قم بتعيين ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` في بيئتك لتعطيل التخزين المؤقت.

.. _hipfft-plan-cache:

ذاكرة التخزين المؤقت لخطة hipFFT/rocFFT
---------------------------------

لا يتم دعم تحديد حجم ذاكرة التخزين المؤقت لخطط hipFFT/rocFFT.

.. _torch-distributed-backends:

خلفيات torch.distributed
--------------------------

حاليًا، يتم دعم خلفيات "nccl" و "gloo" فقط لـ torch.distributed على ROCm.

.. _cuda-api-to_hip-api-mappings:

CUDA API إلى خرائط HIP API في C++
-----------------------------------

يرجى الرجوع إلى: https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP_API_Guide.html

ملاحظة: لا يتم رسميًا تعيين ماكرو CUDA_VERSION، وcudaRuntimeGetVersion وcudaDriverGetVersion APIs إلى نفس القيم مثل ماكرو HIP_VERSION، وhipRuntimeGetVersion وhipDriverGetVersion APIs. يرجى عدم استخدامها بشكل متبادل عند إجراء فحوصات الإصدار.

على سبيل المثال: بدلاً من استخدام

``#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000`` لاستبعاد ROCm/HIP بشكل ضمني،

استخدم ما يلي لعدم اتخاذ مسار الكود لـ ROCm/HIP:

``#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && !defined(USE_ROCM)``

بدلاً من ذلك، إذا كان من المرغوب فيه اتخاذ مسار الكود لـ ROCm/HIP:

``#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11000) || defined(USE_ROCM)``

أو إذا كان من المرغوب فيه اتخاذ مسار الكود لـ ROCm/HIP فقط لإصدارات HIP المحددة:

``#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11000) || (defined(USE_ROCM) && ROCM_VERSION >= 40300)``


الرجوع إلى وثيقة معاني CUDA
----------------------

بالنسبة لأي أقسام غير مدرجة هنا، يرجى الرجوع إلى وثيقة معاني CUDA: :ref:`cuda-semantics`


تمكين تأكيدات kernel
------------------

تأكيدات kernel مدعومة في ROCm، ولكنها معطلة بسبب عبء الأداء. يمكن تمكينها
عن طريق إعادة تجميع PyTorch من المصدر.

يرجى إضافة السطر التالي كحجة إلى معلمات أمر cmake::

    -DROCM_FORCE_ENABLE_GPU_ASSERTS:BOOL=ON