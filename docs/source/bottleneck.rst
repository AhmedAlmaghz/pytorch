torch.utils.bottleneck
======================

.. automodule:: torch.utils.bottleneck
.. currentmodule:: torch.utils.bottleneck

تعد ``torch.utils.bottleneck`` أداة يمكن استخدامها كخطوة أولية لتصحيح الأخطاء في برنامجك. فهو يلخص تشغيل برنامجك باستخدام ملف تعريف Python وملف تعريف PyTorch autograd.

لتشغيله على سطر الأوامر:

::

    python -m torch.utils.bottleneck /path/to/source/script.py [args]

حيث [args] هو أي عدد من الحجج إلى ``script.py``، أو تشغيل ``python -m torch.utils.bottleneck -h`` للحصول على تعليمات الاستخدام.

.. warning::
    نظرًا لأن برنامجك سيتم ملف تعريفه، يرجى التأكد من أنه ينتهي في فترة زمنية محددة.

.. warning::
    بسبب الطبيعة غير المتزامنة لنواة CUDA، عند التشغيل مقابل كود CUDA، قد لا يظهر ملف تعريف cProfile وملف تعريف autograd CPU-mode أوقاتًا صحيحة: حيث يبلغ وقت CPU المبلغ عنه مقدار الوقت المستخدم لبدء تشغيل النواة ولكنه لا يتضمن الوقت الذي استغرقته النواة في التنفيذ على GPU ما لم تقم العملية بالمزامنة. تبدو العمليات التي تقوم بالمزامنة مكلفة للغاية في ملفات تعريف CPU-mode العادية.
    في هذه الحالة حيث تكون الأوقات غير صحيحة، قد يكون ملف تعريف autograd CUDA-mode مفيدًا.

.. note::
    لتحديد مخرج ملف تعريف autograd (CPU-only-mode أو CUDA-mode) الذي يجب النظر فيه، يجب عليك أولاً التحقق مما إذا كان برنامجك مقيدًا بـ CPU ("CPU total time أكبر بكثير من CUDA total time").
    إذا كان مقيدًا بـ CPU، فسوف يساعد النظر في نتائج ملف تعريف autograd CPU-mode. من ناحية أخرى، إذا كان برنامجك يقضي معظم وقته في التنفيذ على وحدة معالجة الرسومات (GPU)، فمن المنطقي إذن البحث عن مشغلي CUDA المسؤولين في مخرج ملف تعريف autograd CUDA-mode.

    بالطبع، الواقع أكثر تعقيدًا وقد لا يكون برنامجك في أحد هذين النقيضين اعتمادًا على جزء النموذج الذي تقوم بتقييمه. إذا لم تكن مخرجات الملف الشخصي مفيدة، فيمكنك تجربة النظر في نتيجة :func:`torch.autograd.profiler.emit_nvtx() <torch.autograd.profiler.emit_nvtx.html>` مع ``nvprof``.
    ومع ذلك، يرجى مراعاة أن Overhead NVTX مرتفع للغاية وغالبًا ما يعطي جدول زمني متحيز بشدة. وبالمثل، يساعد "Intel® VTune™ Profiler" في إجراء مزيد من التحليل للأداء على منصات Intel باستخدام :func:`torch.autograd.profiler.emit_itt() <torch.autograd.profiler.emit_itt.html>`.

.. warning::
    إذا كنت تقوم بملف تعريف رمز CUDA، فإن أول ملف تعريف يقوم بتشغيله ``bottleneck`` (cProfile) سيشتمل على وقت بدء تشغيل CUDA (تكلفة تخصيص مؤشر ترابط CUDA) في إعداد تقارير الوقت الخاصة به. هذا لا يهم إذا كانت عنق الزجاجة لديك تؤدي إلى كود أبطأ بكثير من وقت بدء تشغيل CUDA.

لاستخدامات أكثر تعقيدًا لملفات التعريف (كما هو الحال في حالة استخدام عدة وحدات GPU)، يرجى الاطلاع على https://docs.python.org/3/library/profile.html أو :func:`torch.autograd.profiler.profile() <torch.autograd.profiler.profile.html>` لمزيد من المعلومات.