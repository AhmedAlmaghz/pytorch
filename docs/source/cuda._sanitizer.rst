.. currentmodule:: torch.cuda._sanitizer

CUDA Stream Sanitizer
=====================

.. note::
    هذه ميزة تجريبية، مما يعني أنها في مرحلة مبكرة لتلقي الملاحظات والاختبار، وقد تخضع مكوناتها للتغيير.

نظرة عامة
--------

.. automodule:: torch.cuda._sanitizer

الاستخدام
------

فيما يلي مثال على خطأ تزامن بسيط في PyTorch:

::

    import torch

    a = torch.rand(4, 2, device="cuda")

    with torch.cuda.stream(torch.cuda.Stream()):
        torch.mul(a, 5, out=a)

يتم تهيئة المصفوفة "a" على دفق الافتراضي، وبدون أي طرق تزامن، يتم تعديلها على دفق جديد. سيتم تشغيل نواة المعالجة بشكل متزامن على نفس المصفوفة، مما قد يتسبب في قراءة النواة الثانية لبيانات غير مستهلة قبل أن تتمكن النواة الأولى من كتابتها، أو قد تقوم النواة الأولى بكتابة جزء من نتيجة النواة الثانية.

عند تشغيل هذا البرنامج النصي على سطر الأوامر باستخدام:
::

    TORCH_CUDA_SANITIZER=1 python example_error.py

تتم طباعة الإخراج التالي بواسطة CSAN:

::

    ============================
    اكتشف CSAN احتمال حدوث سباق بيانات على المصفوفة ذات مؤشر البيانات 139719969079296
    تم الوصول إليه بواسطة الدفق 94646435460352 أثناء النواة:
    aten::mul.out(Tensor self، Tensor other، *، Tensor(a!) out) -> Tensor(a!)
    الكتابة إلى الحجة (ق) الذاتية، out، وإلى الإخراج
    مع تتبع المكدس:
      File "example_error.py"، line 6، in <module>
        torch.mul(a, 5, out=a)
      ...
      File "pytorch/torch/cuda/_sanitizer.py"، line 364، in _handle_kernel_launch
        stack_trace = traceback.StackSummary.extract(

    الوصول السابق بواسطة الدفق 0 أثناء النواة:
    aten::rand(int[] size، *، int؟ dtype=None، Device؟ device=None) -> Tensor
    الكتابة إلى الإخراج
    مع تتبع المكدس:
      File "example_error.py"، line 3، in <module>
        a = torch.rand(10000, device="cuda")
      ...
      File "pytorch/torch/cuda/_sanitizer.py"، line 364، in _handle_kernel_launch
        stack_trace = traceback.StackSummary.extract(

    تم تخصيص المصفوفة مع تتبع المكدس:
      File "example_error.py"، line 3، in <module>
        a = torch.rand(10000, device="cuda")
      ...
      File "pytorch/torch/cuda/_sanitizer.py"، line 420، in _handle_memory_allocation
        traceback.StackSummary.extract(

يقدم هذا شرحًا وافيًا لسبب الخطأ:

- تم الوصول إلى مصفوفة بشكل غير صحيح من دفقات ذات معرفات: 0 (دفق الافتراضي) و94646435460352 (دفق جديد)
- تم تخصيص المصفوفة عن طريق استدعاء ``a = torch.rand(10000, device="cuda")``
- تسبب عمليات التشغيل غير الصحيحة في:
    - ``a = torch.rand(10000, device="cuda")`` على الدفق 0
    - ``torch.mul(a, 5, out=a)`` على الدفق 94646435460352
- تعرض رسالة الخطأ أيضًا مخططات مشغلي التشغيل المستدعى، بالإضافة إلى ملاحظة
  تُظهر أي حجج للمشغلين المقابلين للمصفوفة المتأثرة.

  - في المثال، يمكن ملاحظة أن المصفوفة "a" تتوافق مع حجج "self" و"out"
    وقيمة "الإخراج" لمشغل "torch.mul" المستدعى.

.. seealso::
    يمكن عرض قائمة مشغلي تورتش المدعومة ومخططاتهم :doc:`هنا <torch>`.

يمكن إصلاح الخلل عن طريق إجبار الدفق الجديد على الانتظار حتى ينتهي الدفق الافتراضي:

::

    with torch.cuda.stream(torch.cuda.Stream()):
        torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
        torch.mul(a, 5, out=a)

عند تشغيل البرنامج النصي مرة أخرى، لا يتم الإبلاغ عن أي أخطاء.

مرجع API
---------

.. autofunction:: enable_cuda_sanitizer