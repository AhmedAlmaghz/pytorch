.. role:: hidden
    :class: hidden-section

توازي المنسوجات - torch.distributed.tensor.parallel
===============================================

تم بناء توازي المنسوجات (TP) على PyTorch DistributedTensor
(`DTensor <https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md>`__)
ويوفر أساليب توازي مختلفة: التوازي العمودي والتوازي الصفّي، وتوازي التسلسل.

.. warning ::
    واجهات برمجة التطبيقات الخاصة بتوازي المنسوجات تجريبية وقد تتغير.

نقطة الدخول لتوازي وحدتك ``nn.Module`` باستخدام توازي المنسوجات هي:

.. automodule:: torch.distributed.tensor.parallel

.. currentmodule:: torch.distributed.tensor.parallel

.. autofunction::  parallelize_module

يدعم توازي المنسوجات أساليب التوازي التالية:

.. autoclass:: torch.distributed.tensor.parallel.ColwiseParallel
  :members:
  :undoc-members:

.. autoclass:: torch.distributed.tensor.parallel.RowwiseParallel
  :members:
  :undoc-members:

.. autoclass:: torch.distributed.tensor.parallel.SequenceParallel
  :members:
  :undoc-members:

لإعداد مدخلات ومخرجات ``nn.Module`` ببساطة باستخدام تخطيطات DTensor
وإجراء عمليات إعادة توزيع التخطيط الضرورية، دون توزيع معلمات الوحدة
على DTensors، يمكن استخدام أنماط التوازي التالية (ParallelStyles) في
خطة التوازي عند استدعاء ``parallelize_module``:

.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleInput
  :members:
  :undoc-members:

.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleOutput
  :members:
  :undoc-members:

.. note:: عند استخدام ``Shard(dim)`` كتخطيطات المدخلات/المخرجات لأنماط التوازي المذكورة أعلاه،
  نفترض أن منسوجات تنشيط المدخلات/المخرجات مجزأة بالتساوي على
  بعد المنسوجات ``dim`` على ``DeviceMesh`` التي تعمل عليها TP. على سبيل المثال،
  نظرًا لأن ``RowwiseParallel`` يقبل المدخلات التي يتم تجزئتها على البعد الأخير، فإنه يفترض
  أن منسوجة الإدخال قد تم تجزئتها بالتساوي بالفعل على البعد الأخير. في حالة منسوجات التنشيط غير المتساوية،
  يمكن تمرير DTensor مباشرةً إلى الوحدات المجزأة،
  واستخدام ``use_local_output=False`` لإرجاع DTensor بعد كل ``ParallelStyle``، حيث
  يمكن لـ DTensor تتبع معلومات التجزئة غير المتساوية.

بالنسبة للنماذج مثل Transformer، نوصي المستخدمين باستخدام ``ColwiseParallel``
و ``RowwiseParallel`` معًا في خطة التوازي لتحقيق التجزئة المرغوبة
للنموذج بالكامل (أي Attention وMLP).

يتم دعم الحساب الموازي للخسارة المتقاطعة (loss parallelism) عبر مدير السياق التالي:

.. autofunction:: torch.distributed.tensor.parallel.loss_parallel

.. warning ::
    واجهة برمجة التطبيقات loss_parallel تجريبية وقد تتغير.

هل هناك أي شيء آخر يمكنني المساعدة فيه؟