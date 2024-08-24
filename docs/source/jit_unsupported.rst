.. _jit_unsupported:

عمليات PyTorch غير المدعومة في TorchScript
=======================================

الصفات غير المدعومة في Torch و Tensor
----------------------------------

يدعم TorchScript معظم الطرق المحددة في "torch" و "torch.Tensor"، ولكن لا يوجد لدينا تغطية كاملة.
فيما يلي عمليات وفئات محددة من العمليات التي تختلف سلوكياتها بين
بايثون و TorchScript. إذا صادفت شيئًا آخر غير مدعوم، يرجى
إرسال مشكلة على GitHub. العمليات المتقادمة غير مدرجة أدناه.

.. automodule:: torch.jit.unsupported_tensor_ops

الوظائف غير المرتبطة بشكل صحيح على Torch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ستفشل الوظائف التالية إذا تم استخدامها في TorchScript، إما لأنها
غير مرتبطة بـ `torch` أو لأن بايثون تتوقع مخططًا مختلفًا عن
TorchScript.

  * :func:`torch.tensordot`
  * :func:`torch.nn.init.calculate_gain`
  * :func:`torch.nn.init.eye_`
  * :func:`torch.nn.init.dirac_`
  * :func:`torch.nn.init.kaiming_normal_`
  * :func:`torch.nn.init.orthogonal_`
  * :func:`torch.nn.init.sparse`

فئات العمليات ذات المخططات المتباينة بين Torch و Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

توجد الفئات التالية من العمليات ذات المخططات المتباينة:

الوظائف التي تقوم ببناء المتوترات من الإدخالات غير المتوترة لا تدعم الحجة `requires_grad`،
باستثناء `torch.tensor`. يغطي هذا العمليات التالية:

  * :func:`torch.norm`
  * :func:`torch.bartlett_window`
  * :func:`torch.blackman_window`
  * :func:`torch.empty`
  * :func:`torch.empty_like`
  * :func:`torch.empty_strided`
  * :func:`torch.eye`
  * :func:`torch.full`
  * :func:`torch.full_like`
  * :func:`torch.hamming_window`
  * :func:`torch.hann_window`
  * :func:`torch.linspace`
  * :func:`torch.logspace`
  * :func:`torch.normal`
  * :func:`torch.ones`
  * :func:`torch.rand`
  * :func:`torch.rand_like`
  * :func:`torch.randint_like`
  * :func:`torch.randn`
  * :func:`torch.randn_like`
  * :func:`torch.randperm`
  * :func:`torch.tril_indices`
  * :func:`torch.triu_indices`
  * :func:`torch.vander`
  * :func:`torch.zeros`
  * :func:`torch.zeros_like`

تتطلب الوظائف التالية `dtype`، `layout`، `device` كمعلمات في TorchScript،
ولكن هذه المعلمات اختيارية في بايثون.

  * :func:`torch.randint`
  * :func:`torch.sparse_coo_tensor`
  * :meth:`~torch.Tensor.to`

وحدات وفئات PyTorch غير المدعومة
---------------------------

لا يمكن لـ TorchScript حاليًا تجميع عدد من البنيات PyTorch
المستخدمة بشكل شائع. أدناه، ترد الوحدات التي لا يدعمها TorchScript،
وقائمة غير مكتملة بفئات PyTorch غير المدعومة. بالنسبة للوحدات غير المدعومة
نقترح استخدام :meth:`torch.jit.trace`.

  * :class:`torch.nn.RNN`
  * :class:`torch.nn.AdaptiveLogSoftmaxWithLoss`
  * :class:`torch.autograd.Function`
  * :class:`torch.autograd.enable_grad`