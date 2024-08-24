.. _miscellaneous_environment_variables:

متغيرات بيئية متنوعة
=================
.. list-table::
  :header-rows: 1

  * - المتغير
    - الوصف
  * - ``TORCH_FORCE_WEIGHTS_ONLY_LOAD``
    - إذا تم تعيينه على [``1``، ``y``، ``yes``، ``true``]، فسيستخدم ``torch.load`` القيمة ``weight_only=True``. لمزيد من التوثيق حول هذا الموضوع، راجع :func:`torch.load`.
  * - ``TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT``
    - في بعض الحالات، يمكن أن تعلق خيوط autograd عند الإغلاق، لذلك لا ننتظر إغلاقها إلى أجل غير مسمى ولكن نعتمد على مهلة يتم تعيينها افتراضيًا إلى ``10`` ثواني. يمكن استخدام متغير البيئة هذا لتعيين المهلة بالثواني.
  * - ``TORCH_DEVICE_BACKEND_AUTOLOAD``
    - إذا تم تعيينه على ``1``، فسيتم استيراد ملحقات backend خارج الشجرة تلقائيًا عند تشغيل ``import torch``.