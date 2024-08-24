.. py:module:: torch.utils.deterministic
.. currentmodule:: torch.utils.deterministic

.. attribute:: fill_uninitialized_memory

   إن :class:`bool`، إذا كان True، يتسبب في ملء الذاكرة غير المُهيأة بقيمة معروفة عندما يتم تعيين :meth:`torch.use_deterministic_algorithms()` إلى True. يتم تعيين القيم العائمة والمعقدة إلى NaN، والقيم الصحيحة إلى القيمة القصوى.

   الافتراضي: ``True``

   إن ملء الذاكرة غير المُهيأة يضر بالأداء. لذلك، إذا كان برنامجك صالحًا ولا يستخدم الذاكرة غير المُهيأة كمدخلات لعملية ما، فيمكن إيقاف تشغيل هذا الإعداد لتحقيق أداء أفضل مع الحفاظ على الحتمية.

   ستملأ العمليات التالية الذاكرة غير المُهيأة عند تشغيل هذا الإعداد:

       * :func:`torch.Tensor.resize_` عند استدعائها مع مصفوفة غير مُكممة
       * :func:`torch.empty`
       * :func:`torch.empty_strided`
       * :func:`torch.empty_permuted`
       * :func:`torch.empty_like`