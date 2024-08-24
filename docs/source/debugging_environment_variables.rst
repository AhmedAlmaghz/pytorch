.. _debugging_environment_variables:

تصحيح متغيرات البيئة
===============
.. list-table::
  :header-rows: 1

  * - المتغير
    - الوصف
  * - ``TORCH_SHOW_CPP_STACKTRACES``
    - عند تعيينه إلى ``1``، يجعل PyTorch يطبع أثر المكدس عند اكتشاف خطأ في C++.
  * - ``TORCH_CPP_LOG_LEVEL``
    - يحدد مستوى السجل لمرفق تسجيل c10 (يدعم كل من GLOG ومسجلات c10). القيم الصالحة هي ``INFO``، ``WARNING``، ``ERROR``، و ``FATAL`` أو ما يعادلها الرقمي ``0``، ``1``، ``2``، و ``3``.
  * - ``TORCH_LOGS``
    -  للحصول على شرح أكثر تفصيلاً لهذا المتغير البيئي، راجع :doc:`/logging`.