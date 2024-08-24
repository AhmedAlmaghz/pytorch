.. _threading_environment_variables:

متغيرات بيئة الخيوط
==================
.. list-table::
  :header-rows: 1

  * - المتغير
    - الوصف
  * - ``OMP_NUM_THREADS``
    - يحدد العدد الأقصى للخيوط التي سيتم استخدامها في المناطق الموازية OpenMP.
  * - ``MKL_NUM_THREADS``
    - يحدد العدد الأقصى للخيوط التي سيتم استخدامها في مكتبة Intel MKL. لاحظ أن لـ MKL_NUM_THREADS الأسبقية على ``OMP_NUM_THREADS``.