.. warning::
   هناك مشكلات عدم حتمية معروفة لوظائف RNN على بعض إصدارات cuDNN وCUDA. يمكنك فرض سلوك حتمي من خلال تعيين متغيرات البيئة التالية:

   على CUDA 10.1، قم بتعيين متغير البيئة ``CUDA_LAUNCH_BLOCKING=1``.
   قد يؤثر هذا على الأداء.

   على CUDA 10.2 أو أحدث، قم بتعيين متغير البيئة
   (لاحظ رمز الاستعمار المبدئي)
   ``CUBLAS_WORKSPACE_CONFIG=:16:8``
   أو
   ``CUBLAS_WORKSPACE_CONFIG=:4096:2``.

   لمزيد من المعلومات، راجع `ملاحظات إصدار cuDNN 8`_.

.. _ملاحظات إصدار cuDNN 8: https://docs.nvidia.com/deeplearning/sdk/cudnn-release-notes/rel_8.html