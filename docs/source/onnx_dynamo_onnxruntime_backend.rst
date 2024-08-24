ONNX Backend for TorchDynamo
============================

لنظرة عامة سريعة على ``torch.compiler``، راجع :ref:`نظرة عامة على torch.compiler <torch.compiler_overview>`.

.. warning::
  يعد ONNX backend لـ torch.compile تقنية تجريبية سريعة التطور.

.. autofunction:: torch.onnx.is_onnxrt_backend_supported

واجهة برمجة التطبيقات الخلفية ONNX لـ PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

تتيح واجهة برمجة التطبيقات الخلفية ONNX لـ PyTorch تشغيل نماذج PyTorch على أي محرك استدلال يدعم ONNX.

.. figure:: /assets/images/onnx/onnx-backend.png
   :alt: ONNX Backend for PyTorch

   الرسم التخطيطي لواجهة برمجة التطبيقات الخلفية ONNX لـ PyTorch

كما هو موضح في الرسم البياني أعلاه، تقوم واجهة برمجة التطبيقات بتحويل النموذج إلى تمثيل ONNX، ثم تستخدم وقت تشغيل ONNX لتنفيذ النموذج.

تثبيت
^^^^^

قبل استخدام واجهة برمجة التطبيقات، تأكد من تثبيت PyTorch و ONNX Runtime:

.. code:: sh

   pip install torch onnxruntime

الاستخدام الأساسي
^^^^^^^^^^^^^^^^^

يمكن استخدام واجهة برمجة التطبيقات الخلفية ONNX لتشغيل نماذج PyTorch على أي جهاز يدعم ONNX Runtime، بما في ذلك أجهزة وحدة معالجة الرسوميات (GPU) ووحدات معالجة الرسومات التخصصية (TPU).

هنا مثال بسيط يوضح كيفية استخدام واجهة برمجة التطبيقات لتشغيل نموذج تصنيف الصور:

.. code:: python

   import torch
   import torch.onnx
   from onnxruntime import InferenceSession

   # قم بتحميل نموذج PyTorch المدرب
   model = torch.load("model.pth")

   # قم بتحويل النموذج إلى تنسيق ONNX
   input_sample = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, input_sample, "model.onnx")

   # قم بتحميل نموذج ONNX في ONNX Runtime
   session = InferenceSession("model.onnx")

   # قم بتشغيل الاستدلال على صورة جديدة
   input_image = ...
   output = session.run(None, {"input": input_image})[0]

   # قم بمعالجة الإخراج حسب الحاجة
   predicted_class = output.argmax()