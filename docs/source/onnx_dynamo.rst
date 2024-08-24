مُصدِّر ONNX المستند إلى TorchDynamo
================================

.. automodule:: torch.onnx
  :noindex:

.. contents:: :local:
    :depth: 3

.. warning::
  مُصدِّر ONNX لـ TorchDynamo هو تقنية تجريبية سريعة التطور.

نظرة عامة
--------

يستفيد مُصدِّر ONNX من محرك TorchDynamo للربط مع واجهة برمجة تطبيقات تقييم الإطارات في بايثون
وإعادة كتابة بايت كود الخاص بها ديناميكيًا إلى رسم بياني لـ FX.
يتم بعد ذلك تحسين الرسم البياني لـ FX قبل ترجمته في النهاية إلى رسم بياني لـ ONNX.

الميزة الرئيسية لهذا النهج هي أن `رسم FX <https://pytorch.org/docs/stable/fx.html>`_ يتم التقاطه باستخدام
تحليل بايت كود الذي يحافظ على الطبيعة الديناميكية للنموذج بدلاً من استخدام تقنيات التتبع الثابتة التقليدية.

تم تصميم المُصدِّر ليكون نمطيًا وقابلاً للتوسعة. وهو يتكون من المكونات التالية:

  - **مُصدِّر ONNX**: :class:`Exporter` الفئة الرئيسية التي تقوم بتنسيق عملية التصدير.
  - **خيارات تصدير ONNX**: :class:`ExportOptions` لديه مجموعة من الخيارات التي تتحكم في عملية التصدير.
  - **سجل ONNX**: :class:`OnnxRegistry` هو سجل مشغلات ONNX والوظائف.
  - **مستخرج رسم FX**: :class:`FXGraphExtractor` يستخرج رسم FX من نموذج PyTorch.
  - **وضع التزييف**: :class:`ONNXFakeContext` هو مدير سياق يسمح بوضع التزييف للنماذج واسعة النطاق.
  - **برنامج ONNX**: :class:`ONNXProgram` هو ناتج المُصدِّر الذي يحتوي على الرسم البياني لـ ONNX المصدَّر والتشخيصات.
  - **مُسلسل برنامج ONNX**: :class:`ONNXProgramSerializer` يقوم بتهيئة النموذج المصدَّر إلى ملف.
  - **خيارات التشخيص ONNX**: :class:`DiagnosticOptions` لديه مجموعة من الخيارات التي تتحكم في التشخيصات التي يصدرها المُصدِّر.

التبعيات
------

يعتمد مُصدِّر ONNX على حزم بايثون الإضافية:

  - `ONNX <https://onnx.ai>`_
  - `ONNX Script <https://onnxscript.ai>`_

يمكن تثبيتها من خلال `pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

  pip install --upgrade onnx onnxscript

مثال بسيط
--------

انظر أدناه توضيحًا لواجهة برمجة تطبيقات المُصدِّر في العمل مع شبكة عصبية متعددة الطبقات كمثال بسيط:

.. code-block:: python

  import torch
  import torch.nn as nn

  class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(8, 8, bias=True)
        self.fc1 = nn.Linear(8, 4, bias=True)
        self.fc2 = nn.Linear(4, 2, bias=True)
        self.fc3 = nn.Linear(2, 2, bias=True)

    def forward(self, tensor_x: torch.Tensor):
        tensor_x = self.fc0(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        tensor_x = self.fc1(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        tensor_x = self.fc2(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        output = self.fc3(tensor_x)
        return output

  model = MLPModel()
  tensor_x = torch.rand((97, 8), dtype=torch.float32)
  onnx_program = torch.onnx.dynamo_export(model, tensor_x)

كما يوضح الكود أعلاه، كل ما تحتاجه هو تزويد :func:`torch.onnx.dynamo_export` بمثيل للنموذج وإدخاله.
بعد ذلك، سيعيد المُصدِّر مثيلًا لـ :class:`torch.onnx.ONNXProgram` يحتوي على الرسم البياني لـ ONNX المصدَّر بالإضافة إلى معلومات إضافية.

يمكن الوصول إلى النموذج الموجود في الذاكرة من خلال ``onnx_program.model_proto`` وهو كائن ``onnx.ModelProto`` متوافق مع `مواصفات ONNX IR <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.
بعد ذلك، يمكن تهيئة نموذج ONNX إلى ملف `Protobuf <https://protobuf.dev/>`_ باستخدام واجهة برمجة التطبيقات :meth:`torch.onnx.ONNXProgram.save`.

.. code-block:: python

  onnx_program.save("mlp.onnx")

فحص نموذج ONNX باستخدام واجهة المستخدم الرسومية
-----------------------------------

يمكنك عرض النموذج المصدَّر باستخدام `Netron <https://netron.app/>`__.

.. image:: _static/img/onnx/onnx_dynamo_mlp_model.png
    :width: 40%
    :alt: نموذج MLP كما هو موضح باستخدام Netron

لاحظ أن كل طبقة ممثلة في مربع مستطيل مع أيقونة "f" في الزاوية اليمنى العليا.

.. image:: _static/img/onnx/onnx_dynamo_mlp_model_function_highlight.png
    :width: 40%
    :alt: وظيفة ONNX مميزة في نموذج MLP

عن طريق توسيعه، يتم عرض جسم الوظيفة.

.. image:: _static/img/onnx/onnx_dynamo_mlp_model_function_body.png
    :width: 50%
    :alt: جسم وظيفة ONNX

جسم الوظيفة هو تسلسل لمشغلات ONNX أو وظائف أخرى.

تشخيص المشكلات باستخدام SARIF
----------------------------

تتجاوز تشخيصات ONNX السجلات العادية من خلال اعتماد
`تنسيق نتائج التحليل الثابت (المعروف باسم SARIF) <https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>`__
لمساعدة المستخدمين في تصحيح أخطاء نموذجهم وتحسينه باستخدام واجهة المستخدم الرسومية، مثل
`عارض SARIF <https://marketplace.visualstudio.com/items?itemName=MS-SarifVSCode.sarif-viewer>`_ في Visual Studio Code.

المزايا الرئيسية هي:

  - يتم إصدار التشخيصات بتنسيق قابل للتحليل الآلي `تنسيق نتائج التحليل الثابت (SARIF) <https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>`__.
  - طريقة جديدة أوضح وأكثر تنظيماً لإضافة قواعد تشخيصية جديدة وتتبعها.
  - تعمل كقاعدة لمزيد من التحسينات المستقبلية التي تستخدم التشخيصات.

.. toctree::
   :maxdepth: 1
   :caption: قواعد SARIF لتشخيص ONNX
   :glob:

   generated/onnx_dynamo_diagnostics_rules/*

مرجع واجهة برمجة التطبيقات
-------------

.. autofunction:: torch.onnx.dynamo_export

.. autoclass:: torch.onnx.ExportOptions
    :members:

.. autofunction:: torch.onnx.enable_fake_mode

.. autoclass:: torch.onnx.ONNXProgram
    :members:

.. autoclass:: torch.onnx.ONNXProgramSerializer
    :members:

.. autoclass:: torch.onnx.ONNXRuntimeOptions
    :members:

.. autoclass:: torch.onnx.InvalidExportOptionsError
    :members:

.. autoclass:: torch.onnx.OnnxExporterError
    :members:

.. autoclass:: torch.onnx.OnnxRegistry
    :members:

.. autoclass:: torch.onnx.DiagnosticOptions
    :members: