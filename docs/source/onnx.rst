torch.onnx
==========

نظرة عامة
--------

`Open Neural Network eXchange (ONNX) <https://onnx.ai/>`_ هو تنسيق قياسي مفتوح لتمثيل نماذج التعلم الآلي. وتقوم وحدة ``torch.onnx`` بالتقاط مخطط الحساب من نموذج :class:`torch.nn.Module` الأصلي في PyTorch وتحويله إلى
`مخطط ONNX <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.

يمكن استخدام النموذج المصدر من قبل أي من العديد من
`runtimes التي تدعم ONNX <https://onnx.ai/supported-tools.html#deployModel>`_، بما في ذلك
`ONNX Runtime <https://www.onnxruntime.ai>`_ من مايكروسوفت.

**هناك نوعان من واجهة برمجة تطبيقات ONNX المصدرة التي يمكنك استخدامها، كما هو مدرج أدناه:**

مصدر ONNX المستند إلى TorchDynamo
-------------------------------

*مصدر ONNX المستند إلى TorchDynamo هو أحدث مصدر (Beta) لـ PyTorch 2.1 والإصدارات الأحدث*

يتم الاستفادة من محرك TorchDynamo للربط مع واجهة برمجة تطبيقات تقييم الإطارات في Python وإعادة كتابة بايتكود الخاص به ديناميكيًا إلى مخطط FX. ثم يتم صقل مخطط FX الناتج قبل ترجمته في النهاية إلى مخطط ONNX.

الميزة الرئيسية لهذا النهج هي أن مخطط `FX <https://pytorch.org/docs/stable/fx.html>`_ يتم التقاطه باستخدام
تحليل بايتكود الذي يحافظ على الطبيعة الديناميكية للنموذج بدلاً من استخدام تقنيات التتبع الثابتة التقليدية.

:doc:`تعرف أكثر على مصدر ONNX المستند إلى TorchDynamo <onnx_dynamo>`

مصدر ONNX المستند إلى TorchScript
-------------------------------

*مصدر ONNX المستند إلى TorchScript متاح منذ PyTorch 1.2.0*

يتم الاستفادة من `TorchScript <https://pytorch.org/docs/stable/jit.html>`_ لتتبع (من خلال :func:`torch.jit.trace`)
النموذج والتقاط مخطط حساب ثابت.

ونتيجة لذلك، فإن المخطط الناتج لديه بعض القيود:

* لا يسجل أي تدفق تحكم، مثل جمل if أو الحلقات؛
* لا يتعامل مع الفروق الدقيقة بين أوضاع ``التدريب`` و ``التقييم``؛
* لا يتعامل حقًا مع المدخلات الديناميكية

في محاولة لدعم قيود التتبع الثابت، يدعم المصدر أيضًا كتابة TorchScript
(من خلال :func:`torch.jit.script`)، والتي تضيف دعمًا لتدفق التحكم المعتمد على البيانات، على سبيل المثال. ومع ذلك، فإن TorchScript
نفسه هو مجموعة فرعية من لغة Python، لذلك لا يتم دعم جميع الميزات في Python، مثل العمليات في المكان.

:doc:`تعرف أكثر على مصدر ONNX المستند إلى TorchScript <onnx_torchscript>`

المساهمة / التطوير
----------------

مصدر ONNX هو مشروع مجتمعي ونحن نرحب بالمساهمات. نتبع
`إرشادات PyTorch للمساهمات <https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md>`_، ولكن قد تكون مهتمًا أيضًا بقراءة `ويكي التطوير <https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter>`_ الخاص بنا.

.. toctree::
    :hidden:

    onnx_dynamo
    onnx_dynamo_onnxruntime_backend
    onnx_torchscript

.. تحتاج هذه الوحدة إلى توثيق. أضيفها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.onnx.errors
.. py:module:: torch.onnx.operators
.. py:module:: torch.onnx.symbolic_caffe2
.. py:module:: torch.onnx.symbolic_helper
.. py:module:: torch.onnx.symbolic_opset10
.. py:module:: torch.onnx.symbolic_opset11
.. py:module:: torch.onnx.symbolic_opset13
.. py:module:: torch.onnx.symbolic_opset14
.. py:module:: torch.onnx.symbolic_opset15
.. py:module:: torch.onnx.symbolic_opset16
.. py:module:: torch.onnx.symbolic_opset17
.. py:module:: torch.onnx.symbolic_opset18
.. py:module:: torch.onnx.symbolic_opset19
.. py:module:: torch.onnx.symbolic_opset20
.. py:module:: torch.onnx.symbolic_opset7
.. py:module:: torch.onnx.symbolic_opset8
.. py:module:: torch.onnx.symbolic_opset9
.. py:module:: torch.onnx.utils
.. py:module:: torch.onnx.verification
.. py:module:: torch.onnx.symbolic_opset12