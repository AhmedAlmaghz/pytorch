هذا هو ملف التوثيق الرئيسي لـ PyTorch، تم إنشاؤه بواسطة sphinx-quickstart on Fri Dec 23 13:31:47 2016.
يمكنك تكييف هذا الملف تمامًا حسب رغبتك، ولكن يجب أن يحتوي على الأقل على توجيه الجذر `toctree`.

:github_url: https://github.com/pytorch/pytorch

توثيق PyTorch
=============

PyTorch هو مكتبة تنسور محسنة للتعلم العميق باستخدام وحدات معالجة الرسوميات (GPU) ووحدات المعالجة المركزية (CPU).

تم تصنيف الميزات الموضحة في هذه الوثيقة حسب حالة الإصدار:

  *مستقر:*  سيتم الحفاظ على هذه الميزات على المدى الطويل، ولا ينبغي أن تكون هناك قيود أداء كبيرة أو فجوات في التوثيق.
  نتوقع أيضًا الحفاظ على التوافق مع الإصدارات السابقة (على الرغم من أنه يمكن إجراء تغييرات مهمة وسيتم إعطاء إشعار قبل إصدار واحد).

  *بيتا:*  تم وضع علامة على هذه الميزات على أنها Beta لأن واجهة برمجة التطبيقات (API) قد تتغير بناءً على تعليقات المستخدمين، أو لأن الأداء يحتاج إلى تحسين، أو لأن التغطية عبر المشغلين غير مكتملة بعد. بالنسبة للميزات التجريبية، نلتزم بجعل الميزة تصل إلى التصنيف المستقر.
  ومع ذلك، فإننا لا نلتزم بالتوافق مع الإصدارات السابقة.

  *نموذج أولي:*  عادةً ما تكون هذه الميزات غير متوفرة كجزء من التوزيعات الثنائية مثل PyPI أو Conda، باستثناء وجودها أحيانًا خلف أعلام وقت التشغيل، وهي في مرحلة مبكرة لتلقي التعليقات والاختبارات.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: مجتمع

   community/*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: ملاحظات المطور

   notes/*

.. toctree::
   :maxdepth: 1
   :caption: روابط اللغة

   cpp_index
   Javadoc <https://pytorch.org/javadoc/>
   torch::deploy <deploy>

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: واجهة برمجة التطبيقات (API) بلغة بايثون

   torch
   nn
   nn.functional
   tensors
   tensor_attributes
   tensor_view
   torch.amp <amp>
   torch.autograd <autograd>
   torch.library <library>
   cpu
   cuda
   torch.cuda.memory <torch_cuda_memory>
   mps
   xpu
   mtia
   meta
   torch.backends <backends>
   torch.export <export>
   torch.distributed <distributed>
   torch.distributed.algorithms.join <distributed.algorithms.join>
   torch.distributed.elastic <distributed.elastic>
   torch.distributed.fsdp <fsdp>
   torch.distributed.optim <distributed.optim>
   torch.distributed.pipelining <distributed.pipelining>
   torch.distributed.tensor.parallel <distributed.tensor.parallel>
   torch.distributed.checkpoint <distributed.checkpoint>
   torch.distributions <distributions>
   torch.compiler <torch.compiler>
   torch.fft <fft>
   torch.func <func>
   futures
   fx
   fx.experimental
   torch.hub <hub>
   torch.jit <jit>
   torch.linalg <linalg>
   torch.monitor <monitor>
   torch.signal <signal>
   torch.special <special>
   torch.overrides
   torch.package <package>
   profiler
   nn.init
   nn.attention
   onnx
   optim
   complex_numbers
   ddp_comm_hooks
   quantization
   rpc
   torch.random <random>
   masked
   torch.nested <nested>
   size
   sparse
   storage
   torch.testing <testing>
   torch.utils <utils>
   torch.utils.benchmark <benchmark_utils>
   torch.utils.bottleneck <bottleneck>
   torch.utils.checkpoint <checkpoint>
   torch.utils.cpp_extension <cpp_extension>
   torch.utils.data <data>
   torch.utils.deterministic <deterministic>
   torch.utils.jit <jit_utils>
   torch.utils.dlpack <dlpack>
   torch.utils.mobile_optimizer <mobile_optimizer>
   torch.utils.model_zoo <model_zoo>
   torch.utils.tensorboard <tensorboard>
   torch.utils.module_tracker <module_tracker>
   type_info
   named_tensor
   name_inference
   torch.__config__ <config_mod>
   torch.__future__ <future_mod>
   logging
   torch_environment_variables

.. toctree::
   :maxdepth: 1
   :caption: المكتبات

   torchaudio <https://pytorch.org/audio/stable>
   TorchData <https://pytorch.org/data>
   TorchRec <https://pytorch.org/torchrec>
   TorchServe <https://pytorch.org/serve>
   torchtext <https://pytorch.org/text/stable>
   torchvision <https://pytorch.org/vision/stable>
   PyTorch on XLA Devices <https://pytorch.org/xla/>

الفهرس والجداول
=============

* :ref:`genindex`
* :ref:`modindex`