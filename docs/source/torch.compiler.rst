.. _torch.compiler_overview:

torch.compiler
==============

``torch.compiler`` عبارة عن مساحة أسماء يتم من خلالها عرض بعض طرق المترجم الداخلية للاستهلاك من قبل المستخدم. الوظيفة الرئيسية والميزة في هذه المساحة هي ``torch.compile``.

``torch.compile`` هي دالة PyTorch تم تقديمها في PyTorch 2.x تهدف إلى حل مشكلة التقاط الرسوم الدقيقة في PyTorch وفي النهاية تمكين مهندسي البرمجيات من تشغيل برامج PyTorch بشكل أسرع. ``torch.compile`` مكتوبة بلغة Python وهي تمثل انتقال PyTorch من C++ إلى Python.

يستفيد ``torch.compile`` من التقنيات الأساسية التالية:

* **TorchDynamo (torch._dynamo)** عبارة عن واجهة برمجة تطبيقات (API) داخلية تستخدم ميزة في CPython تسمى واجهة برمجة تطبيقات تقييم الإطار (Frame Evaluation API) لالتقاط رسومات PyTorch بشكل آمن. يتم عرض الطرق المتاحة خارجيًا لمستخدمي PyTorch من خلال مساحة أسماء ``torch.compiler``.

* **TorchInductor** هو المترجم الافتراضي لـ ``torch.compile`` deep learning الذي يقوم بتوليد كود سريع لمسرعات وخلفيات متعددة. تحتاج إلى استخدام مترجم خلفي لتحقيق تسريع من خلال ``torch.compile``. بالنسبة لـ NVIDIA و AMD و Intel GPUs، فإنه يستفيد من OpenAI Triton باعتباره العنصر الأساسي.

* **AOT Autograd** لا يلتقط فقط التعليمات البرمجية على مستوى المستخدم، ولكن أيضًا backpropagation، مما يؤدي إلى التقاط تمرير الخلفيات "مسبقًا". يمكّن ذلك من تسريع كل من التمرير للأمام والخلف باستخدام TorchInductor.

.. note:: في بعض الحالات، قد يتم استخدام مصطلحات ``torch.compile`` و TorchDynamo و ``torch.compiler`` بشكل متبادل في هذه الوثائق.

كما ذكرنا سابقًا، لتشغيل سير عملك بشكل أسرع، يتطلب ``torch.compile`` من خلال TorchDynamo وجود خلفية تقوم بتحويل الرسوم البيانية التي تم التقاطها إلى كود آلة سريع. يمكن أن تؤدي الخلفيات المختلفة إلى مكاسب تحسين مختلفة. والخلفية الافتراضية تسمى TorchInductor، والمعروفة أيضًا باسم *inductor*، لدى TorchDynamo قائمة من الخلفيات المدعومة التي طورها شركاؤنا، والتي يمكن الاطلاع عليها عن طريق تشغيل ``torch.compiler.list_backends()`` لكل منها مع تبعياتها الاختيارية.

بعض الخلفيات الأكثر استخدامًا تشمل:

**خلفيات التدريب والاستدلال**

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - الخلفية
     - الوصف
   * - ``torch.compile(m, backend="inductor")``
     - يستخدم خلفية TorchInductor. `اقرأ المزيد <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
   * - ``torch.compile(m, backend="cudagraphs")``
     - رسومات CUDA مع AOT Autograd. `اقرأ المزيد <https://github.com/pytorch/torchdynamo/pull/757>`__
   * - ``torch.compile(m, backend="ipex")``
     - يستخدم IPEX على وحدة المعالجة المركزية. `اقرأ المزيد <https://github.com/intel/intel-extension-for-pytorch>`__
   * - ``torch.compile(m, backend="onnxrt")``
     - يستخدم وقت تشغيل ONNX للتدريب على وحدة المعالجة المركزية/GPU. :doc:`اقرأ المزيد <onnx_dynamo_onnxruntime_backend>`

**خلفيات الاستدلال فقط**

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - الخلفية
     - الوصف
   * - ``torch.compile(m, backend="tensorrt")``
     - يستخدم Torch-TensorRT لتحسينات الاستدلال. يتطلب "استيراد torch_tensorrt" في النص البرمجي المستدعي لتسجيل الخلفية. `اقرأ المزيد <https://github.com/pytorch/TensorRT>`__
   * - ``torch.compile(m, backend="ipex")``
     - يستخدم IPEX للاستدلال على وحدة المعالجة المركزية. `اقرأ المزيد <https://github.com/intel/intel-extension-for-pytorch>`__
   * - ``torch.compile(m, backend="tvm")``
     - يستخدم Apache TVM لتحسينات الاستدلال. `اقرأ المزيد <https://tvm.apache.org/>`__
   * - ``torch.compile(m, backend="openvino")``
     - يستخدم OpenVINO لتحسينات الاستدلال. `اقرأ المزيد <https://docs.openvino.ai/torchcompile>`__

اقرأ المزيد
~~~~~~~~~

.. toctree::
   :caption: البدء لمستخدمي PyTorch
   :maxdepth: 1

   torch.compiler_get_started
   torch.compiler_api
   torch.compiler_fine_grain_apis
   torch.compiler_aot_inductor
   torch.compiler_inductor_profiling
   torch.compiler_profiling_torch_compile
   torch.compiler_faq
   torch.compiler_troubleshooting
   torch.compiler_performance_dashboard

..
  _إذا كنت ترغب في المساهمة بموضوع على مستوى المطور
   الذي يوفر نظرة عامة متعمقة على ميزة torch._dynamo،
   أضف في جدول المحتويات أدناه.

.. toctree::
   :caption: نظرة متعمقة لمطوري PyTorch
   :maxdepth: 1

   torch.compiler_dynamo_overview
   torch.compiler_dynamo_deepdive
   torch.compiler_dynamic_shapes
   torch.compiler_nn_module
   torch.compiler_best_practices_for_backends
   torch.compiler_cudagraph_trees
   torch.compiler_fake_tensor

.. toctree::
   :caption: كيفية الاستخدام لموردي خلفيات PyTorch
   :maxdepth: 1

   torch.compiler_custom_backends
   torch.compiler_transformations
   torch.compiler_ir