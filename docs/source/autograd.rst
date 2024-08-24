:hidden:`torch.autograd`
========================

حزمة التفاضل التلقائي في PyTorch. يوفر ``torch.autograd`` تفاضل تلقائي لجميع عمليات Tensor في PyTorch.

يحتوي هذا القسم على الواجهات التالية:

.. autosummary::
    :nosignatures:

    ~torch.Tensor.backward
    ~torch.autograd.backward
    ~torch.autograd.grad
    ~torch.autograd.gradcheck
    ~torch.autograd.gradgrad
    ~torch.autograd.grad_sample
    ~torch.autograd.grad_sample_with_flags
    ~torch.autograd.grad_mode
    ~torch.autograd.grad_enabled
    ~torch.autograd.set_detect_anomaly
    ~torch.autograd.detect_anomaly
    ~torch.autograd.profiler.profile
    ~torch.autograd.profiler.emit_nvtx
    ~torch.autograd.profiler.record_function
    ~torch.autograd.profiler.record_function_enter
    ~torchMultiplier
    ~torch.autograd.profiler.record_function_exit
    ~torch.autograd.Function
    ~torch.autograd.Function.apply
    ~torch.autograd.Function.backward
    ~torch.autograd.Function.buffer
    ~torch.autograd.Function.get_device
    ~torch.autograd.Function.get_grad_input
    ~torch.autograd.Function.next_functions
    ~torch.autograd.Function.register_hook
    ~torch.autograd.Function.register_forward_hook
    ~torch.autograd.Function.register_backward_hook
    ~torch.autograd.Function.register_full_backward_hook
    ~torch.autograd.anomaly_mode
    ~torch.autograd.anomaly_detected
    ~torch.autograd.grad_mode
    ~torch.autograd.set_grad_enabled
    ~torch.autograd.grad_enabled
    ~torch.autograd.no_grad
    ~torch.autograd.enable_grad
    ~torch.autograd.gradref
    ~torch.autograd.Variable
    ~torch.autograd.OnesTensor
    ~torch.autograd.ZerosTensor
    ~torch.autograd.rand
    ~torch.autograd.randn
    ~torch.autograd.randperm
    ~torch.autograd.Tensor
    ~torch.autograd.ContextObject
    ~torch.autograd.ContextAwareAutogradTensor
    ~torch.autograd.ContextAwareObjectTensor
    ~torch.autograd.ContextAwareTensor
    ~torch.autograd.FunctionMeta
    ~torch.autograd.FunctionState
    ~torch.autograd.FunctionPreprocessContext
    ~torch.autograd.FunctionPostprocessContext
    ~torch.autograd.FunctionBackwardContext
    ~torch.autograd.FunctionForwardContext
    ~torch.autograd.FunctionContextManager
    ~torch.autograd.FunctionContext
    ~torch.autograd.FunctionContextStack
    ~torch.autograd.FunctionContextStackFrame
    ~torch.autograd.FunctionContextFrame
    ~torch.autograd.FunctionFrame
    ~torch.autograd.FunctionFrameInfo
    ~torch.autograd.FunctionFrameAttribute
    ~torch.autograd.FunctionFrameAttributes
    ~torch.autograd.FunctionFrameObject
    ~torch.autograd.FunctionFrameObjects
    ~torch.autograd.FunctionFrameCode
    ~torch.autograd.FunctionFrameCodes
    ~torch.autograd.FunctionFrameLine
    ~torch.autograd.FunctionFrameLines
    ~torch.autograd.FunctionFrameVar
    ~torch.autograd.FunctionFrameVars
    ~torch.autograd.FunctionFrameLocal
    ~torch.autograd.FunctionFrameLocals
    ~torchMultiplier
    ~torch.autograd.FunctionFrameGlobal
    ~torch.autograd.FunctionFrameGlobals
    ~torch.autograd.FunctionFrameExc
    ~torch.autograd.FunctionFrameExcs
    ~torch.autograd.FunctionFrameInfoTuple
    ~torch.autograd.FunctionFrameInfoTuples
    ~torch.autograd.FunctionFrameInfoDict
    ~torch.autograd.FunctionFrameInfoDicts
    ~torch.autograd.FunctionFrameInfoObject
    ~torch.autograd.FunctionFrameInfoObjects
    ~torch.autograd.FunctionFrameInfoCode
    ~torch.autograd.FunctionFrameInfoCodes
    ~torch.autograd.FunctionFrameInfoLine
    ~torch.autograd.FunctionFrameInfoLines
    ~torch.autograd.FunctionFrameInfoVar
    ~torch.autograd.FunctionFrameInfoVars
    ~torch.autograd.FunctionFrameInfoLocal
    ~torch.autograd.FunctionFrameInfoLocals
    ~torch.autograd.FunctionFrameInfoGlobal
    ~torch.autograd.FunctionFrameInfoGlobals
    ~torch.autograd.FunctionFrameInfoExc
    ~torch.autograd.FunctionFrameInfoExcs

.. currentmodule:: torch.autograd

.. toctree::
    :maxdepth: 1

    autograd_cpp_extensions

.. _torch-autograd-function-context-manager:

سياق المدير
~~~~~~~

.. autoclass:: FunctionContextManager

.. _torch-autograd-function-context:

سياق الدالة
~~~~~~~

.. autoclass:: FunctionContext

.. _torch-autograd-function-context-stack:

كومة سياق الدالة
~~~~~~~~~~~

.. autoclass:: FunctionContextStack

.. _torch-autograd-function-context-stack-frame:

إطار كومة سياق الدالة
~~~~~~~~~~~~~~

.. autoclass:: FunctionContextStackFrame

.. _torch-autograd-function-context-frame:

إطار سياق الدالة
~~~~~~~~~~

.. autoclass:: FunctionContextFrame

.. _torch-autograd-function-frame-info:

معلومات إطار الدالة
~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfo

.. _torch-autograd-function-frame-attribute:

سمة إطار الدالة
~~~~~~~~~~

.. autoclass:: FunctionFrameAttribute

.. _torch-autograd-function-frame-attributes:

سمات إطار الدالة
~~~~~~~~~~~

.. autoclass:: FunctionFrameAttributes

.. _torch-autograd-function-frame-object:

كائن إطار الدالة
~~~~~~~~~~

.. autoclass:: FunctionFrameObject

.. _torch-autograd-function-frame-objects:

كائنات إطار الدالة
~~~~~~~~~~~

.. autoclass:: FunctionFrameObjects

.. _torch-autograd-function-frame-code:

رمز إطار الدالة
~~~~~~~~~~

.. autoclass:: FunctionFrameCode

.. _torch-autograd-function-frame-codes:

رموز إطار الدالة
~~~~~~~~~~~

.. autoclass:: FunctionFrameCodes

.. _torch-autograd-function-frame-line:

سطر إطار الدالة
~~~~~~~~~~

.. autoclass:: FunctionFrameLine

.. _torch-autograd-function-frame-lines:

أسطر إطار الدالة
~~~~~~~~~~~

.. autoclass:: FunctionFrameLines

.. _torch-autograd-function-frame-var:

متغير إطار الدالة
~~~~~~~~~~~

.. autoclass:: FunctionFrameVar

.. _torch-autograd-function-frame-vars:

متغيرات إطار الدالة
~~~~~~~~~~~~

.. autoclass:: FunctionFrameVars

.. _torch-autograd-function-frame-local:

متغير محلي لإطار الدالة
~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameLocal

.. _torch-autograd-function-frame-locals:

متغيرات محلية لإطار الدالة
~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameLocals

.. _torch-autograd-function-frame-global:

متغير عالمي لإطار الدالة
~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameGlobal

.. _torch-autograd-function-frame-globals:

متغيرات عالمية لإطار الدالة
~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameGlobals

.. _torch-autograd-function-frame-exc:

استثناء إطار الدالة
~~~~~~~~~~~

.. autoclass:: FunctionFrameExc

.. _torch-autograd-function-frame-excs:

استثناءات إطار الدالة
~~~~~~~~~~~~~

.. autoclass:: FunctionFrameExcs

.. _torch-autograd-function-frame-info-tuple:

معلومات إطار الدالة كمجموعة
~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoTuple

.. _torch-autograd-function-frame-info-tuples:

معلومات إطار الدالة كمجموعات
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoTuples

.. _torch-autograd-function-frame-info-dict:

معلومات إطار الدالة كقاموس
~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoDict

.. _torch-autograd-function-frame-info-dicts:

معلومات إطار الدالة كقاموس
~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoDicts

.. _torch-autograd-function-frame-info-object:

معلومات إطار الدالة ككائن
~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoObject

.. _torch-autograd-function-frame-info-objects:

معلومات إطار الدالة ككائنات
~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoObjects

.. _torch-autograd-function-frame-info-code:

معلومات إطار الدالة كرموز
~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoCode

.. _torch-autograd-function-frame-info-codes:

معلومات إطار الدالة كرموز
~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoCodes

.. _torch-autograd-function-frame-info-line:

معلومات إطار الدالة كسطور
~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoLine

.. _torch-autograd-function-frame-info-lines:

معلومات إطار الدالة كسطور
~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoLines

.. _torch-autograd-function-frame-info-var:

معلومات إطار الدالة كمتغيرات
~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoVar

.. _torch-autograd-function-frame-info-vars:

معلومات إطار الدالة كمتغيرات
~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoVars

.. _torch-autograd-function-frame-info-local:

معلومات إطار الدالة كمتغيرات محلية
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoLocal

.. _torch-autograd-function-frame-info-locals:

معلومات إطار الدالة كمتغيرات محلية
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoLocals

.. _torch-autograd-function-frame-info-global:

معلومات إطار الدالة كمتغيرات عالمية
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoGlobal

.. _torch-autograd-function-frame-info-globals:

معلومات إطار الدالة كمتغيرات عالمية
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoGlobals

.. _torch-autograd-function-frame-info-exc:

معلومات إطار الدالة كاستثناءات
~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoExc

.. _torch-autograd-function-frame-info-excs:

معلومات إطار الدالة كاستثناءات
~~~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionFrameInfoExcs

.. _torch-autograd-function-preprocess-context:

سياق ما قبل معالجة الدالة
~~~~~~~~~~~~~~~~

.. autoclass:: FunctionPreprocessContext

.. _torch-autograd-function-postprocess-context:

سياق ما بعد معالجة الدالة
~~~~~~~~~~~~~~~~

.. autoclass:: FunctionPostprocessContext

.. _torch-autograd-function-backward-context:

سياق الخلفي للدالة
~~~~~~~~~~~

.. autoclass:: FunctionBackwardContext

.. _torch-autograd-function-forward-context:

سياق الأمامي للدالة
~~~~~~~~~~~~

.. autoclass:: FunctionForwardContext

.. _torch-autograd-function:

دالة
~~~

.. autoclass:: Function

.. _torch-autograd-function-apply:

تطبيق الدالة
~~~~~~~

.. automethod:: Function.apply

.. _torch-autograd-function-backward:

خلفي الدالة
~~~~~~~~

.. automethod:: Function.backward

.. _torch-autograd-function-buffer:

مخزن مؤقت للدالة
~~~~~~~~~~~

.. automethod:: Function.buffer

.. _torch-autograd-function-get-device:

حصول على جهاز الدالة
~~~~~~~~~~~~~~~

.. automethod:: Function.get_device

.. _torch-autograd-function-get-grad-input:

حصول على متغير الدخل الخلفي للدالة
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: Function.get_grad_input

.. _torch-autograd-function-next-functions:

الدالة التالية
~~~~~~~

.. automethod:: Function.next_functions

.. _torch-autograd-function-register-hook:

تسجيل خطاف للدالة
~~~~~~~~~~~~

.. automethod:: Function.register_hook

.. _torch-autograd-function-register-forward-hook:

تسجيل خطاف أمامي للدالة
~~~~~~~~~~~~~~~~

.. automethod:: Function.register_forward_hook

.. _torch-autograd-function-register-backward-hook:

تسجيل خطاف خلفي للدالة
~~~~~~~~~~~~~~~~

.. automethod:: Function.register_backward_hook

.. _torch-autograd-function-register-full-backward-hook:

تسجيل خطاف خلفي كامل للدالة
~~~~~~~~~~~~~~~~~~~

.. automethod:: Function.register_full_backward_hook

.. _torch-autograd-anomaly-mode:

وضع الشذوذ
~~~~~~~~

.. autofunction:: anomaly_mode

.. _torch-autograd-anomaly-detected:

شذوذ تم اكتشافه
~~~~~~~~~~

.. autofunction:: anomaly_detected

.. _torch-autograd-grad-mode:

وضع التدرج
~~~~~~~~

.. autofunction:: grad_mode

.. _torch-autograd-set-grad-enabled:

تعيين التدرج المُفعل
~~~~~~~~~~~~

.. autofunction:: set_grad_enabled

.. _torch-autograd-grad-enabled:

التدرج المُفعل
~~~~~~~~

.. autofunction:: grad_enabled

.. _torch-autograd-no-grad:

بدون تدرج
~~~~~~~~

.. autofunction:: no_grad

.. _torch-autograd-enable-grad:

تفعيل التدرج
~~~~~~~~

.. autofunction:: enable_grad

.. _torch-autograd-gradref:

مرجع التدرج
~~~~~~~~~

.. autofunction:: gradref

.. _torch-autograd-variable:

متغير
~~~~

.. autoclass:: Variable

.. _torch-autograd-ones-tensor:

Tensor ذو أحاديات
~~~~~~~~~~~~~~

.. autoclass:: OnesTensor

.. _torch-autograd-zeros-tensor:

Tensor ذو أصفار
~~~~~~~~~~~~~~

.. autoclass:: ZerosTensor

.. _torch-autograd-rand:

rand
~~~~

.. autofunction:: rand

.. _torch-autograd-randn:

randn
~~~~~~

.. autofunction:: randn

.. _torch-autograd-randperm:

randperm
~~~~~~~~

.. autofunction:: randperm

.. _torch-autograd-tensor:

Tensor
~~~~~~~

.. autoclass:: Tensor

.. _torch-autograd-context-object:

كائن السياق
~~~~~~~~

.. autoclass:: ContextObject

.. _torch-autograd-context-aware-autograd-tensor:

Tensor السياق المُدرك للسياق
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ContextAwareAutogradTensor

.. _torch-autograd-context-aware-object-tensor:

Tensor الكائن المُدرك للسياق
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ContextAwareObjectTensor

.. _torch-autograd-context-aware-tensor:

Tensor المُدرك للسياق
~~~~~~~~~~~~~~~~

.. autoclass:: ContextAwareTensor

.. _torch-autograd-function-meta:

ميتا الدالة
~~~~~~~

.. autoclass:: FunctionMeta

.. _torch-autograd-function-state:

حالة الدالة
~~~~~~~

.. autoclass:: FunctionState

.. _torch-autograd-function-preprocess-context:

سياق ما قبل معالجة الدالة
~~~~~~~~~~~~~~~~~

.. autoclass:: FunctionPreprocessContext

.. _torch-autograd-function-postprocess-context:

سياق ما بعد معالجة الدالة
~~~~~~~~~~~~~~~~

.. autoclass:: FunctionPostprocessContext

.. _torch-autograd-function-backward-context:

سياق الخلفي للدالة
~~~~~~~~~~~

.. autoclass:: FunctionBackwardContext

.. _torch-autograd-function-forward-context:

سياق الأمامي للدالة
~~~~~~~~~~~~~~

.. autoclass:: FunctionForwardContext

.. _torch-autograd-function-context-manager:

سياق المدير
~~~~~~~~

.. autoclass:: FunctionContextManager

.. _torch-aut
======================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autosummary::
    :toctree: generated
    :nosignatures:

    backward
    grad

.. _autograd-forward:

التفاضل التلقائي للأمام
^^^^^^^^^^^^^^^

.. warning::
    هذا الـ API في مرحلة البيتا. على الرغم من أن توقيعات الدوال من غير المرجح أن تتغير، إلا أنه من المخطط تحسين تغطية المشغل قبل أن نعتبره مستقرًا.

يرجى الاطلاع على `التفاضل التلقائي للأمام <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
للحصول على الخطوات التفصيلية حول كيفية استخدام هذا الـ API.

.. autosummary::
    :toctree: generated
    :nosignatures:

    forward_ad.dual_level
    forward_ad.make_dual
    forward_ad.unpack_dual
    forward_ad.enter_dual_level
    forward_ad.exit_dual_level
    forward_ad.UnpackedDualTensor

.. _واجهة برمجة التطبيقات الوظيفية عالية المستوى:

واجهة برمجة التطبيقات الوظيفية عالية المستوى
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
    هذا الـ API في مرحلة البيتا. على الرغم من أن توقيعات الدوال من غير المرجح أن تتغير، إلا أنه من المخطط إجراء تحسينات رئيسية على الأداء قبل أن نعتبرها مستقرة.

يحتوي هذا القسم على واجهة برمجة التطبيقات عالية المستوى لـ autograd والتي تستند إلى واجهة برمجة التطبيقات الأساسية أعلاه
وتسمح لك بحساب المصفوفات الجاكوبية، والمصفوفات الهيسية، وما إلى ذلك.

تعمل واجهة برمجة التطبيقات هذه مع الدوال التي يوفرها المستخدم والتي تأخذ فقط الـ Tensors كمدخلات وتعيد
فقط الـ Tensors.
إذا كانت دالتك تأخذ وسائط أخرى غير الـ Tensors أو الـ Tensors التي لا تحتوي على "requires_grad" المحددة،
يمكنك استخدام تعبير لامدا (lambda) لالتقاطها.
على سبيل المثال، بالنسبة للدالة "f" التي تأخذ ثلاث مدخلات، Tensor الذي نريد حساب المشتق الجزئي له، وآخر
Tensor يجب اعتباره ثابتًا، وعلمًا منطقيًا كـ "f(input, constant, flag=flag)"
يمكنك استخدامها كـ "functional.jacobian(lambda x: f(x, constant, flag=flag), input)".

.. autosummary::
    :toctree: generated
    :nosignatures:

    functional.jacobian
    functional.hessian
    functional.vjp
    functional.jvp
    functional.vhp
    functional.hvp

.. _locally-disable-grad-doc:

تعطيل حساب المشتق محليًا
^^^^^^^^^^^^^^^^^

راجع :ref:`locally-disable-grad-doc` للحصول على مزيد من المعلومات حول الاختلافات
بين no-grad وinference mode بالإضافة إلى الآليات الأخرى ذات الصلة التي
قد يتم الخلط بينها وبين الاثنين. راجع أيضًا :ref:`torch-rst-local-disable-grad`
للحصول على قائمة بالدوال التي يمكن استخدامها لتعطيل المشتقات محليًا.

.. _تخطيطات المشتق الافتراضية:

تخطيطات المشتق الافتراضية
^^^^^^^^^^^^^^^^^

عندما تتلقى "param" غير المتناثرة تدرجًا غير متناثر أثناء
:func:`torch.autograd.backward` أو :func:`torch.Tensor.backward`
يتم تراكم "param.grad" على النحو التالي.

إذا كان "param.grad" في البداية "None":

1. إذا كانت ذاكرة "param" غير متداخلة وكثيفة، يتم إنشاء "grad"
   باستخدام خطوات مطابقة لـ "param" (مما يؤدي إلى مطابقة تخطيط "param").
2. وإلا، يتم إنشاء "grad" باستخدام خطوات متجاورة للصف الرئيسي.

إذا كان لدى "param" بالفعل تدرج "grad" غير متناثر:

3. إذا كانت "create_graph=False"، فإن "backward()" تتراكم في مكانها في "grad"
   مما يحافظ على خطواتها.
4. إذا كانت "create_graph=True"، فإن "backward()" تستبدل "grad" بـ "tensor" جديد "grad + new grad"
   والذي يحاول (ولكن لا يضمن) مطابقة خطوات "grad" الموجودة مسبقًا.

السلوك الافتراضي (ترك "grad" كـ "None" قبل أول
"backward()"، بحيث يتم إنشاؤها وفقًا لـ 1 أو 2،
ويتم الاحتفاظ بها بمرور الوقت وفقًا لـ 3 أو 4) يوصى به لتحقيق أفضل أداء.
لن تؤثر المكالمات إلى "model.zero_grad()" أو "optimizer.zero_grad()" على تخطيطات "grad".

في الواقع، إعادة تعيين جميع "grad" إلى "None" قبل كل
مرحلة تراكم، على سبيل المثال::

    for iterations...
        ...
        for param in model.parameters():
            param.grad = None
        loss.backward()

بحيث يتم إعادة إنشائها وفقًا لـ 1 أو 2 في كل مرة،
هو بديل صالح لـ "model.zero_grad()" أو "optimizer.zero_grad()"
الذي قد يحسن الأداء لبعض الشبكات.

تخطيطات المشتق اليدوية
----------------

إذا كنت بحاجة إلى التحكم اليدوي في خطوات "grad"،
قم بتعيين "param.grad =" tensor صفري مع الخطوات المرغوبة
قبل أول "backward()"، ولا تقم مطلقًا بإعادة تعيينه إلى "None".
تضمن 3 أن تخطيطك محفوظ طالما أن "create_graph=False".
تشير 4 إلى أن تخطيطك *من المحتمل* أن يتم الاحتفاظ به حتى إذا كانت "create_graph=True".

العمليات في المكان على الـ Tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^

إن دعم العمليات في المكان في autograd أمر صعب، ونحن لا نشجع
استخدامها في معظم الحالات. إن إعادة استخدام الـ autograd العدوانية وتحرير الـ buffer تجعلها
فعالة للغاية، وهناك مناسبات قليلة جدًا عندما تقلل العمليات في المكان
من استخدام الذاكرة بالفعل بأي كمية كبيرة. ما لم تكن تعمل تحت ضغط ذاكرة شديد، فقد لا تحتاج أبدًا إلى استخدامها.

التحقق من صحة العمليات في المكان
-----------------------

تحتفظ جميع الـ :class:`Tensor` s بتتبع للعمليات في المكان المطبقة عليها، وإذا
اكتشف التنفيذ أنه تم تطبيق عملية في المكان على tensor بعد حفظه للخلف في إحدى
الدوال، فسيتم رفع خطأ بمجرد بدء عملية الخلف. يضمن هذا أنه إذا كنت تستخدم دوال في المكان
ولا ترى أي أخطاء، فيمكنك التأكد من صحة المشتقات المحسوبة.

متغير (تم إهماله)
^^^^^^^^^^^^

.. warning::
    تم إهمال واجهة برمجة تطبيقات الـ Variable: لم تعد الـ Variables ضرورية
    لاستخدام autograd مع الـ tensors. يدعم autograd تلقائيًا الـ Tensors مع
    "requires_grad" المحدد إلى "True". فيما يلي دليل سريع لما

تغير:

    - "Variable(tensor)" و"Variable(tensor, requires_grad)" لا تزال تعمل كما هو متوقع،
      ولكنها تعيد الـ Tensors بدلاً من الـ Variables.
    - "var.data" هي نفس الشيء مثل "tensor.data".
    - تعمل الطرق مثل "var.backward()"، و"var.detach()"، و"var.register_hook()" الآن على الـ tensors
      بنفس أسماء الطرق.

    بالإضافة إلى ذلك، يمكن الآن إنشاء الـ tensors باستخدام "requires_grad=True" باستخدام طرق المصنع مثل :func:`torch.randn`، و:func:`torch.zeros`، و:func:`torch.ones`، وغيرها
    مثل ما يلي:

    ``autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)``

دوال الـ Tensor في autograd
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :nosignatures:

   torch.Tensor.grad
   torch.Tensor.requires_grad
   torch.Tensor.is_leaf
   torch.Tensor.backward
   torch.Tensor.detach
   torch.Tensor.detach_
   torch.Tensor.register_hook
   torch.Tensor.register_post_accumulate_grad_hook
   torch.Tensor.retain_grad

:hidden:`Function`
^^^^^^^^^^^^^^^^^^^

.. autoclass:: Function

.. autosummary::
    :toctree: generated
    :nosignatures:

    Function.forward
    Function.backward
    Function.jvp
    Function.vmap

.. _طرق سياق الخلط:

طرق سياق الخلط
^^^^^^^^^^^^
عند إنشاء :class:`Function` جديد، تكون الطرق التالية متاحة لـ "ctx".

.. autosummary::
    :toctree: generated
    :nosignatures:

    function.FunctionCtx.mark_dirty
    function.FunctionCtx.mark_non_differentiable
    function.FunctionCtx.save_for_backward
    function.FunctionCtx.set_materialize_grads

أدوات الـ Function المخصصة
^^^^^^^^^^^^^^^^^^^^^^
ديكور لطريقة الخلف.

.. autosummary::
    :toctree: generated
    :nosignatures:

    function.once_differentiable

قاعدة الـ Function المخصصة المستخدمة لبناء أدوات PyTorch

.. autosummary::
    :toctree: generated
    :nosignatures:

    function.BackwardCFunction
    function.InplaceFunction
    function.NestedIOFunction


.. _التحقق من المشتق العددي:

التحقق من المشتق العددي
^^^^^^^^^^^^^^^^


.. automodule:: torch.autograd.gradcheck
.. currentmodule:: torch.autograd.gradcheck

.. autosummary::
    :toctree: generated
    :nosignatures:

    gradcheck
    gradgradcheck
    GradcheckError

.. فقط لإعادة تعيين مسار الأساس لبقية هذا الملف
.. currentmodule:: torch.autograd

مُوصِّف
^^^^^

يتضمن autograd مُوصِّفًا يسمح لك بفحص تكلفة المشغلين المختلفين داخل نموذجك -
على كل من وحدة المعالجة المركزية (CPU) ووحدة معالجة الرسوميات (GPU). هناك ثلاث طرائق
مطبقة في الوقت الحالي - CPU-only باستخدام :class:`~torch.autograd.profiler.profile`.
nvprof based (تسجل نشاط كل من وحدة المعالجة المركزية ووحدة معالجة الرسوميات) باستخدام
:class:`~torch.autograd.profiler.emit_nvtx`.
وvtune profiler based باستخدام
:class:`~torch.autograd.profiler.emit_itt`.

.. autoclass:: torch.autograd.profiler.profile

.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.profile.export_chrome_trace
    profiler.profile.key_averages
    profiler.profile.self_cpu_time_total
    profiler.profile.total_average
    profiler.parse_nvprof_trace
    profiler.EnforceUnique
    profiler.KinetoStepTracker
    profiler.record_function
    profiler_util.Interval
    profiler_util.Kernel
    profiler_util.MemRecordsAcc
    profiler_util.StringTable

.. autoclass:: torch.autograd.profiler.emit_nvtx
.. autoclass:: torch.autograd.profiler.emit_itt


.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.load_nvprof

تصحيح الأخطاء والكشف عن الشذوذ
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: detect_anomaly

.. autoclass:: set_detect_anomaly

.. autosummary::
    :toctree: generated
    :nosignatures:

    grad_mode.set_multithreading_enabled



مخطط autograd
^^^^^^^^^^^^^^
يكشف autograd الطرق التي تسمح بفحص المخطط والتدخل في السلوك أثناء
مرور الخلف.

يحتوي "grad_fn" الخاص بـ :class:`torch.Tensor` على :class:`torch.autograd.graph.Node`
إذا كان الـ tensor هو ناتج عملية تم تسجيلها بواسطة autograd (أي تم تمكين grad_mode
وكان واحدًا على الأقل من المدخلات يتطلب المشتقات)، أو "None" في حالة أخرى.

.. autosummary::
    :toctree: generated
    :nosignatures:

    graph.Node.name
    graph.Node.metadata
    graph.Node.next_functions
    graph.Node.register_hook
    graph.Node.register_prehook
    graph.increment_version

تحتاج بعض العمليات إلى نتائج وسيطة ليتم حفظها أثناء المرور للأمام
من أجل تنفيذ مرور الخلف.
يتم حفظ هذه النتائج الوسيطة كسمات على "grad_fn" ويمكن الوصول إليها.
على سبيل المثال::

    >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
    >>> b = a.exp()
    >>> print(isinstance(b.grad_fn, torch.autograd.graph.Node))
    True
    >>> print(dir(b.grad_fn))
    ['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_raw_saved_result', '_register_hook_dict', '_saved_result', 'metadata', 'name', 'next_functions', 'register_hook', 'register_prehook', 'requires_grad']
    >>> print(torch.allclose(b.grad_fn._saved_result, b))
    True

يمكنك أيضًا تحديد كيفية تعبئة / فك تعبئة هذه الـ tensors المحفوظة باستخدام الـ hooks.
تتمثل إحدى التطبيقات الشائعة في التداول بين الحوسبة والذاكرة عن طريق حفظ هذه النتائج الوسيطة
على القرص أو على وحدة المعالجة المركزية بدلاً من تركها على وحدة معالجة الرسوميات. هذا مفيد بشكل خاص إذا لاحظت أن نموذجك يناسب وحدة معالجة الرسوميات أثناء التقييم، ولكن ليس أثناء التدريب.
راجع أيضًا :ref:`saved-tensors-hooks-doc`.

.. autoclass:: torch.autograd.graph.saved_tensors_hooks

.. autoclass:: torch.autograd.graph.save_on_cpu

.. autoclass:: torch.autograd.graph.disable_saved_tensors_hooks

.. autoclass:: torch.autograd.graph.register_multi_grad_hook

.. autoclass:: torch.autograd.graph.allow_mutation_on_saved_tensors

.. autoclass:: torch.autograd.graph.GradientEdge

.. autofunction:: torch.autograd.graph.get_gradient_edge



.. تحتاج هذه الوحدة إلى توثيق. إضافتها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.autograd.anomaly_mode
.. py:module:: torch.autograd.forward_ad
.. py:module:: torch.autograd.function
.. py:module:: torch.autograd.functional
.. py:module:: torch.autograd.grad_mode
.. py:module:: torch.autograd.graph
.. py:module:: torch.autograd.profiler
.. py:module:: torch.autograd.profiler_legacy
.. py:module:: torch.autograd.profiler_util
.. py:module:: torch.autograd.variable