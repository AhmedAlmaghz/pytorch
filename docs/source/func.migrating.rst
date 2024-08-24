# الانتقال من functorch إلى torch.func

torch.func، الذي كان يُعرف سابقًا باسم "functorch"، هو تحويلات وظائف قابلة للتكوين على غرار `JAX <https://github.com/google/jax>`_ لـ PyTorch.

بدأ functorch كمكتبة خارجية في مستودع `pytorch/functorch <https://github.com/pytorch/functorch>`_. كان هدفنا دائمًا هو دمج functorch مباشرة في PyTorch وتوفيره كمكتبة أساسية لـ PyTorch.

وكخطوة أخيرة في عملية الدمج، قررنا الانتقال من كونه حزمة من المستوى الأعلى ("functorch") إلى كونه جزءًا من PyTorch ليعكس كيف يتم دمج تحويلات الوظائف مباشرة في جوهر PyTorch. بدءًا من PyTorch 2.0، سنوقف استخدام "import functorch" ونطلب من المستخدمين الانتقال إلى واجهات برمجة التطبيقات (APIs) الأحدث، والتي سنحافظ عليها في المستقبل. سيتم الاحتفاظ بـ "import functorch" للحفاظ على التوافق مع الإصدارات السابقة لبضع إصدارات.

تحويلات الوظيفة
--------------

واجهات برمجة التطبيقات التالية هي بديل مباشر لواجهات برمجة تطبيقات functorch التالية. وهي متوافقة تمامًا مع الإصدارات السابقة.

============================== =======================================
واجهة برمجة تطبيقات functorch         واجهة برمجة تطبيقات PyTorch (اعتبارًا من PyTorch 2.0)
============================== =======================================
functorch.vmap                  :func:`torch.vmap` أو :func:`torch.func.vmap`
functorch.grad                  :func:`torch.func.grad`
functorch.vjp                   :func:`torch.func.vjp`
functorch.jvp                   :func:`torch.func.jvp`
functorch.jacrev                :func:`torch.func.jacrev`
functorch.jacfwd                :func:`torch.func.jacfwd`
functorch.hessian               :func:`torch.func.hessian`
functorch.functionalize         :func:`torch.func.functionalize`
============================== =======================================

علاوة على ذلك، إذا كنت تستخدم واجهات برمجة تطبيقات torch.autograd.functional، فيرجى تجربة ما يعادلها في :mod:`torch.func` بدلاً من ذلك. تعد تحويلات الوظائف في :mod:`torch.func` أكثر قابلية للتكوين وأكثر كفاءة في الأداء في العديد من الحالات.

=========================================== =======================================
واجهة برمجة تطبيقات torch.autograd.functional               واجهة برمجة تطبيقات torch.func (اعتبارًا من PyTorch 2.0)
=========================================== =======================================
:func:`torch.autograd.functional.vjp`       :func:`torch.func.grad` أو :func:`torch.func.vjp`
:func:`torch.autograd.functional.jvp`       :func:`torch.func.jvp`
:func:`torch.autograd.functional.jacobian`  :func:`torch.func.jacrev` أو :func:`torch.func.jacfwd`
:func:`torch.autograd.functional.hessian`   :func:`torch.func.hessian`
=========================================== =======================================

مرافق وحدات NN
----------------

لقد قمنا بتغيير واجهات برمجة التطبيقات لتطبيق تحويلات الوظائف على وحدات NN لجعلها تتوافق بشكل أفضل مع فلسفة تصميم PyTorch. تختلف واجهة برمجة التطبيقات الجديدة، لذا يرجى قراءة هذا القسم بعناية.

functorch.make_functional
^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.func.functional_call` هو البديل لـ `functorch.make_functional <https://pytorch.org/functorch/1.13/generated/functorch.make_functional.html#functorch.make_functional>`_ و `functorch.make_functional_with_buffers <https://pytorch.org/functorch/1.13/generated/functorch.make_functional_with_buffers.html#functorch.make_functional_with_buffers>`_. ومع ذلك، فهو ليس بديلاً مباشرًا.

إذا كنت في عجلة من أمرك، فيمكنك استخدام `وظائف المساعدة في هذا المقتطف <https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf>`_ التي تحاكي سلوك functorch.make_functional و functorch.make_functional_with_buffers. نوصي باستخدام :func:`torch.func.functional_call` مباشرة لأنه واجهة برمجة تطبيقات أكثر صراحة ومرونة.

بشكل ملموس، يعيد functorch.make_functional وحدة وظيفية ومعلمات. تقبل الوحدة الوظيفية المعلمات والمدخلات إلى النموذج كحجج. يسمح :func:`torch.func.functional_call` بالاستدعاء إلى الأمام للوحدة النمطية الموجودة باستخدام المعلمات والذاكرات والمدخلات الجديدة.

فيما يلي مثال على كيفية حساب تدرجات معلمات نموذج باستخدام functorch مقابل :mod:`torch.func`::

    # ---------------
    # باستخدام functorch
    # ---------------
    استيراد الشعلة
    استيراد functorch
    المدخلات = الشعلة.randn(64، 3)
    الأهداف = الشعلة.randn(64، 3)
    النموذج = الشعلة.nn.Linear (3، 3)

    fmodel، params = functorch.make_functional (النموذج)

    def compute_loss (params، inputs، targets):
        التنبؤ = fmodel (params، inputs)
        return torch.nn.functional.mse_loss (التنبؤ، الأهداف)

    grads = functorch.grad (compute_loss) (params، inputs، الأهداف)

    # ------------------------------------
    # باستخدام torch.func (اعتبارًا من PyTorch 2.0)
    # ------------------------------------
    استيراد الشعلة
    المدخلات = الشعلة.randn(64، 3)
    الأهداف = الشعلة.randn(64، 3)
    النموذج = الشعلة.nn.Linear (3، 3)

    المعلمات = ديكت (النموذج.named_parameters ())

    def compute_loss (params، inputs، targets):
        التنبؤ = torch.func.functional_call (النموذج، params، (inputs،))
        return torch.nn.functional.mse_loss (التنبؤ، الأهداف)

    grads = torch.func.grad (compute_loss) (params، inputs، الأهداف)

وهنا مثال على كيفية حساب المشتقات الجزئية لمعلمات النموذج::

    # ---------------
    # باستخدام functorch
    # ---------------
    استيراد الشعلة
    استيراد functorch
    المدخلات = الشعلة.randn(64، 3)
    النموذج = الشعلة.nn.Linear (3، 3)

    fmodel، params = functorch.make_functional (النموذج)
    المشتقات الجزئية = functorch.jacrev (fmodel) (params، inputs)

    # ------------------------------------
    # باستخدام torch.func (اعتبارًا من PyTorch 2.0)
    # ------------------------------------
    استيراد الشعلة
    من الشعلة. استيراد func jacrev، functional_call

    المدخلات = الشعلة.randn(64، 3)
    النموذج = الشعلة.nn.Linear (3، 3)

    المعلمات = ديكت (النموذج.named_parameters ())
    # تحسب jacrev المشتقات الجزئية للargnums=0 بشكل افتراضي.
    # نحدده على 1 لحساب المشتقات الجزئية للمعلمات
    المشتقات الجزئية = jacrev (functional_call، argnums=1) (النموذج، المعلمات، (المدخلات،))

لاحظ أنه من المهم لاستهلاك الذاكرة ألا تحمل سوى نسخة واحدة من معلماتك. لا يقوم "model.named_parameters()" بنسخ المعلمات. إذا قمت في تدريب النموذج الخاص بك بتحديث معلمات النموذج في المكان، فإن "nn.Module" الذي هو نموذجك يحتوي على نسخة واحدة من المعلمات وكل شيء على ما يرام.

ومع ذلك، إذا كنت تريد حمل معلماتك في قاموس وتحديثها خارج المكان، فهناك نسختان من المعلمات: واحدة في القاموس وواحدة في "النموذج". في هذه الحالة، يجب عليك تغيير "النموذج" لعدم الاحتفاظ بالذاكرة عن طريق تحويله إلى الجهاز الميتا عبر "النموذج.to('meta')".

functorch.combine_state_for_ensemble
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يرجى استخدام :func:`torch.func.stack_module_state` بدلاً من `functorch.combine_state_for_ensemble <https://pytorch.org/functorch/1.13/generated/functorch.combine_state_for_ensemble.html>`_ :func:`torch.func.stack_module_state` يعيد قاموسين، أحدهما للمعلمات المكدسة، والآخر للذاكرات المكدسة، والتي يمكن استخدامها بعد ذلك مع :func:`torch.vmap` و :func:`torch.func.functional_call` للتصويت.

على سبيل المثال، فيما يلي مثال على كيفية التصويت على نموذج بسيط جدًا::

    استيراد الشعلة
    عدد النماذج = 5
    حجم الدفعة = 64
    في_الميزات، out_features = 3، 3
    النماذج = [الشعلة.nn.Linear (في_الميزات، out_features) ل i في نطاق (عدد النماذج)]
    البيانات = الشعلة.randn (حجم الدفعة، 3)

    # ---------------
    # باستخدام functorch
    # ---------------
    استيراد functorch
    fmodel، params، buffers = functorch.combine_state_for_ensemble (النماذج)
    الإخراج = functorch.vmap (fmodel، (0، 0، لا شيء)) (params، buffers، data)
    التأكيد على أن الإخراج. الشكل == (عدد النماذج، حجم الدفعة، out_features)

    # ------------------------------------
    # باستخدام torch.func (اعتبارًا من PyTorch 2.0)
    # ------------------------------------
    استيراد النسخ

    # قم ببناء إصدار من النموذج بدون ذاكرة عن طريق وضع Tensors على
    # جهاز الميتا.
    النموذج الأساسي = copy.deepcopy (النماذج [0])
    النموذج الأساسي.to ('ميتا')

    المعلمات، الذاكرات = torch.func.stack_module_state (النماذج)

    # من الممكن vmap مباشرة على torch.func.functional_call،
    # لكن لفها في وظيفة يجعل من الواضح ما يحدث.
    def call_single_model (params، buffers، data):
        return torch.func.functional_call (النموذج الأساسي، (params، buffers)، (data،))

    الإخراج = الشعلة.vmap (call_single_model، (0، 0، لا شيء)) (params، buffers، data)
    التأكيد على أن الإخراج. الشكل == (عدد النماذج، حجم الدفعة، out_features)


functorch.compile
-----------------

لم نعد ندعم functorch.compile (المعروف أيضًا باسم AOTAutograd) كواجهة أمامية للتجميع في PyTorch؛ لقد قمنا بتكامل AOTAutograd في قصة التجميع في PyTorch. إذا كنت مستخدمًا، فيرجى استخدام :func:`torch.compile` بدلاً من ذلك.