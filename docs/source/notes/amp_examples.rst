.. _amp-examples:

أمثلة الدقة المختلطة التلقائية

-----------------
.. currentmodule:: torch.amp

عادةً، يعني "التدريب المختلط التلقائي الدقة" التدريب باستخدام :class:`torch.autocast` و :class:`torch.amp.GradScaler` معًا.

تتيح حالات :class:`torch.autocast` التحويل التلقائي للدقة المختلطة للمناطق المختارة.
يختار التحويل التلقائي للدقة المختلطة تلقائيًا الدقة لعمليات GPU لتحسين الأداء مع الحفاظ على الدقة.

تساعد حالات :class:`torch.amp.GradScaler` في تنفيذ خطوات
تدرج التدرج بسهولة. تحسين التقارب للشبكات مع ``float16`` (بشكل افتراضي على CUDA و XPU)
التدرجات عن طريق تقليل تدرج التدفق، كما هو موضح :ref:`هنا<gradient-scaling>`.

:class:`torch.autocast` و :class:`torch.amp.GradScaler` قابلة للتطوير.
في العينات أدناه، يتم استخدام كل منها كما تشير وثائقه الفردية.

(العينات هنا توضيحية.  راجع
`وصفة الدقة المختلطة التلقائية <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>`_
للمشي القابل للتنفيذ.)

.. contents:: :local:

التدريب النموذجي على الدقة المختلطة
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # إنشاء نموذج ومحسن في الدقة الافتراضية
    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(), ...)

    # إنشاء GradScaler مرة واحدة في بداية التدريب.
    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()

            # تشغيل التمرير للأمام مع التحويل التلقائي للدقة المختلطة.
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)

            # تدرج المقاييس. يدعو backward() على الخسارة المُدرجة لإنشاء تدرجات مُدرجة.
            # لا يوصى بعمليات backward تحت autocast.
            # تعمل عمليات backward في نفس dtype التي اختارتها autocast لعمليات forward المقابلة.
            scaler.scale(loss).backward()

            # أولاً يقوم scaler.step() بإلغاء تدرج تدرجات المعلمات المعينة للمحسن.
            # إذا لم تحتوي هذه التدرجات على infs أو NaNs، يتم استدعاء optimizer.step()،
            # وإلا، يتم تخطي optimizer.step().
            scaler.step(optimizer)

            # تحديث المقياس للتكرار التالي.
            scaler.update()

.. _working-with-unscaled-gradients:

العمل مع التدرجات غير المُدرجة
^^^^^^^^^^^^^^^^^^^^^^^^

جميع التدرجات التي ينتجها ``scaler.scale(loss).backward()`` مُدرجة.  إذا كنت ترغب في تعديل أو فحص
صفات ``.grad`` للمعلمات بين ``backward()`` و ``scaler.step(optimizer)``، يجب عليك
قم بإلغاء تدرجها أولاً. على سبيل المثال، تتلاعب قصاصة التدرج بمجموعة من التدرجات بحيث يكون معيارها العالمي
(راجع :func:`torch.nn.utils.clip_grad_norm_`) أو الحد الأقصى للقيمة (راجع :func:`torch.nn.utils.clip_grad_value_`)
هو :math:`<=` بعض عتبات المستخدم المفروضة.  إذا حاولت القص *بدون* إلغاء التدرج، فسيتم أيضًا تدرج معيار التدرج/القيمة القصوى، لذا فإن عتبة الطلب (التي كان من المفترض أن تكون عتبة للتدرجات *غير المُدرجة*) ستكون غير صالحة.

``scaler.unscale_(optimizer)`` يلغي تدرج التدرجات التي تحتفظ بها معلمات "optimizer".
إذا احتوى نموذجك أو نماذجك على معلمات أخرى تم تعيينها لمحسن آخر
(قل "optimizer2")، فيمكنك استدعاء ``scaler.unscale_(optimizer2)`` بشكل منفصل لإلغاء تدرج تلك
تدرجات المعلمات أيضًا.

قص التدرج
--------

يسمح استدعاء ``scaler.unscale_(optimizer)`` قبل القص بقص التدرجات غير المُدرجة كالمعتاد::

    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)
            scaler.scale(loss).backward()

            # يلغي تدرج تدرجات المعلمات المعينة للمحسن في المكان
            scaler.unscale_(optimizer)

            # نظرًا لأن تدرجات المعلمات المعينة للمحسن غير مُدرجة، فالقص كالمعتاد:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # تدرجات المحسن غير مُدرجة بالفعل، لذا لا يقوم scaler.step بإلغاء تدرجها،
            # على الرغم من أنه لا يزال يتخطى optimizer.step() إذا كانت التدرجات تحتوي على infs أو NaNs.
            scaler.step(optimizer)

            # تحديث المقياس للتكرار التالي.
            scaler.update()

يسجل "scaler" أن ``scaler.unscale_(optimizer)`` تم استدعاؤه بالفعل لهذا المحسن
هذا التكرار، لذا فإن ``scaler.step(optimizer)`` يعرف عدم إلغاء تدرج التدرجات بشكل زائد
قبل (داخليًا) استدعاء ``optimizer.step()``.

.. currentmodule:: torch.amp.GradScaler

.. warning::
    :meth:`unscale_<unscale_>` يجب استدعاء مرة واحدة فقط لكل محسن لكل مكالمة :meth:`step<step>`،
    وفقط بعد تراكم جميع التدرجات لمعلمات المحسن المعين.
    يؤدي استدعاء :meth:`unscale_<unscale_>` مرتين لمحسن معين بين كل مكالمة :meth:`step<step>` إلى تشغيل RuntimeError.


العمل مع التدرجات المُدرجة
^^^^^^^^^^^^^^^^^^^

تراكم التدرج
----------

يضيف تراكم التدرج التدرجات عبر دفعة فعالة بحجم "batch_per_iter * iters_to_accumulate"
("* num_procs" إذا كان موزعًا). يجب معايرة المقياس للدفعة الفعالة، مما يعني التحقق من inf/NaN،
تخطي الخطوة إذا تم العثور على تدرجات inf/NaN، وتحديثات المقياس يجب أن تحدث على مستوى الدفعة الفعالة.
أيضًا، يجب أن تظل التدرجات مُدرجة، ويجب أن يظل عامل التدرج ثابتًا، أثناء تراكم التدرجات لدفعة فعالة معينة.  إذا تم إلغاء تدرج التدرجات (أو تغيير عامل التدرج) قبل اكتمال التراكم،
ستضيف عملية backward التالية التدرجات المُدرجة إلى التدرجات غير المُدرجة (أو التدرجات المُدرجة بعامل مختلف)
بعد ذلك، يصبح من المستحيل استرداد التدرجات المتراكمة غير المُدرجة التي يجب أن تطبقها الخطوة :meth:`step<step>`.

لذلك، إذا كنت ترغب في :meth:`unscale_<unscale_>` التدرجات (على سبيل المثال، للسماح بقص التدرجات غير المُدرجة)،
استدعاء :meth:`unscale_<unscale_>` قبل :meth:`step<step>` مباشرةً، بعد تراكم جميع (المُدرجة) التدرجات لـ
:meth:`step<step>` القادم. أيضًا، قم بالاتصال فقط :meth:`update<update>` في نهاية التكرارات
حيث قمت بالاتصال :meth:`step<step>` لدفعة فعالة كاملة::

    scaler = GradScaler()

    for epoch in epochs:
        for i, (input, target) in enumerate(data):
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)
                loss = loss / iters_to_accumulate

            # تراكم التدرجات المُدرجة.
            scaler.scale(loss).backward()

            if (i + 1) % iters_to_accumulate == 0:
                # قد تقوم بإلغاء التدرج هنا إذا كنت ترغب في ذلك (على سبيل المثال، للسماح بقص التدرجات غير المُدرجة)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

.. currentmodule:: torch.amp

عقوبة التدرج
-----------

تنفيذ العقوبة التدرجية عادةً ما يقوم بإنشاء تدرجات باستخدام
:func: `torch.autograd.grad` ، ودمجها لإنشاء قيمة العقوبة،
ويضيف قيمة العقوبة إلى الخسارة.

فيما يلي مثال عادي لعقوبة L2 بدون قياس التدرج أو التحويل التلقائي::

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)

            # إنشاء التدرجات
            grad_params = torch.autograd.grad(outputs=loss,
                                              inputs=model.parameters(),
                                              create_graph=True)

            # حساب مصطلح العقوبة وإضافته إلى الخسارة
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

            loss.backward()

            # قص التدرجات هنا، إذا رغبت في ذلك

            optimizer.step()

لتنفيذ عقوبة التدرج *مع* قياس التدرج، يجب تحجيم تنسور (s)
الناتجة التي يتم تمريرها إلى: func: `torch.autograd.grad`. لذلك، ستكون التدرجات الناتجة محددة النطاق، ويجب إلغاء تحجيمها قبل دمجها لإنشاء
قيمة العقوبة.

أيضًا، يمثل حساب مصطلح العقوبة جزءًا من عملية التغذية الأمامية، وبالتالي يجب أن يكون
داخل سياق :class: `autocast`.

هكذا يبدو الأمر بالنسبة لعقوبة L2 العادية::

    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)

            # تحجيم الخسارة لخلفي autograd.grad ، مما ينتج عنه scaled_grad_params
            scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
                                                     inputs=model.parameters(),
                                                     create_graph=True)

            # إنشاء grad_params غير محددة النطاق قبل حساب العقوبة. لا يتم امتلاك scaled_grad_params
            # بواسطة أي محسن، لذلك يتم استخدام القسمة العادية بدلاً من scaler.unscale_:
            inv_scale = 1./scaler.get_scale()
            grad_params = [p * inv_scale for p in scaled_grad_params]

            # حساب مصطلح العقوبة وإضافته إلى الخسارة
            with autocast(device_type='cuda', dtype=torch.float16):
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

            # تطبيق التحجيم على مكالمة الخلفي كما هو معتاد.
            # تراكم التدرجات الورقية التي يتم تحجيمها بشكل صحيح.
            scaler.scale(loss).backward()

            # قد unscale_ هنا إذا رغبت في ذلك (على سبيل المثال، للسماح بقص التدرجات غير المحددة النطاق)

            # تتقدم الخطوة () والتحديث () كما هو معتاد.
            scaler.step(optimizer)
            scaler.update()


العمل مع نماذج وخسائر ومحسنات متعددة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torch.amp.GradScaler

إذا كانت شبكتك تحتوي على خسائر متعددة، فيجب عليك استدعاء: meth: `scaler.scale <scale>` على كل منها بشكل فردي.
إذا كانت شبكتك تحتوي على محسنات متعددة، فيمكنك استدعاء: meth: `scaler.unscale_ <unscale_>` على أي منها بشكل فردي،
ويجب عليك استدعاء: meth: `scaler.step <step>` على كل منها بشكل فردي.

ومع ذلك، يجب استدعاء: meth: `scaler.update <update>` مرة واحدة فقط،
بعد أن يتم تنفيذ جميع المحسنات المستخدمة في هذه الحلقة::

    scaler = torch.amp.GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                output0 = model0(input)
                output1 = model1(input)
                loss0 = loss_fn(2 * output0 + 3 * output1, target)
                loss1 = loss_fn(3 * output0 - 5 * output1, target)

            # (retain_graph هنا لا علاقة له بـ amp ، فهو موجود لأن في هذا
            # المثال، يشارك كلا الاستدعاءين بعض أجزاء الرسم البياني.)
            scaler.scale(loss0).backward(retain_graph=True)
            scaler.scale(loss1).backward()

            # يمكنك اختيار المحسنات التي تتلقى unscaling صريح، إذا كنت
            # تريد فحص أو تعديل التدرجات من المعلمات التي تمتلكها.
            scaler.unscale_(optimizer0)

            scaler.step(optimizer0)
            scaler.step(optimizer1)

            scaler.update()

يفحص كل محسن التدرجات الخاصة به بحثًا عن infs/NaNs ويتخذ قرارًا مستقلًا
ما إذا كان سيتم تخطي الخطوة أم لا. قد يؤدي ذلك إلى تخطي أحد المحسنات للخطوة
في حين أن الآخر لا يفعل ذلك. نظرًا لأن تخطي الخطوة يحدث نادرًا (كل بضع مئات من الحلقات)
هذا لا ينبغي أن يعوق التقارب. إذا لاحظت تقاربًا ضعيفًا بعد إضافة قياس التدرج
إلى نموذج متعدد المحسنات، يرجى الإبلاغ عن خطأ.

.. currentmodule:: torch.amp

.. _amp-multigpu:

العمل مع وحدات معالجة الرسومات المتعددة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

المشكلات الموضحة هنا تؤثر فقط على: class: `autocast`. لا يتغير استخدام :class: `GradScaler`.

.. _amp-dataparallel:

DataParallel في عملية واحدة
-------------------------

حتى إذا كان :class: `torch.nn.DataParallel` يولد خيوطًا لتشغيل التغذية الأمامية على كل جهاز.
يتم نشر حالة autocast في كل منها وسيعمل ما يلي::

    model = MyModel()
    dp_model = nn.DataParallel(model)

    # تعيين autocast في الخيط الرئيسي
    with autocast(device_type='cuda', dtype=torch.float16):
        # سوف تُمكِّن خيوط dp_model الداخلية autocast.
        output = dp_model(input)
        # loss_fn autocast أيضًا
        loss = loss_fn(output)

DistributedDataParallel ، وحدة معالجة رسومات واحدة لكل عملية
----------------------------------------------------

يوصي توثيق :class: `torch.nn.parallel.DistributedDataParallel` بوحدة معالجة رسومات واحدة لكل عملية للحصول على أفضل
الأداء. في هذه الحالة، لا يقوم "DistributedDataParallel" بإنشاء خيوط داخلية،
لذلك لا تتأثر استخدامات :class: `autocast` و: class: `GradScaler`.

DistributedDataParallel ، وحدات معالجة الرسومات المتعددة لكل عملية
------------------------------------------------------

هنا، قد يقوم :class: `torch.nn.parallel.DistributedDataParallel` بإنشاء خيط جانبي لتشغيل التغذية الأمامية على كل
جهاز، مثل :class: `torch.nn.DataParallel`. الإصلاح هو نفسه:
قم بتطبيق autocast كجزء من طريقة "forward" للنموذج لضمان تمكينه في الخيوط الجانبية.

.. _amp-custom-examples:

Autocast و Custom Autograd Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا كانت شبكتك تستخدم: ref: `custom autograd functions <extending-autograd>`
(فئات فرعية من: class: `torch.autograd.Function`)، تكون التغييرات مطلوبة
توافق autocast إذا كان أي دالة

* يأخذ مدخلات Tensor العائمة متعددة،
* يلتف أي عملية autocastable (راجع: ref: `Autocast Op Reference <autocast-op-reference>`)، أو
* يتطلب "dtype" معين (على سبيل المثال، إذا كان يلتف
  `ملحقات CUDA <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_
  التي تم تجميعها فقط لـ "dtype").

في جميع الحالات، إذا كنت تقوم باستيراد الدالة ولا يمكنك تغيير تعريفها، فإن التراجع الآمن
هو تعطيل autocast وإجبار التنفيذ في "float32" (أو "dtype") في أي نقاط الاستخدام التي تحدث فيها الأخطاء::

    with autocast(device_type='cuda', dtype=torch.float16):
        ...
        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            output = imported_function(input1.float(), input2.float())

إذا كنت مؤلف الدالة (أو يمكنك تغيير تعريفها)، فإن الحل الأفضل هو استخدام
:func: `مُزينات torch.amp.custom_fwd` و: func: `torch.amp.custom_bwd` كما هو موضح في
الحالة ذات الصلة أدناه.

وظائف مع مدخلات متعددة أو عمليات autocastable
--------------------------------------

قم بتطبيق: func: `custom_fwd <custom_fwd>` و: func: `custom_bwd <custom_bwd>` (بدون حجج) على "forward" و
"backward" على التوالي. تضمن هذه الدوال تشغيل "forward" مع حالة autocast الحالية وتشغيل "backward"
مع نفس حالة autocast مثل "forward" (والذي يمكن أن يمنع أخطاء عدم تطابق النوع)::

    class MyMM(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, a، b):
            ctx.save_for_backward(a، b)
            return a.mm(b)
        @staticmethod
        @custom_bwd
        def backward(ctx، grad):
            a، b = ctx.saved_tensors
            return grad.mm(b.t())، a.t().mm(grad)

الآن يمكن استدعاء "MyMM" في أي مكان، دون تعطيل autocast أو يدويًا يلقي المدخلات::

    mymm = MyMM.apply

    with autocast(device_type='cuda', dtype=torch.float16):
        output = mymm(input1, input2)

وظائف تتطلب "dtype" معين
-----------------------

ضع في اعتبارك وظيفة مخصصة تتطلب مدخلات "torch.float32".
تطبيق: func: `custom_fwd (device_type = 'cuda'، cast_inputs = torch.float32) <custom_fwd>` على "forward"
و: func: `custom_bwd (device_type = 'cuda') <custom_bwd>` إلى "backward".
إذا تم تشغيل "forward" في منطقة ممكّنة لـ autocast ، فإن الديكورات تقوم بتحويل تنسورات المدخلات العائمة إلى "float32" على الجهاز المحدد الذي تم تعيينه بواسطة حجة `device_type <../amp.html>`_،
"CUDA" في هذا المثال، وتعطيل autocast محليًا أثناء "forward" و "backward"::

    class MyFloat32Func(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
        def forward(ctx, input):
            ctx.save_for_backward(input)
            ...
            return fwd_output
        @staticmethod
        @custom_bwd(device_type='cuda')
        def backward(ctx, grad):
            ...

الآن يمكن استدعاء "MyFloat32Func" في أي مكان، دون تعطيل autocast أو يدويًا يلقي المدخلات::

    func = MyFloat32Func.apply

    with autocast(device_type='cuda', dtype=torch.float16):
        # سيتم تشغيل func في float32، بغض النظر عن حالة autocast المحيطة
        output = func(input)