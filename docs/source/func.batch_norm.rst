ترقيع Batch Norm
==================

ماذا يحدث؟
-----------------
يتطلب Batch Norm تحديثات في المكان لـ running_mean و running_var بنفس حجم الإدخال.
Functorch لا يدعم التحديث في المكان إلى Tensor عادي يأخذ Tensor مجمعة (أي
"regular.add_ (batched)" غير مسموح به). لذلك عندما نقوم بالرسم فوق دفعة من الإدخالات إلى وحدة واحدة،
نحصل على هذا الخطأ

كيفية الإصلاح
----------
إحدى أفضل الطرق المدعومة هي التبديل من BatchNorm إلى GroupNorm. الخياران 1 و 2 يدعمان هذا

تفترض جميع هذه الخيارات أنك لا تحتاج إلى الإحصائيات الجارية. إذا كنت تستخدم وحدة نمطية، فهذا يعني
من المفترض أنك لن تستخدم المعيار المعياري للدفعة في وضع التقييم. إذا كانت لديك حالة استخدام تتضمن
تشغيل معيار الدُفعة مع vmap في وضع التقييم، يرجى تقديم مشكلة

الخيار 1: تغيير BatchNorm
^^^^^^^^^^^^^^^^^^^^^^^^^
إذا كنت تريد التغيير إلى GroupNorm، ففي أي مكان يوجد فيه BatchNorm، استبدله بما يلي:

.. code-block:: python

    BatchNorm2d(C, G, track_running_stats=False)

هنا "C" هي نفسها "C" كما في BatchNorm الأصلي. "G" هو عدد المجموعات
قم بتقسيم "C" إلى. وبالتالي، "C % G == 0" وكحل بديل، يمكنك تعيين "C == G"، مما يعني
سيتم التعامل مع كل قناة بشكل منفصل.

إذا كنت مضطرًا لاستخدام BatchNorm وقمت ببناء الوحدة النمطية بنفسك، فيمكنك تغيير الوحدة النمطية لعدم
استخدام الإحصائيات الجارية. وبعبارة أخرى، في أي مكان يوجد فيه وحدة نمطية BatchNorm، قم بتعيين
علم "track_running_stats" إلى False

.. code-block:: python

    BatchNorm2d(64, track_running_stats=False)


الخيار 2: معلمة torchvision
^^^^^^^^^^^^^^^^^^^^^^^^^
يمكن لبعض نماذج torchvision، مثل resnet و regnet، أن تأخذ
معلمة "norm_layer". غالبًا ما تكون هذه المعلمات الافتراضية هي BatchNorm2d إذا تم تعيينها كافتراضي.

بدلاً من ذلك، يمكنك تعيينه إلى GroupNorm.

.. code-block:: python

    import torchvision
    from functools import partial
    torchvision.models.resnet18(norm_layer=lambda c: GroupNorm(num_groups=g, c))

مرة أخرى، "c % g == 0" لذلك، كحل بديل، قم بتعيين "g = c".

إذا كنت مرتبطًا بـ BatchNorm، فتأكد من استخدام إصدار لا يستخدم الإحصائيات الجارية

.. code-block:: python

    import torchvision
    from functools import partial
    torchvision.models.resnet18(norm_layer=partial(BatchNorm2d, track_running_stats=False))

الخيار 3: ترقيع functorch
^^^^^^^^^^^^^^^^^^^^^^^^
أضاف functorch بعض الوظائف للسماح بالترقيع السريع في المكان للوحدة النمطية لعدم
استخدام الإحصائيات الجارية. تغيير طبقة المعيار أكثر هشاشة، لذلك لم نقدم ذلك. إذا كان لديك
شبكة حيث تريد أن لا يستخدم BatchNorm الإحصائيات الجارية، يمكنك تشغيل
"replace_all_batch_norm_modules_" لتحديث الوحدة النمطية في المكان لعدم استخدام الإحصائيات الجارية

.. code-block:: python

    from torch.func import replace_all_batch_norm_modules_
    replace_all_batch_norm_modules_(net)

الخيار 4: وضع التقييم
^^^^^^^^^^^^^^^^
عند التشغيل في وضع التقييم، لن يتم تحديث running_mean و running_var. لذلك، يمكن لـ vmap دعم هذا الوضع

.. code-block:: python

    model.eval()
    vmap(model)(x)
    model.train()