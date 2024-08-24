.. _MPS-Backend:

الخلفية الخلفية لـ MPS
=====================

تتيح وحدة :mod:`mps` التدريب عالي الأداء على وحدات معالجة الرسوميات (GPU) لأجهزة MacOS مع إطار عمل برمجة Metal. ويقدم وحدة جديدة لرسم خرائط الرسوم البيانية والبدائيات الحسابية للتعلم الآلي على إطار عمل Metal Performance Shaders Graph وإطارات عمل Metal Performance Shaders على التوالي.

توسع خلفية MPS الجديدة نظام PyTorch البيئي وتوفر للمخطوطات الموجودة إمكانيات لإعداد وتشغيل العمليات على وحدة معالجة الرسومات.

لتبدأ، ما عليك سوى نقل Tensor و Module إلى جهاز "mps":

.. code:: python

    # تحقق من توفر MPS
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS غير متوفر لأن تثبيت PyTorch الحالي لم يتم بناؤه "
                  "مع تمكين MPS.")
        else:
            print("MPS غير متوفر لأن إصدار MacOS الحالي ليس 12.3+ "
                  "و/أو لا يوجد لديك جهاز ممكّن لـ MPS على هذه الآلة.")

    else:
        mps_device = torch.device("mps")

        # إنشاء Tensor مباشرة على جهاز mps
        x = torch.ones(5, device=mps_device)
        # أو
        x = torch.ones(5, device="mps")

        # أي عملية تحدث على وحدة معالجة الرسوميات
        y = x * 2

        # قم بنقل نموذجك إلى mps مثل أي جهاز آخر
        model = YourFavoriteNet()
        model.to(mps_device)

        # الآن يتم تشغيل كل مكالمة على وحدة معالجة الرسوميات
        pred = model(x)