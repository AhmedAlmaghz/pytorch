:العربية:
مرجع واجهة برمجة التطبيقات الكمّية
torch.ao.quantization
~~~~~~~~~~~~~~~~~~~~~

يحتوي هذا النموذج على واجهات برمجة التطبيقات (APIs) لوضع التهيئة Eager mode.

.. currentmodule:: torch.ao.quantization

واجهات برمجة التطبيقات (APIs) عالية المستوى
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    التهيئة
    التهيئة_ديناميكية
    التهيئة_qat
    الإعداد
    الإعداد_qat
    التحويل

إعداد النموذج للتهيئة
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    دمج_الوحدات.دمج_الوحدات
    QuantStub
    DeQuantStub
    QuantWrapper
    إضافة_التهيئة_إلغاء_التهيئة

وظائف المنفعة
^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    تبديل_الوحدة
    انتشار_qconfig_
    وظيفة_التقييم_الافتراضية


torch.ao.quantization.quantize_fx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

يحتوي هذا النموذج على واجهات برمجة التطبيقات (APIs) لوضع الرسم البياني FX (النموذج الأولي).

.. currentmodule:: torch.ao.quantization.quantize_fx

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    الإعداد_fx
    الإعداد_qat_fx
    التحويل_fx
    دمج_fx

torch.ao.quantization.qconfig_mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

يحتوي هذا النموذج على QConfigMapping لتهيئة وضع الرسم البياني FX.

.. currentmodule:: torch.ao.quantization.qconfig_mapping

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    QConfigMapping
    الحصول_على_تهيئة_qconfig_افتراضية
    الحصول_على_تهيئة_qat_qconfig_افتراضية

torch.ao.quantization.backend_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

يحتوي هذا النموذج على BackendConfig، وهو كائن تهيئة يحدد كيفية دعم التهيئة
في backend. يستخدم حاليًا فقط بواسطة وضع الرسم البياني FX Quantization، ولكن قد نقوم بتوسيع وضع التهيئة Eager
للعمل مع هذا أيضًا.

.. currentmodule:: torch.ao.quantization.backend_config

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BackendConfig
    BackendPatternConfig
    DTypeConfig
    DTypeWithConstraints
    ObservationType

torch.ao.quantization.fx.custom_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

يحتوي هذا النموذج على بعض فئات CustomConfig التي تستخدم في كل من وضع التهيئة Eager ووضع الرسم البياني FX.


.. currentmodule:: torch.ao.quantization.fx.custom_config

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    FuseCustomConfig
    PrepareCustomConfig
    ConvertCustomConfig
    StandaloneModuleConfigEntry

torch.ao.quantization.quantizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: torch.ao.quantization.quantizer

torch.ao.quantization.pt2e (تنفيذ التهيئة في تصدير pytorch 2.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: torch.ao.quantization.pt2e
.. automodule:: torch.ao.quantization.pt2e.representation

torch.ao.quantization.pt2e.export_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torch.ao.quantization.pt2e.export_utils

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    النموذج_المصدر

.. currentmodule:: torch.ao.quantization

PT2 Export (pt2e) Numeric Debugger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    إنشاء_مُعالج_التصحيح_الرقمي
    NUMERIC_DEBUG_HANDLE_KEY
    الإعداد_لـ_مقارنة_الانتشار
    استخراج_النتائج_من_سجلات_النشاط
    مقارنة_النتائج

torch (وظائف متعلقة بالتهيئة)
~~~~~~~~~~~~~~~~~~~~~

يصف هذا القسم وظائف التهيئة ذات الصلة في مساحة الاسم "torch".

.. currentmodule:: torch

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    التهيئة_لكل_موتر
    التهيئة_لكل_قناة
    إلغاء_التهيئة

torch.Tensor (أساليب متعلقة بالتهيئة)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

تدعم المتوترات الكمية مجموعة فرعية محدودة من أساليب معالجة البيانات للمتوتر
العادي عالي الدقة.

.. currentmodule:: torch.Tensor

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    view
    as_strided
    expand
    flatten
    select
    ne
    eq
    ge
    le
    gt
    lt
    copy_
    clone
    dequantize
    equal
    int_repr
    max
    mean
    min
    q_scale
    q_zero_point
    q_per_channel_scales
    q_per_channel_zero_points
    q_perِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِِ
هذا هو النص المترجم إلى اللغة العربية بتنسيق ReStructuredText:

تمت معايرة الكميات الديناميكية لـ :class:`~torch.nn.Linear`، و :class:`~torch.nn.LSTM`،
و :class:`~torch.nn.LSTMCell`، و :class:`~torch.nn.GRUCell`، و
:class:`~torch.nn.RNNCell`.

.. currentmodule:: torch.ao.nn.quantized.dynamic

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Linear
    LSTM
    GRU
    RNNCell
    LSTMCell
    GRUCell

أنواع البيانات المكمية ومخططات الكم
~~~~~~~~~~~~~~~~~~~~~~~~

لاحظ أن تطبيقات المشغل تدعم حاليًا الكميات لكل قناة لأوزان المشغلين **conv** و **linear** فقط. علاوة على ذلك، يتم
تعيين بيانات الإدخال بشكل خطي إلى البيانات المكمية والعكس صحيح
كما يلي:

    .. math::

        \begin{aligned}
            \text{Quantization:}&\\
            &Q_\text{out} = \text{clamp}(x_\text{input}/s+z, Q_\text{min}, Q_\text{max})\\
            \text{Dequantization:}&\\
            &x_\text{out} = (Q_\text{input}-z)*s
        \end{aligned}

حيث :math:`\text{clamp}(.)` هو نفسه :func:`~torch.clamp` بينما
يتم حساب المقياس :math:`s` ونقطة الصفر :math:`z` كما هو موضح في :class:`~torch.ao.quantization.observer.MinMaxObserver`، على وجه التحديد:

    .. math::

        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}

حيث :math:`[x_\text{min}, x_\text{max}]` يمثل نطاق بيانات الإدخال في حين
:math:`Q_\text{min}` و :math:`Q_\text{max}` هما على التوالي الحد الأدنى والحد الأقصى للقيم لنوع البيانات المكمية.

لاحظ أن اختيار :math:`s` و :math:`z` يعني أن الصفر يتم تمثيله بدون خطأ في الكميات كلما كان الصفر ضمن
نطاق بيانات الإدخال أو كان الكم المتماثل قيد الاستخدام.

يمكن تنفيذ أنواع بيانات ومخططات كميات إضافية من خلال
آلية المشغل المخصص <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_.

* :attr:`torch.qscheme` — نوع لوصف مخطط الكميات لموتر.
  الأنواع المدعومة:

  * :attr:`torch.per_tensor_affine` — لكل موتر، غير متماثل
  * :attr:`torch.per_channel_affine` — لكل قناة، غير متماثل
  * :attr:`torch.per_tensor_symmetric` — لكل موتر، متماثل
  * :attr:`torch.per_channel_symmetric` — لكل قناة، متماثل

* ``torch.dtype`` — نوع لوصف البيانات. الأنواع المدعومة:

  * :attr:`torch.quint8` — عدد صحيح غير موقع 8 بت
  * :attr:`torch.qint8` — عدد صحيح موقع 8 بت
  * :attr:`torch.qint32` — عدد صحيح موقع 32 بت


.. تفتقر هذه الوحدات إلى الوثائق. نقوم بإضافتها هنا فقط للمتابعة
.. automodule:: torch.ao.nn.quantizable.modules
   :noindex:
.. automodule:: torch.ao.nn.quantized.reference
   :noindex:
.. automodule:: torch.ao.nn.quantized.reference.modules
   :noindex:

.. automodule:: torch.nn.quantizable
.. automodule:: torch.nn.qat.dynamic.modules
.. automodule:: torch.nn.qat.modules
.. automodule:: torch.nn.qat
.. automodule:: torch.nn.intrinsic.qat.modules
.. automodule:: torch.nn.quantized.dynamic
.. automodule:: torch.nn.intrinsic
.. automodule:: torch.nn.intrinsic.quantized.modules
.. automodule:: torch.quantization.fx
.. automodule:: torch.nn.intrinsic.quantized.dynamic
.. automodule:: torch.nn.qat.dynamic
.. automodule:: torch.nn.intrinsic.qat
.. automodule:: torch.nn.quantized.modules
.. automodule:: torch.nn.intrinsic.quantized
.. automodule:: torch.nn.quantizable.modules
.. automodule:: torch.nn.quantized
.. automodule:: torch.nn.intrinsic.quantized.dynamic.modules
.. automodule:: torch.nn.quantized.dynamic.modules
.. automodule:: torch.quantization
.. automodule:: torch.nn.intrinsic.modules