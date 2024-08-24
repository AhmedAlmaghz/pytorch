torch.monitor
================

.. warning::

    هذا النموذج هو إصدار أولي، وقد تتغير واجهاته ووظائفه دون سابق إنذار في الإصدارات المستقبلية من PyTorch.

يوفر ``torch.monitor`` واجهة لتسجيل الأحداث والعدادات من PyTorch.

تم تصميم واجهات الإحصائيات لتتبع المقاييس عالية المستوى التي يتم تسجيلها بشكل دوري لاستخدامها في مراقبة أداء النظام. نظرًا لأن الإحصائيات تجمع ضمن حجم نافذة محدد، فيمكنك تسجيلها من الحلقات الحرجة مع تأثير ضئيل على الأداء.

بالنسبة للأحداث أو القيم الأقل تكرارًا، مثل الخسارة والدقة وتتبع الاستخدام، يمكن استخدام واجهة الحدث مباشرة.

يمكن تسجيل معالجات الأحداث للتعامل مع الأحداث وإرسالها إلى مصدر أحداث خارجي.

مرجع API
--------

.. automodule:: torch.monitor

.. autoclass:: torch.monitor.Aggregation
    :members:

.. autoclass:: torch.monitor.Stat
    :members:
    :special-members: __init__

.. autoclass:: torch.monitor.data_value_t
    :members:

.. autoclass:: torch.monitor.Event
    :members:
    :special-members: __init__

.. autoclass:: torch.monitor.EventHandlerHandle
    :members:

.. autofunction:: torch.monitor.log_event

.. autofunction:: torch.monitor.register_event_handler

.. autofunction:: torch.monitor.unregister_event_handler

.. autoclass:: torch.monitor.TensorboardEventHandler
    :members:
    :special-members: __init__