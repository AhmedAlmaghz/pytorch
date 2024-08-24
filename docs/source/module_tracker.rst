torch.utils.module_tracker
============================
.. automodule:: torch.utils.module_tracker

يمكن استخدام هذه الأداة المساعدة لتتبع الموقع الحالي داخل هرمية :class:`torch.nn.Module`.
يمكن استخدامها ضمن أدوات التتبع الأخرى لتكون قادرًا على ربط الكميات المقاسة بسهولة بأسماء سهلة الاستخدام. ويستخدم هذا بشكل خاص في FlopCounterMode اليوم.

.. autoclass:: torch.utils.module_tracker.ModuleTracker