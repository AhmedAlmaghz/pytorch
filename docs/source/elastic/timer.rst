مؤقتات انتهاء الصلاحية
================

.. automodule:: torch.distributed.elastic.timer
.. currentmodule:: torch.distributed.elastic.timer

طرق العميل
----------
.. autofunction:: torch.distributed.elastic.timer.configure

.. autofunction:: torch.distributed.elastic.timer.expires

تنفيذيات الخادم/العميل
-----------------
فيما يلي أزواج خادم المؤقت وعميله التي يوفرها torchelastic.

.. note:: يجب دائمًا تنفيذ خادم المؤقت والعملاء واستخدامها
          في أزواج نظرًا لوجود بروتوكول رسائل بين الخادم
          والعميل.

فيما يلي زوج من خادم المؤقت والعميل الذي يتم تنفيذه بناءً على
``multiprocess.Queue``.

.. autoclass:: LocalTimerServer

.. autoclass:: LocalTimerClient

فيما يلي زوج آخر من خادم المؤقت والعميل الذي يتم تنفيذه
بناءً على Pipe مسمى.

.. autoclass:: FileTimerServer

.. autoclass:: FileTimerClient

كتابة خادم مؤقت/عميل مخصص
----------------------

لإنشاء خادم مؤقت وعميل خاص بك، قم بتوسيع
``torch.distributed.elastic.timer.TimerServer`` للخادم و
``torch.distributed.elastic.timer.TimerClient`` للعميل. يتم استخدام كائن
``TimerRequest`` لنقل الرسائل بين
الخادم والعميل.

.. autoclass:: TimerRequest
   :members:

.. autoclass:: TimerServer
   :members:

.. autoclass:: TimerClient
   :members:

تسجيل معلومات التصحيح
-------------------

.. automodule:: torch.distributed.elastic.timer.debug_info_logging

.. autofunction:: torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers