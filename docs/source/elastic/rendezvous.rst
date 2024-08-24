.. _rendezvous-api:

نقطة الالتقاء
==========

.. automodule:: torch.distributed.elastic.rendezvous

فيما يلي مخطط حالة يصف كيفية عمل نقطة الالتقاء.

.. image:: etcd_rdzv_diagram.png

السجل
-------

.. autoclass:: RendezvousParameters
   :members:

.. autoclass:: RendezvousHandlerRegistry

.. automodule:: torch.distributed.elastic.rendezvous.registry

مُعالج
-------

.. currentmodule:: torch.distributed.elastic.rendezvous

.. autoclass:: RendezvousHandler
   :members:

فئات البيانات
-----------
.. autoclass:: RendezvousInfo

.. currentmodule:: torch.distributed.elastic.rendezvous.api

.. autoclass:: RendezvousStoreInfo

   .. automethod:: build(rank, store)

الاستثناءات
---------
.. autoclass:: RendezvousError
.. autoclass:: RendezvousClosedError
.. autoclass:: RendezvousTimeoutError
.. autoclass:: RendezvousConnectionError
.. autoclass:: RendezvousStateError
.. autoclass:: RendezvousGracefulExitError

التنفيذ
-------

نقطة الالتقاء الديناميكية
******************

.. currentmodule:: torch.distributed.elastic.rendezvous.dynamic_rendezvous

.. autofunction:: create_handler

.. autoclass:: DynamicRendezvousHandler()
   :members: from_backend

.. autoclass:: RendezvousBackend
   :members:

.. autoclass:: RendezvousTimeout
   :members:

الخلفي C10d
^^^^^^^^^^^^

.. currentmodule:: torch.distributed.elastic.rendezvous.c10d_rendezvous_backend

.. autofunction:: create_backend

.. autoclass:: C10dRendezvousBackend
   :members:

خلفي Etcd
^^^^^^^^^^^^

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_rendezvous_backend

.. autofunction:: create_backend

.. autoclass:: EtcdRendezvousBackend
   :members:

نقطة الالتقاء Etcd (الإصدار القديم)
************************

.. warning::
    تفوق فئة ``DynamicRendezvousHandler`` على فئة ``EtcdRendezvousHandler``
    ويوصى بها لمعظم المستخدمين. ``EtcdRendezvousHandler`` في
    وضع الصيانة وسيتم إيقافها في المستقبل.

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_rendezvous

.. autoclass:: EtcdRendezvousHandler

متجر Etcd
**********

"EtcdStore" هو نوع مثيل "Store" C10d الذي يتم إرجاعه بواسطة
``next_rendezvous()`` عندما يتم استخدام etcd كخلفية نقطة الالتقاء.

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_store

.. autoclass:: EtcdStore
   :members:

خادم Etcd
***********

"EtcdServer" هو فئة ملائمة تجعل من السهل عليك
بدء وإيقاف خادم etcd في عملية فرعية. هذا مفيد لاختبار
أو عمليات النشر للعقدة الواحدة (متعددة العمال) حيث من الصعب إعداد
خادم etcd يدويًا.

.. warning:: بالنسبة لعمليات النشر الإنتاجية والمتعددة العقد، يرجى مراعاة
             نشر خادم etcd عالي التوفر بشكل صحيح، حيث أن هذا هو
             نقطة فشل واحدة لوظائفك الموزعة.

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_server

.. autoclass:: EtcdServer

آمل أن يكون هذا ما تبحث عنه! إذا كنت تريد تنسيق النص أو الترجمة بطريقة مختلفة، فما عليك سوى أن تطلب.