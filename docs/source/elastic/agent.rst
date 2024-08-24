Elastic Agent
==============

.. automodule:: torch.distributed.elastic.agent
.. currentmodule:: torch.distributed.elastic.agent

الخادم
------

.. automodule:: torch.distributed.elastic.agent.server

فيما يلي رسم تخطيطي لعامل يقوم بإدارة مجموعة محلية من العمال.

.. image:: agent_diagram.jpg

المفاهيم
-----

يصف هذا القسم الفئات والمفاهيم عالية المستوى
ذات الصلة بفهم دور "العامل" في torchelastic.

.. currentmodule:: torch.distributed.elastic.agent.server

.. autoclass:: ElasticAgent
   :members:

.. autoclass:: WorkerSpec
   :members:

.. autoclass:: WorkerState
   :members:

.. autoclass:: Worker
   :members:

.. autoclass:: WorkerGroup
   :members:

التطبيقات
--------

فيما يلي تطبيقات العامل التي يوفرها torchelastic.

.. currentmodule:: torch.distributed.elastic.agent.server.local_elastic_agent
.. autoclass:: LocalElasticAgent

توسيع العامل
----------

لتوسيع العامل، يمكنك تنفيذ "ElasticAgent" مباشرة، ولكن
نوصي بتوسيع "SimpleElasticAgent" بدلاً من ذلك، والذي يوفر
معظم الهيكل ويترك لك بعض الأساليب المجردة المحددة
لتطبيقها.

.. currentmodule:: torch.distributed.elastic.agent.server
.. autoclass:: SimpleElasticAgent
   :members:
   :private-members:

.. autoclass:: torch.distributed.elastic.agent.server.api.RunResult

المراقبة في العامل
--------------

يمكن تمكين مراقبة المسماة على أساس الأنبوب في "LocalElasticAgent" إذا تم
تحديد متغير بيئة "TORCHELASTIC_ENABLE_FILE_TIMER" بقيمة 1 في عملية "LocalElasticAgent".
اختياريًا، يمكن تعيين متغير بيئة آخر "TORCHELASTIC_TIMER_FILE"
باسم ملف فريد لأنبوب المسمى. إذا لم يتم تعيين متغير البيئة "TORCHELASTIC_TIMER_FILE"،
فسيقوم "LocalElasticAgent" بإنشاء اسم ملف فريد داخليًا وتعيينه إلى متغير البيئة
"TORCHELASTIC_TIMER_FILE"، وسيتم نشر متغير البيئة هذا إلى عمليات العامل للسماح لهم
بالاتصال بنفس أنبوب المسمى الذي يستخدمه "LocalElasticAgent".

خادم الفحص الصحي
---------------

يمكن تمكين خادم مراقبة الفحص الصحي في "LocalElasticAgent"
إذا تم تحديد متغير بيئة "TORCHELASTIC_HEALTH_CHECK_PORT" في عملية "LocalElasticAgent".
إضافة واجهة لخادم الفحص الصحي الذي يمكن توسيعه عن طريق بدء خادم tcp/http
على رقم المنفذ المحدد.
بالإضافة إلى ذلك، سيكون لدى خادم الفحص الصحي استدعاء رد اتصال للتحقق مما إذا كان المراقب على قيد الحياة.

.. automodule:: torch.distributed.elastic.agent.server.health_check_server

.. autoclass:: HealthCheckServer
   :members:

.. autofunction:: torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server