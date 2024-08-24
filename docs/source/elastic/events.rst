.. _events-api:

الأحداث
========

.. automodule:: torch.distributed.elastic.events

طرق واجهة برمجة التطبيقات
-------------------

.. autofunction:: torch.distributed.elastic.events.record

.. autofunction:: torch.distributed.elastic.events.construct_and_record_rdzv_event

.. autofunction:: torch.distributed.elastic.events.get_logging_handler

كائنات الحدث
-----------

.. currentmodule:: torch.distributed.elastic.events.api

.. autoclass:: torch.distributed.elastic.events.api.Event

.. autoclass:: torch.distributed.elastic.events.api.EventSource

.. autoclass:: torch.distributed.elastic.events.api.EventMetadataValue