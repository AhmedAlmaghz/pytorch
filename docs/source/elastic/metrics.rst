.. _metrics-api:

المقاييس
=========

.. automodule:: torch.distributed.elastic.metrics

معالجات المقاييس
--------------

.. currentmodule:: torch.distributed.elastic.metrics.api

فيما يلي معالجات المقاييس التي تأتي مضمنة مع torchelastic.

.. autoclass:: MetricHandler

.. autoclass:: ConsoleMetricHandler

.. autoclass:: NullMetricHandler

الطرق
--------

.. autofunction:: torch.distributed.elastic.metrics.configure

.. autofunction:: torch.distributed.elastic.metrics.prof

.. autofunction:: torch.distributed.elastic.metrics.put_metric