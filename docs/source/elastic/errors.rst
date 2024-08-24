.. _elastic_errors-api:

انتشار الخطأ
=============

.. automodule:: torch.distributed.elastic.multiprocessing.errors

الدوال والفئات
-------------

.. currentmodule:: torch.distributed.elastic.multiprocessing.errors

.. autofunction:: torch.distributed.elastic.multiprocessing.errors.record

.. autoclass:: ChildFailedError

    استثناء يتم إلقاؤه عندما يفشل أحد العمليات الفرعية.

.. autoclass:: ErrorHandler

    معالج الأخطاء المسؤول عن التعامل مع الأخطاء التي تحدث في العمليات الفرعية.

.. autoclass:: ProcessFailure

    استثناء يتم إلقاؤه عندما تفشل إحدى العمليات.