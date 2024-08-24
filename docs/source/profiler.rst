.. currentmodule:: torch.profiler

torch.profiler
==============

نظرة عامة
--------
.. automodule:: torch.profiler

مرجع واجهة برمجة التطبيقات
-------------

.. autoclass:: torch.profiler._KinetoProfile
  :members:

.. autoclass:: torch.profiler.profile
  :members:

.. autoclass:: torch.profiler.ProfilerAction
  :members:

.. autoclass:: torch.profiler.ProfilerActivity
  :members:

.. autofunction:: torch.profiler.schedule

.. autofunction:: torch.profiler.tensorboard_trace_handler

واجهات برمجة التطبيقات لتكنولوجيا Intel Instrumentation and Tracing
----------------------------------------------------------

.. autofunction:: torch.profiler.itt.is_available

.. autofunction:: torch.profiler.itt.mark

.. autofunction:: torch.profiler.itt.range_push

.. autofunction:: torch.profiler.itt.range_pop

.. يحتاج هذا الموديول إلى توثيق. نقوم بإضافته هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.profiler.itt
.. py:module:: torch.profiler.profiler
.. py:module:: torch.profiler.python_tracer