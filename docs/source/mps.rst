torch.mps
============
.. automodule:: torch.mps
.. currentmodule:: torch.mps

.. autosummary::
    :toctree: generated
    :nosignatures:

    device_count
    synchronize
    get_rng_state
    set_rng_state
    manual_seed
    seed
    empty_cache
    set_per_process_memory_fraction
    current_allocated_memory
    driver_allocated_memory
    recommended_max_memory

ملف تعريف MPS
------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.start
    profiler.stop
    profiler.profile

حدث MPS
------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    event.Event


.. تحتاج هذه الوحدة إلى توثيق. نقوم بإضافتها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.mps.event
.. py:module:: torch.mps.profiler