torch.cuda
===================================
.. automodule:: torch.cuda
.. currentmodule:: torch.cuda

.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    يمكن_جهاز_الوصول_إلى_نظيره
    current_blas_handle
    current_device
    current_stream
    cudart
    default_stream
    الجهاز
    device_count
    device_of
    get_arch_list
    get_device_capability
    get_device_name
    get_device_properties
    get_gencode_flags
    get_sync_debug_mode
    init
    ipc_collect
    is_available
    is_initialized
    memory_usage
    set_device
    set_stream
    set_sync_debug_mode
    stream
    synchronize
    utilization
    temperature
    power_draw
    clock_rate
    OutOfMemoryError

مولد الأرقام العشوائية
-------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    set_rng_state
    set_rng_state_all
    manual_seed
    manual_seed_all
    seed
    seed_all
    initial_seed


التجمعات التواصلية
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    comm.broadcast
    comm.broadcast_coalesced
    comm.reduce_add
    comm.scatter
    comm.gather

التدفقات والأحداث
---------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    ExternalStream
    Event

الرسوم البيانية (تجريبي)
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_current_stream_capturing
    graph_pool_handle
    CUDAGraph
    graph
    make_graphed_callables

.. _cuda-memory-management-api:

إدارة الذاكرة
-----------------
.. autosummary::
    :toctree: generated
    :nosignatures:

     empty_cache
     list_gpu_processes
     mem_get_info
     memory_stats
     memory_summary
     memory_snapshot
     memory_allocated
     max_memory_allocated
     reset_max_memory_allocated
     memory_reserved
     max_memory_reserved
     set_per_process_memory_fraction
     memory_cached
     max_memory_cached
     reset_max_memory_cached
     reset_peak_memory_stats
     caching_allocator_alloc
     caching_allocator_delete
     get_allocator_backend
     CUDAPluggableAllocator
     change_current_allocator
     MemPool
     MemPoolContext
.. FIXME لا يبدو أن ما يلي موجود. هل من المفترض أن يكون موجوداً؟
   https://github.com/pytorch/pytorch/issues/27785
   .. autofunction:: reset_max_memory_reserved

NVIDIA Tools Extension (NVTX)
-----------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    nvtx.mark
    nvtx.range_push
    nvtx.range_pop
    nvtx.range

Jiterator (تجريبي)
-----------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    jiterator._create_jit_fn
    jiterator._create_multi_output_jit_fn

TunableOp
---------

يمكن تنفيذ بعض العمليات باستخدام أكثر من مكتبة أو أكثر من تقنية. على
سبيل المثال، يمكن تنفيذ GEMM لـ CUDA أو ROCm باستخدام مكتبات
cublas/cublasLt أو hipblas/hipblasLt، على التوالي. كيف يمكن معرفة أي
تنفيذ هو الأسرع ويجب اختياره؟ هذا ما يوفره TunableOp. تم تنفيذ بعض
المشغلين باستخدام استراتيجيات متعددة كمشغلين قابلين للضبط. أثناء وقت
التشغيل، يتم تحديد جميع الاستراتيجيات واختيار أسرعها لجميع العمليات
اللاحقة.

راجع :doc:`الوثائق <cuda.tunable>` للحصول على معلومات حول كيفية استخدامها.

.. toctree::
    :hidden:

    cuda.tunable


Stream Sanitizer (نموذج أولي)
----------------------------

CUDA Sanitizer هي أداة نموذج أولي للكشف عن أخطاء المزامنة بين التدفقات في
PyTorch. راجع :doc:`الوثائق <cuda._sanitizer>` للحصول على معلومات حول كيفية
استخدامها.

.. toctree::
    :hidden:

    cuda._sanitizer


.. تحتاج هذه الوحدة إلى توثيق. أضيفها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.cuda.comm
.. py:module:: torch.cuda.error
.. py:module:: torch.cuda.gds
.. py:module:: torch.cuda.graphs
.. py:module:: torch.cuda.jiterator
.. py:module:: torch.cuda.memory
.. py:module:: torch.cuda.nccl
.. py:module:: torch.cuda.nvtx
.. py:module:: torch.cuda.profiler
.. py:module:: torch.cuda.random
.. py:module:: torch.cuda.sparse
.. py:module:: torch.cuda.streams