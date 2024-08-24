torch.mtia
=============

تم تنفيذ backend MTIA خارج الشجرة، ويتم تعريف الواجهات هنا فقط.

.. automodule:: torch.mtia
.. currentmodule:: torch.mtia

.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    current_device
    current_stream
    default_stream
    device_count
    init
    is_available
    is_initialized
    memory_stats
    set_device
    set_stream
    stream
    synchronize
    device
    set_rng_state
    get_rng_state
    DeferredMtiaCallError

التدفقات والأحداث
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream