.. currentmodule:: torch.cuda.tunable

TunableOp
=========

.. note::
    هذه ميزة تجريبية، مما يعني أنها في مرحلة مبكرة لتلقي الملاحظات والاختبارات، وقد تتغير مكوناتها.

نظرة عامة
--------

.. automodule:: torch.cuda.tunable

مرجع واجهة برمجة التطبيقات
-------------------

.. autofunction:: enable
.. autofunction:: is_enabled
.. autofunction:: tuning_enable
.. autofunction:: tuning_is_enabled
.. autofunction:: set_max_tuning_duration
.. autofunction:: get_max_tuning_duration
.. autofunction:: set_max_tuning_iterations
.. autofunction:: get_max_tuning_iterations
.. autofunction:: set_filename
.. autofunction:: get_filename
.. autofunction:: get_results
.. autofunction:: get_validators
.. autofunction:: write_file_on_exit
.. autofunction:: write_file
.. autofunction:: read_file