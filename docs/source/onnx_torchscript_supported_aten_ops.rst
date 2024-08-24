:orphan:

مشغلو TorchScript المدعومة من ONNX
====================================

.. يتم إنشاء هذا الملف تلقائيًا أثناء بناء الوثائق
.. عن طريق المراجعة المرجعية لرموز مشغل ONNX مع مشغلي TorchScript عبر
.. "scripts/build_onnx_torchscript_supported_aten_op_csv_table.py".
.. لا تعدل هذا الملف مباشرة وقم بدلاً من ذلك `بإعادة بناء الوثائق <https://github.com/pytorch/pytorch#building-the-documentation>`_.

تسرد هذه الصفحة مشغلي TorchScript المدعومين/غير المدعومين من قبل ONNX export.

المشغلون المدعومون
---------------

.. csv-table:: دعم ONNX لمشغلي TorchScript
   :file: ../build/onnx/auto_gen_supported_op_list.csv
   :widths: 70, 30
   :header-rows: 1


المشغلون غير المدعومون
-------------------

المشغلون الذين لم يتم دعمهم بعد

.. csv-table:: المشغلون غير المدعومون
   :file: ../build/onnx/auto_gen_unsupported_op_list.csv
   :widths: 70, 30
   :header-rows: 1