.. _torch-library-docs:

torch.library
===============
.. py:module:: torch.library
.. currentmodule:: torch.library

تمثل torch.library مجموعة من واجهات برمجة التطبيقات (APIs) لتوسيع المكتبة الأساسية لـ PyTorch من المشغلات. تحتوي على برامج مساعدة لاختبار المشغلات المخصصة، وإنشاء مشغلات مخصصة جديدة، وتوسيع المشغلات المحددة باستخدام واجهات برمجة تطبيقات تسجيل مشغل C++ الخاصة بـ PyTorch (مثل مشغلات aten).

للحصول على دليل مفصل حول الاستخدام الفعال لهذه الواجهات، يرجى الاطلاع على:

يرجى الاطلاع على :ref:`custom-ops-landing-page` للحصول على مزيد من التفاصيل حول كيفية الاستخدام الفعال لهذه الواجهات.

اختبار المشغلات المخصصة
------------------

استخدم :func:`torch.library.opcheck` لاختبار المشغلات المخصصة للبحث عن الاستخدام غير الصحيح لواجهة برمجة تطبيقات Python torch.library و/أو واجهات برمجة تطبيقات C++ TORCH_LIBRARY. أيضًا، إذا كان مشغل التدريب الخاص بك يدعم التدريب، فاستخدم :func:`torch.autograd.gradcheck` لاختبار دقة التدرجات رياضيًا.

.. autofunction:: opcheck

إنشاء مشغلات مخصصة جديدة في Python
------------------------------

استخدم :func:`torch.library.custom_op` لإنشاء مشغلات مخصصة جديدة.

.. autofunction:: custom_op

توسيع المشغلات المخصصة (المنشأة من Python أو C++)
------------------------------------------

استخدم طرق register.*، مثل :func:`torch.library.register_kernel` و:func:`torch.library.register_fake`، لإضافة عمليات تنفيذ لأي مشغلات (قد تكون تم إنشاؤها باستخدام :func:`torch.library.custom_op` أو عبر واجهات برمجة تطبيقات تسجيل مشغل C++ الخاصة بـ PyTorch).

.. autofunction:: register_kernel
.. autofunction:: register_autograd
.. autofunction:: register_fake
.. autofunction:: register_vmap
.. autofunction:: impl_abstract
.. autofunction:: get_ctx
.. autofunction:: register_torch_dispatch
.. autofunction:: infer_schema
.. autoclass:: torch._library.custom_ops.CustomOpDef

    .. automethod:: set_kernel_enabled

واجهات برمجة التطبيقات منخفضة المستوى
------------------------------

تمثل واجهات برمجة التطبيقات التالية روابط مباشرة إلى واجهات برمجة التطبيقات منخفضة المستوى لتسجيل المشغل C++ الخاصة بـ PyTorch.

.. warning::
   تعد واجهات برمجة التطبيقات لتسجيل المشغل منخفض المستوى ومبدل PyTorch مفهومًا معقدًا في PyTorch. نوصي باستخدام واجهات برمجة التطبيقات عالية المستوى المذكورة أعلاه (التي لا تتطلب كائن torch.library.Library) كلما أمكن ذلك. تمثل هذه التدوينة <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ نقطة انطلاق جيدة لمعرفة المزيد حول مبدل PyTorch.

يتوفر البرنامج التعليمي الذي يشرح لك بعض الأمثلة حول كيفية استخدام هذه الواجهة على `Google Colab <https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing>`_.

.. autoclass:: torch.library.Library
  :members:

.. autofunction:: fallthrough_kernel

.. autofunction:: define

.. autofunction:: impl