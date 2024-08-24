.. currentmodule:: torch

.. _type-info-doc:

معلومات النوع
=============

يمكن الوصول إلى الخصائص العددية لـ :class:`torch.dtype` إما من خلال :class:`torch.finfo` أو :class:`torch.iinfo`.

.. _finfo-doc:

torch.finfo
-----------

.. class:: torch.finfo

:class:`torch.finfo` هو كائن يمثل الخصائص العددية لنوع النقطة العائمة
:class:`torch.dtype`، (أي ``torch.float32``، ``torch.float64``، ``torch.float16``، و ``torch.bfloat16``). وهذا مشابه لـ `numpy.finfo <https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html>`_.

يوفر :class:`torch.finfo` السمات التالية:

===============        =====   ==========================================================================
الاسم                   النوع    الوصف
===============        =====   ==========================================================================
bits                   int     عدد البتات التي يشغلها النوع.
eps                    float   أصغر عدد يمكن تمثيله بحيث ``1.0 + eps != 1.0``.
max                    float   أكبر عدد يمكن تمثيله.
min                    float   أصغر عدد يمكن تمثيله (عادةً ``-max``).
tiny                   float   أصغر عدد طبيعي موجب. مكافئ لـ ``smallest_normal``.
smallest_normal        float   أصغر عدد طبيعي موجب. راجع الملاحظات.
resolution             float   الدقة العشرية التقريبية لهذا النوع، أي ``10**-precision``.
===============        =====   ==========================================================================

.. note::
  يمكن استدعاء منشئ :class:`torch.finfo` بدون وسيط، وفي هذه الحالة يتم إنشاء الفئة لنوع pytorch الافتراضي (كما هو موضح بواسطة :func:`torch.get_default_dtype`).

.. note::
  تعيد `smallest_normal` أصغر عدد *طبيعي*، ولكن هناك أعداد أصغر
  تسمى subnormal. راجع https://en.wikipedia.org/wiki/Denormal_number
  لمزيد من المعلومات.

.. _iinfo-doc:

torch.iinfo
-----------

.. class:: torch.iinfo

:class:`torch.iinfo` هو كائن يمثل الخصائص العددية لنوع صحيح
:class:`torch.dtype` (أي ``torch.uint8``، ``torch.int8``، ``torch.int16``، ``torch.int32``، و ``torch.int64``). وهذا مشابه لـ `numpy.iinfo <https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html>`_.

يوفر :class:`torch.iinfo` السمات التالية:

=========   =====   ========================================
الاسم        النوع    الوصف
=========   =====   ========================================
bits        int     عدد البتات التي يشغلها النوع.
max         int     أكبر عدد يمكن تمثيله.
min         int     أصغر عدد يمكن تمثيله.
=========   =====   ========================================