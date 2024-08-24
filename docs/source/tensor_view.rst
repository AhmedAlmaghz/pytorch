.. currentmodule:: torch

.. _tensor-view-doc:

عرض الموتينات (Tensor Views)
============================

يسمح PyTorch بأن يكون الموتينور (tensor) عرضاً (View) لموتينور موجود مسبقاً. يشارك موتينور العرض نفس البيانات الأساسية مع الموتينور الأساسي الخاص به. وبدعم العرض، يتم تجنب النسخ الصريح للبيانات، مما يسمح لنا بإجراء عمليات إعادة تشكيل وتقطيع وعمليات عنصر-حكيمة سريعة وفعالة في الذاكرة.

على سبيل المثال، للحصول على عرض لموتينور موجود مسبقاً "t"، يمكنك استدعاء "t.view(...)".

::

    >>> t = torch.rand(4, 4)
    >>> b = t.view(2, 8)
    >>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` يتشاركان نفس البيانات الأساسية.
    True
    # تعديل موتينور العرض يغير الموتينور الأساسي أيضاً.
    >>> b[0][0] = 3.14
    >>> t[0][0]
    tensor(3.14)

نظراً لأن العروض تشارك البيانات الأساسية مع الموتينور الأساسي الخاص بها، إذا قمت بتعديل البيانات في العرض، فسيتم عكس ذلك في الموتينور الأساسي أيضاً.

عادةً ما تقوم عملية PyTorch بإرجاع موتينور جديد كناتج، على سبيل المثال: :meth:`~torch.Tensor.add`.
ولكن في حالة عمليات العرض، تكون النواتج عروضاً للموتينورات المدخلة لتجنب نسخ البيانات غير الضروري.
لا يحدث نقل للبيانات عند إنشاء عرض، حيث يقوم موتينور العرض فقط بتغيير طريقة تفسير نفس البيانات. وقد يؤدي أخذ عرض لموتينور متجاور إلى إنتاج موتينور غير متجاور.
يجب على المستخدمين إيلاء اهتمام إضافي حيث قد يكون للتجاور تأثير ضمني على الأداء.
:meth:`~torch.Tensor.transpose` هو مثال شائع على ذلك.

::

    >>> base = torch.tensor([[0, 1],[2, 3]])
    >>> base.is_contiguous()
    True
    >>> t = base.transpose(0, 1)  # `t` هو عرض لـ `base`. لم يحدث نقل للبيانات هنا.
    # قد تكون موتينورات العرض غير متجاورة.
    >>> t.is_contiguous()
    False
    # للحصول على موتينور متجاور، قم باستدعاء `.contiguous()` لفرض
    # نسخ البيانات عندما لا يكون `t` متجاوراً.
    >>> c = t.contiguous()

وللإشارة، إليك قائمة كاملة بعمليات العرض في PyTorch:

- عملية التقطيع والفهرسة الأساسية، على سبيل المثال "tensor[0, 2:, 1:7:2]" تعيد عرضاً للموتينور الأساسي "tensor"، راجع الملاحظة أدناه.
- :meth:`~torch.Tensor.adjoint`
- :meth:`~torch.Tensor.as_strided`
- :meth:`~torch.Tensor.detach`
- :meth:`~torch.Tensor.diagonal`
- :meth:`~torch.Tensor.expand`
- :meth:`~torch.Tensor.expand_as`
- :meth:`~torch.Tensor.movedim`
- :meth:`~torch.Tensor.narrow`
- :meth:`~torch.Tensor.permute`
- :meth:`~torch.Tensor.select`
- :meth:`~torch.Tensor.squeeze`
- :meth:`~torch.Tensor.transpose`
- :meth:`~torch.Tensor.t`
- :attr:`~torch.Tensor.T`
- :attr:`~torch.Tensor.H`
- :attr:`~torch.Tensor.mT`
- :attr:`~torch.Tensor.mH`
- :attr:`~torch.Tensor.real`
- :attr:`~torch.Tensor.imag`
- :meth:`~torch.Tensor.view_as_real`
- :meth:`~torch.Tensor.unflatten`
- :meth:`~torch.Tensor.unfold`
- :meth:`~torch.Tensor.unsqueeze`
- :meth:`~torch.Tensor.view`
- :meth:`~torch.Tensor.view_as`
- :meth:`~torch.Tensor.unbind`
- :meth:`~torch.Tensor.split`
- :meth:`~torch.Tensor.hsplit`
- :meth:`~torch.Tensor.vsplit`
- :meth:`~torch.Tensor.tensor_split`
- :meth:`~torch.Tensor.split_with_sizes`
- :meth:`~torch.Tensor.swapaxes`
- :meth:`~torch.Tensor.swapdims`
- :meth:`~torch.Tensor.chunk`
- :meth:`~torch.Tensor.indices` (لموتينورات متفرقة فقط)
- :meth:`~torch.Tensor.values`  (لموتينورات متفرقة فقط)

.. note::
   عند الوصول إلى محتويات موتينور عبر الفهرسة، يتبع PyTorch سلوكيات Numpy التي تعيد الفهرسة الأساسية عروضاً، في حين تعيد الفهرسة المتقدمة نسخة.
   التخصيص عبر الفهرسة الأساسية أو المتقدمة يكون في المكان. راجع المزيد من الأمثلة في
   `وثائق فهرسة Numpy <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.

من الجدير بالذكر أيضاً أن هناك بعض العمليات ذات السلوكيات الخاصة:

- :meth:`~torch.Tensor.reshape`، :meth:`~torch.Tensor.reshape_as` و :meth:`~torch.Tensor.flatten` يمكن أن تعيد إما عرض أو موتينور جديد، وينبغي لرمز المستخدم ألا يعتمد على ما إذا كان عرضاً أم لا.
- :meth:`~torch.Tensor.contiguous` تعيد **نفسها** إذا كان الموتينور المدخل متجاوراً بالفعل، وإلا فإنها تعيد موتينور متجاوراً جديداً عن طريق نسخ البيانات.

للحصول على دليل تفصيلي أكثر لتنفيذ PyTorch الداخلي،
يرجى الرجوع إلى `منشور مدونة ezyang حول داخليات PyTorch <http://blog.ezyang.com/2019/05/pytorch-internals/>`_.