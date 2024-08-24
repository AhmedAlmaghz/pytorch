torch.func جولة سريعة
====================

ما هو torch.func؟
------------------

.. currentmodule:: torch.func

torch.func، المعروف سابقًا باسم functorch، هو مكتبة لـ
`JAX <https://github.com/google/jax>`_-like composable function transforms في
PyTorch.

- "تحويل الدالة" هو دالة من الدرجة العليا تقبل دالة رقمية وتعيد دالة جديدة تحسب كمية مختلفة.
- يحتوي torch.func على تحويلات التفاضل التلقائي (تعيد ``grad(f)`` دالة تحسب تدرج ``f``)، وتحويل vectorization/batching (تعيد ``vmap(f)`` دالة تحسب ``f`` عبر دفعات من المدخلات)، وغيرها.
- يمكن لهذه الدوال التحويلية أن تتركب مع بعضها البعض بشكل تعسفي. على سبيل المثال، يحسب تكوين ``vmap(grad(f))`` كمية تسمى per-sample-gradients لا يمكن لبرنامج PyTorch الأساسي حسابها بكفاءة اليوم.

لماذا التحويلات الوظيفية القابلة للتكوين؟
----------------------------
هناك عدد من حالات الاستخدام التي يصعب تنفيذها في PyTorch اليوم:
- حساب per-sample-gradients (أو غيرها من الكميات لكل عينة)

- تشغيل مجموعات من النماذج على آلة واحدة
- تجميع المهام بكفاءة في الحلقة الداخلية لـ MAML
- حساب المصفوفات المشتقة والهيسية بكفاءة
- حساب المصفوفات المشتقة والهيسية المجمعة بكفاءة

يسمح تكوين تحويلات :func:`vmap` و :func:`grad` و :func:`vjp` و :func:`jvp` بالتعبير عن ما سبق دون تصميم نظام فرعي منفصل لكل منها.

ما هي التحويلات؟
-------------

:func:`grad` (حساب التدرج)
^^^^^^^^^^^^^^^^^^^^^^^^^^

``grad(func)`` هو تحويل حساب التدرج لدينا. إنه يعيد دالة جديدة
تحسب تدرجات ``func``. يفترض أن ``func`` تعيد Tensor ذو عنصر واحد
وبشكل افتراضي يحسب تدرجات إخراج ``func`` فيما يتعلق
بالمدخلات الأولى.

.. code-block:: python

    import torch
    from torch.func import grad
    x = torch.randn([])
    cos_x = grad(lambda x: torch.sin(x))(x)
    assert torch.allclose(cos_x, x.cos())

    # تدرجات من الدرجة الثانية
    neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
    assert torch.allclose(neg_sin_x, -x.sin())

:func:`vmap` (auto-vectorization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ملاحظة: يفرض :func:`vmap` قيودًا على الكود الذي يمكن استخدامه. لمزيد من
التفاصيل، يرجى الاطلاع على :ref:`ux-limitations`.

``vmap(func)(*inputs)`` هو تحويل يضيف بُعدًا إلى جميع عمليات Tensor
في ``func``. ``vmap(func)`` تعيد دالة جديدة تقوم بتعيين ``func``
عبر بُعد (افتراضي: 0) لكل Tensor في المدخلات.

vmap مفيد لإخفاء أبعاد الدفعات: يمكن للمرء كتابة دالة func تعمل
على أمثلة ثم رفعها إلى دالة يمكنها التعامل مع دفعات من
الأمثلة باستخدام ``vmap(func)``، مما يؤدي إلى تجربة نمذجة أبسط:

.. code-block:: python

    import torch
    from torch.func import vmap
    batch_size, feature_size = 3, 5
    weights = torch.randn(feature_size, requires_grad=True)

    def model(feature_vec):
        # نموذج خطي بسيط جدًا مع تنشيط
        assert feature_vec.dim() == 1
        return feature_vec.dot(weights).relu()

    examples = torch.randn(batch_size, feature_size)
    result = vmap(model)(examples)

عند تكوينه مع :func:`grad`، يمكن استخدام :func:`vmap` لحساب per-sample-gradients:

.. code-block:: python

    from torch.func import vmap
    batch_size, feature_size = 3, 5

    def model(weights,feature_vec):
        # نموذج خطي بسيط جدًا مع تنشيط
        assert feature_vec.dim() == 1
        return feature_vec.dot(weights).relu()

    def compute_loss(weights, example, target):
        y = model(weights, example)
        return ((y - target) ** 2).mean()  # MSELoss

    weights = torch.randn(feature_size, requires_grad=True)
    examples = torch.randn(batch_size, feature_size)
    targets = torch.randn(batch_size)
    inputs = (weights,examples, targets)
    grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)

:func:`vjp` (منتج المصفوفة المشتقة المتجهة)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يطبق تحويل :func:`vjp` ``func`` على ``inputs`` ويعيد دالة جديدة
تحسب منتج المصفوفة المشتقة المتجهة (vjp) نظرًا لبعض ``cotangents`` Tensors.

.. code-block:: python

    from torch.func import vjp

    inputs = torch.randn(3)
    func = torch.sin
    cotangents = (torch.randn(3),)

    outputs, vjp_fn = vjp(func, inputs); vjps = vjp_fn(*cotangents)

:func:`jvp` (منتج المصفوفة المشتقة المتجهة)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يحسب تحويل :func:`jvp` منتجات المصفوفة المشتقة المتجهة وهو معروف أيضًا باسم
"التفاضل التلقائي للأمام". إنه ليس دالة من الدرجة العليا على عكس معظم التحويلات الأخرى،
ولكنه يعيد مخرجات ``func(inputs)`` وكذلك jvps.

.. code-block:: python

    from torch.func import jvp
    x = torch.randn(5)
    y = torch.randn(5)
    f = lambda x, y: (x * y)
    _, out_tangent = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
    assert torch.allclose(out_tangent, x + y)

:func:`jacrev` و :func:`jacfwd` و :func:`hessian`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يعيد تحويل :func:`jacrev` دالة جديدة تأخذ ``x`` وتعيد
المصفوفة المشتقة للدالة فيما يتعلق بـ ``x`` باستخدام التفاضل العكسي.

.. code-block:: python

    from torch.func import jacrev
    x = torch.randn(5)
    jacobian = jacrev(torch.sin)(x)
    expected = torch.diag(torch.cos(x))
    assert torch.allclose(jacobian, expected)

يمكن تكوين :func:`jacrev` مع :func:`vmap` لإنتاج المصفوفات المشتقة المجمعة:

.. code-block:: python

    x = torch.randn(64, 5)
    jacobian = vmap(jacrev(torch.sin))(x)
    assert jacobian.shape == (64, 5, 5)

:func:`jacfwd` هو بديل مباشر لـ :func:`jacrev` يحسب المصفوفات المشتقة باستخدام
التفاضل الأمامي:

.. code-block:: python

    from torch.func import jacfwd
    x = torch.randn(5)
    jacobian = jacfwd(torch.sin)(x)
    expected = torch.diag(torch.cos(x))
    assert torch.allclose(jacobian, expected)

يمكن تكوين :func:`jacrev` مع نفسه أو :func:`jacfwd` لإنتاج الهيسية:

.. code-block:: python

    def f(x):
        return x.sin().sum()

    x = torch.randn(5)
    hessian0 = jacrev(jacrev(f))(x)
    hessian1 = jacfwd(jacrev(f))(x)

:func:`hessian` هو دالة ملائمة تجمع بين :func:`jacfwd` و :func:`jacrev`:

.. code-block:: python

    from torch.func import hessian

    def f(x):
        return x.sin().sum()

    x = torch.randn(5)
    hess = hessian(f)(x)