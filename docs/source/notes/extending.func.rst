.. _func-autograd-function:

توسيع torch.func باستخدام autograd.Function
===========================================

.. currentmodule:: torch.autograd

ربما ترغب في استخدام :class:`torch.autograd.Function` مع تحويلات :mod:`torch.func`
مثل :func:`torch.vmap`، :func:`torch.func.grad`، وما إلى ذلك.

هناك حالتان استخدام رئيسيتان:

- ترغب في استدعاء كود لا يحتوي على عمليات PyTorch وجعله يعمل مع تحويلات الدالة. أي أن :class:`torch.autograd.Function`'s
  forward/backward/etc يستدعي وظائف من أنظمة أخرى مثل C++، CUDA، numpy.
- ترغب في تحديد قواعد تدرج مخصصة، مثل
  `custom_vjp/custom_jvp <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_ من JAX

يضم PyTorch كلا المفهومين في :class:`torch.autograd.Function`.

الاستخدام الأساسي
-----------

يفترض هذا الدليل أنك على دراية بـ :ref:`extending-autograd`،
الذي يشرح كيفية استخدام :class:`torch.autograd.Function`.

يمكن أن يكون لـ :class:`torch.autograd.Function` إما :meth:`~Function.forward` الذي يقبل كائن ctx،
أو يمكن أن يكون له :meth:`~Function.forward` منفصل (الذي لا يقبل ``ctx``) و :meth:`~Function.setup_context`
staticmethod الذي يعدل كائن ``ctx``.

يتم دعم الأخير فقط مع تحويلات الوظائف:

- :meth:`~Function.forward` هو الكود الذي يؤدي العملية ويجب ألا يقبل
  كائن ``ctx``.
- ``setup_context(ctx، inputs، output)`` هو الكود الذي يمكنك من خلاله
  استدعاء الطرق على ``ctx``. هنا حيث يجب عليك حفظ المنسوجات للخلف
  (عن طريق استدعاء ``ctx.save_for_backward(*tensors)``)، أو حفظ غير المنسوجات
  عن طريق تعيينها على كائن ``ctx``.

بما أن :meth:`~Function.setup_context` يقبل فقط ``inputs`` و ``output``،
الكميات الوحيدة التي يمكن حفظها هي إما كائنات (مثل المنسوجات) في
الإدخالات أو الإخراج أو الكميات (مثل ``Tensor.shape``) المستمدة منها.
إذا كنت ترغب في حفظ تنشيط وسيط غير إدخال من
:meth:`Function.forward` للخلف، ثم ستحتاج إلى إعادته كإخراج
من :meth:`~Function.forward` حتى يتم تمريره إلى
:meth:`~Function.setup_context`.

اعتمادًا على التحويل،

- لدعم AD باتجاه عكسي (:func:`torch.func.grad`، :func:`torch.func.vjp`)،
  يحتاج :class:`torch.autograd.Function` إلى :meth:`~Function.backward` staticmethod.
- لدعم :func:`torch.vmap`، يحتاج :class:`torch.autograd.Function` إلى :meth:`~Function.vmap` staticmethod.
- لدعم :func:`torch.func.jvp`، يحتاج :class:`torch.autograd.Function` إلى :meth:`~Function.jvp` staticmethod.
- لدعم تركيبات من التحويلات (مثل :func:`torch.func.jacrev`،
  :func:`torch.func.jacfwd`، :func:`torch.func.hessian`) -- قد تحتاج إلى عدة
  من ما سبق.

لجعل :class:`torch.autograd.Function` قابلًا للتكوين بشكل تعسفي مع تحويلات الدالة، نوصي بأن تكون جميع الطرق الثابتة الأخرى بخلاف :meth:`~Function.forward` و
:meth:`~Function.setup_context` قابلة للتحويل: أي أنها يجب أن تتكون فقط من مشغلي PyTorch أو استدعاء :class:`torch.autograd.Function` آخر (قد يستدعي C++/CUDA/etc).

دعونا نلقي نظرة على بعض الأمثلة على حالات الاستخدام الشائعة.

المثال 1: autograd.Function يستدعي نظامًا آخر
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

الحالة الشائعة هي :class:`torch.autograd.Function` مع كل من forward() و backward() استدعاء
إلى نظام آخر (مثل C++، CUDA، numpy، triton).

::

    import torch
    import numpy as np

    def to_numpy(tensor):
        return tensor.cpu().numpy()

    class NumpySort(torch.autograd.Function):
        # لاحظ أن forward لا تأخذ ctx
        @staticmethod
        def forward(x، dim):
            الجهاز = x.device
            x = to_numpy(x)
            ind = np.argsort(x، axis=dim)
            ind_inv = np.argsort(ind، axis=dim)
            النتيجة = np.take_along_axis(x، ind، axis=dim)
            # يجب إرجاع أي وسيطات مؤقتة يتم حفظها في الخلف كـ
            # الإخراج.
            return (
                # الإخراج المطلوب
                torch.tensor(result، device=device)،
                # الوسيط المؤقت لحفظ الخلف
                torch.tensor(ind، device=device)،
                # الوسيط المؤقت لحفظ الخلف
                torch.tensor(ind_inv، device=device)،
            )

        # setup_context مسؤول عن استدعاء الطرق و/أو تعيين كائن
        # ctx. يرجى عدم إجراء حسابات إضافية (مثل إضافة
        # المنسوجات معًا) في setup_context.
        @staticmethod
        def setup_context(ctx، inputs، output):
            x، dim = inputs
            # لاحظ أن الإخراج هو ما قمت بإرجاعه من forward.
            # إذا قمت بإرجاع قيم متعددة، فإن الإخراج عبارة عن Tuple من قيم متعددة.
            # إذا قمت بإرجاع Tensor واحد، فإن الإخراج هو Tensor.
            # إذا قمت بإرجاع Tuple مع Tensor واحد، فإن الإخراج عبارة عن
            # Tuple مع Tensor واحد.
            _، ind، ind_inv = output
            ctx.mark_non_differentiable(ind، ind_inv)
            # يجب حفظ المنسوجات عبر ctx.save_for_backward. يرجى عدم
            # تعيينها مباشرة على كائن ctx.
            ctx.save_for_backward(ind، ind_inv)
            # يمكن حفظ غير المنسوجات عن طريق تعيينها كسمات على كائن ctx.
            ctx.dim = dim

        @staticmethod
        def backward(ctx، grad_output، _0، _1):
            # لكي يكون autograd.Function قابلاً للتكوين بشكل تعسفي مع تحويلات الدالة
            # يجب تنفيذ جميع الطرق الثابتة الأخرى بخلاف forward و setup_context
            # بطريقة "قابلة للتحويل"؛ أي أنها يجب أن
            # تتكون فقط من عمليات PyTorch أو autograd.Function.
            #
            # على سبيل المثال، يسمح لنا ذلك بإجراء عمليات التراجع المزدوج و/أو حساب
            # المشتقات من الدرجة الثانية.
            #
            # لقد كتبنا تمريرة الخلف لـ NumpySort من حيث autograd.Function آخر، NumpyTake.
            ind، ind_inv = ctx.saved_tensors
            return NumpyTake.apply(grad_output، ind_inv، ind، ctx.dim)، None

    class NumpyTake(torch.autograd.Function):
        @staticmethod
        def forward(x، ind، ind_inv، dim):
            الجهاز = x.device
            x = to_numpy(x)
            ind = to_numpy(ind)
            return torch.tensor(np.take_along_axis(x، ind، dim)، device=device)

        @staticmethod
        def setup_context(ctx، inputs، output):
            x، ind، ind_inv، dim = inputs
            ctx.save_for_backward(ind، ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx، grad_output):
            ind، ind_inv = ctx.saved_tensors
            النتيجة = NumpyTake.apply(grad_output، ind_inv، ind، ctx.dim)
            return النتيجة، None، None، None


الآن، لتسهيل استخدام ``NumpySort`` (لإخفاء الوسيطات المؤقتة التي
أعدناها كإخراج، وكذلك السماح بالوسيطات الافتراضية والوسيطات المتغيرة)، نقوم بإنشاء دالة جديدة تستدعيها::

    def numpy_sort(x، dim=-1):
        result، _، _ = NumpySort.apply(x، dim)
        return result

وهنا فحص السلامة::

    x = torch.randn(2، 3)
    grad_x = torch.func.grad(lambda x: numpy_sort(x).sum())(x)
    التأكيد على أن torch.allclose(grad_x، torch.ones_like(x))



المثال 2: autograd.Function يحدد قواعد تدرج مخصصة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

الحالة الشائعة الأخرى هي :class:`torch.autograd.Function` الذي يتم تنفيذه باستخدام عمليات PyTorch. يمكن لـ PyTorch حساب التدرجات لعمليات PyTorch تلقائيًا،
ولكن ربما نرغب في تخصيص كيفية حساب التدرجات. بعض الأسباب التي قد نرغب في الحصول على تدرج مخصص مختلف عما يقدمه PyTorch هي:

- تحسين الاستقرار العددي
- تغيير خصائص الأداء للخلف
- تغيير كيفية التعامل مع حالات الحواف (على سبيل المثال، nans، inf)
- تعديل التدرج (على سبيل المثال، قص التدرج)

فيما يلي مثال على :class:`torch.autograd.Function` للدالة ``y = x ** 3`` حيث نقوم
تغيير خصائص الأداء (يحدث بعض الحساب الذي يحدث عادة
أثناء تمريرة الخلف، يتم حساب dx، في تمريرة للأمام).

::

  class MyCube(torch.autograd.Function):
      @staticmethod
      def forward(x):
          result = x ** 3
          # في PyTorch العادي، إذا كنا قد قمنا ببساطة بتشغيل y = x ** 3، فإن تمريرة الخلف
          # يحسب dx = 3 * x ** 2. في هذا autograd.Function، لقد قمنا
          # هذا الحساب هنا في تمريرة للأمام بدلاً من ذلك.
          dx = 3 * x ** 2
          return result، dx

      @staticmethod
      def setup_context(ctx، inputs، output):
          x، = inputs
          result، dx = output
          ctx.save_for_backward(x، dx)

      @staticmethod
      def backward(ctx، grad_output، grad_dx):
          x، dx = ctx.saved_tensors
          # لكي يعمل autograd.Function مع المشتقات من الدرجة العليا
          # يجب علينا إضافة مساهمة التدرج من `dx`.
          النتيجة = grad_output * dx + grad_dx * 6 * x
          return result

الآن، لتسهيل استخدام ``NumpySort`` (وإخفاء الوسيطات المؤقتة التي
أعدناها كإخراج) نقوم بإنشاء دالة جديدة تستدعيها::

    def my_cube(x):
        result، _ = MyCube.apply(x)
        return result

فيما يلي فحص سلامة لحساب المشتقات من الدرجة الثانية::

    x = torch.randn([])
    ggx = torch.func.grad(torch.func.grad(my_cube))(x)
    التأكيد على أن torch.allclose(ggx، 6 * x)

القيود والمشكلات
^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

    يرجى قراءة هذه القيود المفروضة على :class:`torch.autograd.Function` مع تحويلات torch.func
    بعناية. لا يمكننا التقاط العديد من هذه المواقف والخطأ بسلاسة
    لذلك سوف يؤدي إلى سلوك غير محدد.

يرجى عدم التقاط المنسوجات التي يتم تحويلها، يكون لها
requires_grad=True، أو هي المنسوجات المزدوجة، في طرق
:class:`torch.autograd.Function`. الطريقة الآمنة تمامًا هي التأكد من أن المنسوجات الوحيدة المستخدمة داخل أي طريقة من :class:`torch.autograd.Function` يجب أن يتم تمريرها مباشرة كإدخالات (أو عبر كائن ctx) بدلاً من المجيء من خارج
:class:`torch.autograd.Function`.

:class:`torch.autograd.Function` لا يتعامل مع المنسوجات في pytrees (هياكل البيانات المتداخلة التعسفية التي قد تحتوي أو لا تحتوي على المنسوجات).
لتتبع هذه المنسوجات بواسطة autograd، يجب تمريرها مباشرة كـ
حجة إلى :class:`torch.autograd.Function`. هذا على النقيض من
jax.{custom_vjp، custom_jvp}، والتي تقبل pytrees.

يرجى استخدام :meth:`~torch.autograd.function.FunctionCtx.save_for_backward` أو
:meth:`~torch.autograd.function.FunctionCtx.save_for_forward` لحفظ المنسوجات فقط.
يرجى عدم تعيين المنسوجات أو مجموعات المنسوجات مباشرة على كائن ctx -
لن يتم تتبع هذه المنسوجات


دعم :func:`torch.vmap`
لاستخدام :class:`torch.autograd.Function` مع :func:`torch.vmap`، يجب عليك إما:

- توفير :meth:`~Function.vmap` طريقة ثابتة تخبرنا بسلوك :class:`torch.autograd.Function`
  تحت :func:`torch.vmap`
- اطلب منا توليدها تلقائيًا عن طريق تعيين ``generate_vmap_rule=True``.

توليد قاعدة vmap تلقائيًا
^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا استوفت :class:`torch.autograd.Function` الخاص بك القيود الإضافية التالية، فسنتمكن
من إنشاء قاعدة vmap لها. إذا لم يستوف القيود أو إذا كنت
تريد سلوكًا مخصصًا في vmap، فيرجى تعريف طريقة vmap ثابتة يدويًا (راجع القسم التالي).

.. warning::

     لا يمكننا التحقق بسهولة من القيود التالية والخروج
     بسلاسة. قد يؤدي انتهاك القيود إلى سلوك غير محدد.

- يجب أن تكون :meth:`~Function.forward` لـ :class:`torch.autograd.Function`، :meth:`~Function.backward` (إذا كان موجودًا) و:meth:`~Function.jvp`
  (إذا كان موجودًا) يمكن تحويلها عبر :func:`torch.vmap`. أي
  يجب أن تتكون فقط من عمليات PyTorch (على عكس على سبيل المثال NumPy أو CUDA
  نواة مخصصة).

مثال::

    class MyCube(torch.autograd.Function):
        # Set generate_vmap_rule to True to ask PyTorch to automatically generate
        # a vmap rule.
        generate_vmap_rule = True

        @staticmethod
        def forward(x):
            result = x ** 3
            dx = 3 * x ** 2
            return result, dx

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, = inputs
            result, dx = output
            ctx.save_for_backward(x, dx)

        @staticmethod
        def backward(ctx, grad_output, grad_dx):
            x, dx = ctx.saved_tensors
            result = grad_output * dx + grad_dx * 6 * x
            return result

    def my_cube(x):
        result, dx = MyCube.apply(x)
        return result

    x = torch.randn(3)
    result = torch.vmap(my_cube)(x)
    assert torch.allclose(result, x ** 3)


تعريف طريقة vmap الثابتة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

إذا كانت :class:`torch.autograd.Function` تستدعي نظامًا آخر (مثل NumPy أو C++ أو CUDA أو Triton)،
ثم لجعله يعمل مع :func:`torch.vmap` أو التحويلات التي تستخدمها، ستحتاج
لتحديد طريقة :meth:`~Function.vmap` ثابتة يدويًا.

اعتمادًا على التحويلات التي تريد استخدامها وحالتك الاستخدامية، قد لا تحتاج
إلى إضافة :meth:`~Function.vmap` طريقة ثابتة إلى جميع :class:`torch.autograd.Function`:

- على سبيل المثال، يؤدي :func:`torch.func.jacrev` إلى :func:`~torch.vmap` عبر تمريرة الخلف.
  لذا إذا كنت مهتمًا فقط باستخدام :func:`torch.func.jacrev`، فقط
  يجب أن تكون طريقة :meth:`~Function.backward` ثابتة قابلة للتحويل إلى vmap.

نوصي بالتأكد من أن جميع :class:`torch.autograd.Function` لديكم دعم لـ
:func:`torch.vmap` على الرغم من ذلك، خاصة إذا كنت تكتب مكتبة تابعة لجهة خارجية وتريد :class:`torch.autograd.Function`
للعمل مع جميع مجموعات :func:`torch.func` من التحويلات.

مفاهيمياً، الطريقة الثابتة vmap مسؤولة عن تحديد كيفية سلوك :meth:`~Function.forward`
تحت :func:`torch.vmap`. أي، فهو يحدد كيفية تحويل
:meth:`~Function.forward` لتشغيله على مدخلات ذات بُعد إضافي (البعد
يتم إجراء vmap عليه). هذا مشابه لكيفية تنفيذ :func:`torch.vmap` على
عمليات PyTorch: لكل عملية، نحدد قاعدة vmap (يشار إليها أحيانًا أيضًا باسم "قاعدة التجميع").

فيما يلي كيفية تحديد :meth:`~Function.vmap` الطريقة الثابتة:

- التوقيع هو ``vmap(info, in_dims: Tuple[Optional[int]], *args)``، حيث
  ``*args`` هو نفسه مثل args إلى :meth:`~Function.forward`.
- الطريقة الثابتة vmap مسؤولة عن تحديد كيفية سلوك :meth:`~Function.forward`
  في ظل :func:`torch.vmap`. أي، بالنظر إلى المدخلات ذات البعد الإضافي
  (محدد بواسطة ``in_dims``)، كيف نحسب الإصدار المجمع من :meth:`~Function.forward`؟
- لكل arg في ``args``، يحتوي ``in_dims`` على ``Optional[int]`` المقابل.
  إنه ``None`` إذا لم يكن arg عبارة عن Tensor أو إذا لم يتم إجراء vmap على arg،
  وإلا، فهو رقم صحيح يحدد البعد في Tensor الذي يتم إجراء vmap عليه.
- ``info`` عبارة عن مجموعة من البيانات الوصفية الإضافية التي قد تكون مفيدة:
  يحدد ``info.batch_size`` حجم البعد الذي يتم إجراء vmap عليه، بينما
  ``info.randomness`` هو خيار "randomness" الذي تم تمريره إلى :func:`torch.vmap`.
- تكون نتيجة الطريقة الثابتة vmap عبارة عن زوج مرتب من ``(output, out_dims)``. مشابه
  لـ ``in_dims``، يجب أن يكون لـ ``out_dims`` نفس بنية ``output`` وأن يحتوي
  على ``out_dim`` واحد لكل إخراج يحدد ما إذا كان الإخراج يحتوي على البعد المحدد
  وما رقمه.


مثال::

    def to_numpy(tensor):
        return tensor.cpu().numpy()

    class NumpySort(torch.autograd.Function):
        @staticmethod
        def forward(x, dim):
            device = x.device
            x = to_numpy(x)
            ind = np.argsort(x, axis=dim)
            ind_inv = np.argsort(ind, axis=dim)
            result = np.take_along_axis(x, ind, axis=dim)
            return (
                torch.tensor(result, device=device),
                torch Mieczyslaw,
                torch.tensor(ind_inv, device=device),
            )

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, dim = inputs
            _, ind, ind_inv = output
            ctx.mark_non_differentiable(ind, ind_inv)
            ctx.save_for_backward(ind, ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output, _0, _1):
            ind, ind_inv = ctx.saved_tensors
            return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

        # The signature of the vmap staticmethod is:
        # vmap(info, in_dims: Tuple[Optional[int]], *args)
        # where *args is the same as the arguments to `forward`.
        @staticmethod
        def vmap(info, in_dims, x, dim):
            # For every input (x and dim), in_dims stores an Optional[int]
            # that is:
            # - None if the input is not being vmapped over or if the input
            #   is not a Tensor
            # - an integer if the input is being vmapped over that represents
            #   the index of the dimension being vmapped over.
            x_bdim, _ = in_dims

            # A "vmap rule" is the logic of how to perform the operation given
            # inputs with one additional dimension. In NumpySort, x has an
            # additional dimension (x_bdim). The vmap rule is simply
            # to call NumpySort again but pass it a different `dim`.
            x = x.movedim(x_bdim, 0)
            # Handle negative dims correctly
            dim = dim if dim >= 0 else dim + x.dim() - 1
            result = NumpySort.apply(x, dim + 1)

            # The vmap rule must return a tuple of two things
            # 1. the output. Should be the same amount of things
            #    as returned by the forward().
            # 2. one Optional[int] for each output specifying if each output
            # is being vmapped over, and if so, the index of the
            # dimension being vmapped over.
            #
            # NumpySort.forward returns a Tuple of 3 Tensors. Since we moved the
            # dimension being vmapped over to the front of `x`, that appears at
            # dimension 0 of all outputs.
            # The return is (output, out_dims) -- output is a tuple of 3 Tensors
            # and out_dims is a Tuple of 3 Optional[int]
            return NumpySort.apply(x, dim + 1), (0, 0, 0)

    class NumpyTake(torch.autograd.Function):
        @staticmethod
        def forward(x, ind, ind_inv, dim):
            device = x.device
            x = to_numpy(x)
            ind = to_numpy(ind)
            return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, ind, ind_inv, dim = inputs
            ctx.save_for_backward(ind, ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output):
            ind, ind_inv = ctx.saved_tensors
            result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
            return result, None, None, None

        @staticmethod
        def vmap(info, in_dims, x, ind, ind_inv, dim):
            x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims

            # The strategy is: expand {x, ind, ind_inv} to all have the dimension
            # being vmapped over.
            # Then, call back into NumpyTake(expanded_x, expanded_ind, expanded_ind_inv, new_dim).

            # Handle negative dims by wrapping them to be positive
            logical_dim = x.dim() if x_bdim is None else x_bdim - 1
            dim = dim if dim >= 0 else dim + logical_dim

            def maybe_expand_bdim_at_front(x, x_bdim):
                if x_bdim is None:
                    return x.expand(info.batch_size, *x.shape)
                return x.movedim(x_bdim, 0)

            # If the Tensor doesn't have the dimension being vmapped over,
            # expand it out. Otherwise, move it to the front of the Tensor
            x = maybe_expand_bdim_at_front(x, x_bdim)
            ind = maybe_expand_bdim_at_front(ind, ind_bdim)
            ind_inv = maybe_expand_bdim_at_front(ind_inv, ind_inv_bdim)

            # The return is a tuple (output, out_dims). Since output is a Tensor,
            # then out_dims is an Optional[int] (instead of being a Tuple).
            return NumpyTake.apply(x, ind, ind_inv, dim + 1), 0

    def numpy_sort(x, dim=-1):
        result, _, _ = NumpySort.apply(x, dim)
        return result

    x = torch.randn(2, 3)
    result = torch.vmap(numpy_sort)(x)
    assert torch.allclose(result, numpy_sort(result, 1))


.. note::

    يجب أن تهدف طريقة vmap الثابتة إلى الحفاظ على دلالية
    :class:`~torch.autograd.Function` بأكمله. أي، (كود زائف) ``grad(vmap(MyFunc))``
    يجب أن يكون قابلاً للاستبدال بـ ``grad(map(MyFunc))``.

    إذا كان لديك autograd.Function أي سلوك مخصص في تمريرة الخلف، يرجى
    ضع ذلك في اعتبارك.

.. note::

    من الاستخدامات المشروعة كتابة طريقة vmap مخصصة ثابتة لـ
    :class:`~torch.autograd.Function` يمكن لـ PyTorch إنشاء قاعدة vmap لها
    عبر ``generate_vmap_rule=True``. قد ترغب في القيام بذلك إذا كانت قاعدة vmap المولدة
    لا يحتوي على الدلالات التي تبحث عنها.

:func:`torch.func.jvp` الدعم
------------------------------

لدعم التمايز التلقائي للأمام، يجب أن يكون لـ :class:`torch.autograd.Function` :meth:`~Function.jvp` طريقة ثابتة.
يرجى الاطلاع على :ref:`forward-ad-autograd-function` للحصول على التفاصيل.