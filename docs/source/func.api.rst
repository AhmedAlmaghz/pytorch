مرجع واجهة برمجة التطبيقات لـ torch.func
================================

.. currentmodule:: torch.func

.. automodule:: torch.func

تحويلات الدالة
----------
.. autosummary::
    :toctree: generated
    :nosignatures:

     vmap
     grad
     grad_and_value
     vjp
     jvp
     linearize
     jacrev
     jacfwd
     hessian
     functionalize

مرافق للعمل مع وحدات torch.nn
-----------------------------

بشكل عام، يمكنك إجراء تحويل على دالة تستدعي ``torch.nn.Module``.
على سبيل المثال، ما يلي هو مثال على حساب جاكوبي لدالة
تأخذ ثلاث قيم وتعيد ثلاث قيم:

.. code-block:: python

    model = torch.nn.Linear(3, 3)

    def f(x):
        return model(x)

    x = torch.randn(3)
    jacobian = jacrev(f)(x)
    assert jacobian.shape == (3, 3)

ومع ذلك، إذا كنت تريد القيام بشيء مثل حساب جاكوبي على معلمات
النموذج، يجب أن تكون هناك طريقة لبناء دالة حيث تكون المعلمات هي المدخلات للدالة.
هذا ما يفعله :func: `functional_call`:
فهو يقبل nn.Module، و ``parameters`` المحولة، والمدخلات إلى
تمرير النموذج إلى الأمام. ويعيد قيمة تشغيل تمرير النموذج إلى الأمام
مع استبدال المعلمات.

هكذا سنحسب جاكوبي على المعلمات

.. code-block:: python

    model = torch.nn.Linear(3, 3)

    def f(params, x):
        return torch.func.functional_call(model, params, x)

    x = torch.randn(3)
    jacobian = jacrev(f)(dict(model.named_parameters()), x)


.. autosummary::
    :toctree: generated
    :nosignatures:

    functional_call
    stack_module_state
    replace_all_batch_norm_modules

إذا كنت تبحث عن معلومات حول إصلاح وحدات معايرة الدُفعات، يرجى اتباع الإرشادات هنا

.. toctree::
   :maxdepth: 1

   func.batch_norm