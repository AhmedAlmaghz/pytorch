.. role:: hidden
    :class: hidden-section

torch.nn.attention.bias
========================

.. automodule:: torch.nn.attention.bias
.. currentmodule:: torch.nn.attention.bias

CausalBias
==========

.. autofunction:: CausalBias

.. autofunction:: causal_lower_right

.. autofunction:: causal_upper_left

.. autofunction:: CausalVariant

في هذا النص، يتم استخدام تنسيق ReStructuredText لوصف وحدة "torch.nn.attention.bias" في مكتبة "PyTorch". يتم تعريف أربع وظائف/فئات: "CausalBias"، "causal_lower_right"، "causal_upper_left"، و "CausalVariant". هذه الوظائف مرتبطة بالاهتمام السببي، والذي يستخدم في نماذج معالجة اللغة الطبيعية والمهام التسلسلية.

"CausalBias": يمثل تحيز الاهتمام السببي، والذي يستخدم لتوجيه نموذج للتركيز على المواضع السابقة عند حساب الاهتمام.

"causal_lower_right": وظيفة لإنشاء مصفوفة تحيز مثلث سفلي أيمن، والتي تستخدم في الاهتمام السببي.

"causal_upper_left": وظيفة لإنشاء مصفوفة تحيز على شكل مثلث علوي أيسر، والتي يمكن أن تكون مفيدة في بعض بنى الاهتمام.

"CausalVariant": يمثل فئة متغير الاهتمام السببي، والتي توفر خيارات مختلفة لتحيز الاهتمام السببي.