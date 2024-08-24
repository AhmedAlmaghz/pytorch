.. _modules:

الوحدات
-----
يستخدم PyTorch الوحدات النمطية لتمثيل الشبكات العصبية. الوحدات النمطية هي:

* **لبنات البناء للحساب ذي الحالة.**
   يوفر PyTorch مكتبة قوية من الوحدات النمطية ويجعل من السهل تحديد وحدات نمطية مخصصة جديدة، مما يسمح
   ببناء شبكات عصبية معقدة ومتعددة الطبقات بسهولة.

* **مدمجة بإحكام مع نظام PyTorch**
  `autograd <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_
  **.** تجعل الوحدات النمطية من السهل تحديد المعلمات القابلة للتعلم لتحديثها بواسطة محسنات PyTorch.

* **سهلة العمل والتحويل.** من السهل حفظ الوحدات النمطية واستعادتها ونقلها بين
  أجهزة وحدة المعالجة المركزية / وحدة معالجة الرسومات / وحدة معالجة المصفوفات، وتشذيبها، وتحديد كمياتها، والمزيد.

تصف هذه الملاحظة الوحدات النمطية، وهي مخصصة لجميع مستخدمي PyTorch. نظرًا لأن الوحدات النمطية أساسية جدًا في PyTorch،
  يتم تناول العديد من الموضوعات في هذه الملاحظة بالتفصيل في ملاحظات أو تعليمات برمجية أخرى، ويتم أيضًا توفير روابط للعديد من تلك المستندات
  هنا.

.. contents:: :local:

وحدة نمطية مخصصة بسيطة
----------------------

لتبدأ، دعنا نلقي نظرة على إصدار مخصص أبسط لوحدة PyTorch النمطية :class: `~ torch.nn.Linear`.
تطبق هذه الوحدة تحويلًا أفينيًا على مدخلاتها.

.. code-block:: python

   import torch
   from torch import nn

   class MyLinear(nn.Module):
     def __init__(self, in_features, out_features):
       super().__init__()
       self.weight = nn.Parameter(torch.randn(in_features, out_features))
       self.bias = nn.Parameter(torch.randn(out_features))

     def forward(self, input):
       return (input @ self.weight) + self.bias

تتمتع هذه الوحدة النمطية البسيطة بالخصائص الأساسية التالية للوحدات النمطية:

* **يرث من فئة الوحدة النمطية الأساسية.**
  يجب أن تكون جميع الوحدات النمطية فئة فرعية من :class: `~ torch.nn.Module` لسهولة التركيب مع الوحدات النمطية الأخرى.

* **تحدد بعض "الحالة" التي تستخدم في الحساب.**
  هنا، تتكون الحالة من وزن عشوائي ومتجهات "انحياز" تحدد التحويل الأفيني. نظرًا لأن كل منهما معرف
  كـ :class: `~ torch.nn.parameter.Parameter`، يتم
  *تسجيلها* للوحدة النمطية وسيتم تتبعها تلقائيًا وإرجاعها من الاستدعاءات
  إلى :func: `~ torch.nn.Module.parameters`. يمكن اعتبار المعلمات الجوانب "القابلة للتعلم" لحساب الوحدة النمطية (سنتحدث عن هذا لاحقًا). لاحظ أن الوحدات النمطية
  غير مطلوبة للحالة، ويمكن أن تكون أيضًا بدون حالة.

* **تحدد دالة forward() التي تؤدي الحساب.** بالنسبة لهذه الوحدة النمطية للتحويل الأفيني، يتم ضرب الإدخال
  بالمصفوفة مع معلمة "الوزن" (باستخدام التدوين المختصر "@") وإضافته إلى معلمة "الانحياز"
  لإنتاج الإخراج. بشكل أكثر عمومية، يمكن لتنفيذ "forward()" للوحدة النمطية إجراء حسابات عشوائية
  تنطوي على أي عدد من المدخلات والمخرجات.

توضح هذه الوحدة النمطية البسيطة كيف تقوم الوحدات النمطية بتغليف الحالة والحساب معًا. يمكن إنشاء مثيلات هذه الوحدة النمطية واستدعاؤها:

.. code-block:: python

   m = MyLinear(4, 3)
   sample_input = torch.randn(4)
   m(sample_input)
   : tensor([-0.3037, -1.0413, -4.2057], grad_fn=<AddBackward0>)

لاحظ أن الوحدة النمطية نفسها قابلة للاستدعاء، وأن استدعاءها يستدعي دالتها "forward()".
يشير هذا الاسم إلى مفاهيم "التمرير للأمام" و"التمرير للخلف"، والتي تنطبق على كل وحدة نمطية.
"التمرير للأمام" مسؤول عن تطبيق الحساب الذي تمثله الوحدة النمطية
  على الإدخال (الإدخالات) المعطاة (كما هو موضح في المقتطف أعلاه). يحسب "التمرير للخلف" تدرجات
  إخراج الوحدة النمطية فيما يتعلق بمدخلاتها، والتي يمكن استخدامها "لتدريب" المعلمات من خلال أساليب الانحدار التدريجي. يعتني نظام autograd في PyTorch تلقائيًا بحساب التمرير للخلف، لذلك
  لا يلزم تنفيذ دالة "backward()" يدويًا لكل وحدة نمطية. تتم تغطية عملية تدريب
  معلمات الوحدة النمطية من خلال التمريرات الأمامية / الخلفية المتعاقبة بالتفصيل في
  :ref: `Neural Network Training with Modules`.

يمكن إجراء تكرار لمجموعة المعلمات الكاملة التي سجلتها الوحدة النمطية من خلال استدعاء
:func: `~ torch.nn.Module.parameters` أو :func: `~ torch.nn.Module.named_parameters`،
والذي يتضمن الأخير اسم كل معلمة:

.. code-block:: python

   for parameter in m.named_parameters():
     print(parameter)
   : ('weight', Parameter containing:
   tensor([[ 1.0597,  1.1796,  0.8247],
           [-0.5080, -1.2635, -1.1045],
           [ 0.0593,  0.2469, -1.4299],
           [-0.4926, -0.5457,  0.4793]], requires_grad=True))
   ('bias', Parameter containing:
   tensor([ 0.3634,  0.2015, -0.8525], requires_grad=True))

بشكل عام، تكون المعلمات التي تسجلها الوحدة النمطية جوانب من حساب الوحدة النمطية التي يجب
"التعلم منها". يوضح قسم لاحق من هذه الملاحظة كيفية تحديث هذه المعلمات باستخدام إحدى محسنات PyTorch.
ولكن قبل أن نصل إلى ذلك، دعنا نلقي نظرة أولاً على كيفية تركيب الوحدات النمطية مع بعضها البعض.

الوحدات النمطية كلبنات بناء
--------------------------

يمكن أن تحتوي الوحدات النمطية على وحدات نمطية أخرى، مما يجعلها لبنات بناء مفيدة لتطوير وظائف أكثر تعقيدًا.
أبسط طريقة للقيام بذلك هي استخدام الوحدة النمطية :class: `~ torch.nn.Sequential`. يتيح لنا ذلك ربط عدة وحدات نمطية معًا:

.. code-block:: python

   net = nn.Sequential(
     MyLinear(4, 3),
     nn.ReLU(),
     MyLinear(3, 1)
   )

   sample_input = torch.randn(4)
   net(sample_input)
   : tensor([-0.6749], grad_fn=<AddBackward0>)

لاحظ أن :class: `~ torch.nn.Sequential` يقوم تلقائيًا بإدخال إخراج أول وحدة "MyLinear" كإدخال
في :class: `~ torch.nn.ReLU`، وإخراج ذلك كإدخال في وحدة "MyLinear" الثانية. كما هو موضح، فإنه يقتصر على التسلسل في ترتيب الوحدات النمطية ذات الإدخال والإخراج الفرديين.

بشكل عام، يوصى بتعريف وحدة نمطية مخصصة لأي شيء يتجاوز أبسط حالات الاستخدام، حيث يمنح ذلك
مرونة كاملة في كيفية استخدام الوحدات الفرعية لحساب الوحدة النمطية.

على سبيل المثال، إليك شبكة عصبية بسيطة تم تنفيذها كوحدة نمطية مخصصة:

.. code-block:: python

   import torch.nn.functional as F

   class Net(nn.Module):
     def __init__(self):
       super().__init__()
       self.l0 = MyLinear(4, 3)
       self.l1 = MyLinear(3, 1)
     def forward(self, x):
       x = self.l0(x)
       x = F.relu(x)
       x = self.l1(x)
       return x

تتكون هذه الوحدة النمطية من "أطفال" أو "وحدات فرعية" (\ ``l0`` و ``l1``\ ) تحدد طبقات
الشبكة العصبية وتستخدم للحساب داخل طريقة "forward()" للوحدة النمطية. يمكن إجراء تكرار للأطفال المباشرين لوحدة نمطية عبر استدعاء :func: `~ torch.nn.Module.children` أو
:func: `~ torch.nn.Module.named_children`:

.. code-block:: python

   net = Net()
   for child in net.named_children():
     print(child)
   : ('l0', MyLinear())
   ('l1', MyLinear())

للتعمق أكثر من مجرد الأطفال المباشرين، تقوم الدالتان :func: `~ torch.nn.Module.modules` و
:func: `~ torch.nn.Module.named_modules` بالتنقل بشكل *متكرر* خلال الوحدة النمطية ووحداتها الفرعية:

.. code-block:: python

   class BigNet(nn.Module):
     def __init__(self):
       super().__init__()
       self.l1 = MyLinear(5, 4)
       self.net = Net()
     def forward(self, x):
       return self.net(self.l1(x))

   big_net = BigNet()
   for module in big_net.named_modules():
     print(module)
   : ('', BigNet(
     (l1): MyLinear()
     (net): Net(
       (l0): MyLinear()
       (l1): MyLinear()
     )
   ))
   ('l1', MyLinear())
   ('net', Net(
     (l0): MyLinear()
     (l1): MyLinear()
   ))
   ('net.l0', MyLinear())
   ('net.l1', MyLinear())

في بعض الأحيان، يكون من الضروري أن تقوم الوحدة النمطية بتحديد وحدات فرعية ديناميكيًا.
الوحدتان النمطيتان :class: `~ torch.nn.ModuleList` و :class: `~ torch.nn.ModuleDict` مفيدتان هنا؛ حيث تقومان
بتسجيل الوحدات الفرعية من قائمة أو قاموس:

.. code-block:: python

   class DynamicNet(nn.Module):
     def __init__(self, num_layers):
       super().__init__()
       self.linears = nn.ModuleList(
         [MyLinear(4, 4) for _ in range(num_layers)])
       self.activations = nn.ModuleDict({
         'relu': nn.ReLU(),
         'lrelu': nn.LeakyReLU()
       })
       self.final = MyLinear(4, 1)
     def forward(self, x, act):
       for linear in self.linears:
         x = linear(x)
       x = self.activations[act](x)
       x = self.final(x)
       return x

   dynamic_net = DynamicNet(3)
   sample_input = torch.randn(4)
   output = dynamic_net(sample_input, 'relu')

بالنسبة لأي وحدة نمطية معينة، تتكون معلماتها من معلماتها المباشرة بالإضافة إلى معلمات جميع الوحدات الفرعية.
هذا يعني أن الاستدعاءات إلى :func: `~ torch.nn.Module.parameters` و :func: `~ torch.nn.Module.named_parameters` ستتضمن
بشكل متكرر معلمات الطفل، مما يسمح بتحسين ملائم لجميع المعلمات داخل الشبكة:

.. code-block:: python

   for parameter in dynamic_net.named_parameters():
     print(parameter)
   : ('linears.0.weight', Parameter containing:
   tensor([[-1.2051,  0.7601,  1.1065,  0.1963],
           [ 3.0592,  0.4354,  1.6598,  0.9828],
           [-0.4446,  0.4628,  0.8774,  1.6848],
           [-0.1222,  1.5458,  1.1729,  1.4647]], requires_grad=True))
   ('linears.0.bias', Parameter containing:
   tensor([ 1.5310,  1.0609, -2.0940,  1.1266], requires_grad=True))
   ('linears.1.weight', Parameter containing:
   tensor([[ 2.1113, -0.0623, -1.0806,  0.3508],
           [-0.0550,  1.5317,  1.1064, -0.5562],
           [-0.4028, -0.6942,  1.5793, -1.0140],
           [-0.0329,  0.1160, -1.7183, -1.0434]], requires_grad=True))
   ('linears.1.bias', Parameter containing:
   tensor([ 0.0361, -0.9768, -0.3889,  1.1613], requires_grad=True))
   ('linears.2.weight', Parameter containing:
   tensor([[-2.6340, -0.3887, -0.9979,  0.0767],
           [-0.3526,  0.8756, -1.5847, -0.6016],
           [-0.3269, -0.1608,  0.2897, -2.0829],
           [ 2.6338,  0.9239,  0.6943, -1.5034]], requires_grad=True))
   ('linears.2.bias', Parameter containing:
   tensor([ 1.0268,  0.4489, -0.9403,  0.1571], requires_grad=True))
   ('final.weight', Parameter containing:
   tensor([[ 0.2509], [-0.5052], [ 0.3088], [-1.4951]], requires_grad=True))
   ('final.bias', Parameter containing:
   tensor([0.3381], requires_grad=True))

من السهل أيضًا نقل جميع المعلمات إلى جهاز مختلف أو تغيير دقتها باستخدام
:func: `~ torch.nn.Module.to`:

.. code-block:: python

   # Move all parameters to a CUDA device
   dynamic_net.to(device='cuda')

   # Change precision of all parameters
   dynamic_net.to(dtype=torch.float64)

   dynamic_net(torch.randn(5, device='cuda', dtype=torch.float64))
   : tensor([6.5166], device='cuda:0', dtype=torch.float64, grad_fn=<AddBackward0>)

وبشكل أكثر عمومية، يمكن تطبيق دالة تعسفية على وحدة نمطية ووحداتها الفرعية بشكل متكرر باستخدام
دالة :func: `~ torch.nn.Module.apply`. على سبيل المثال، لتطبيق التهيئة المخصصة على معلمات
وحدة نمطية ووحداتها الفرعية:

.. code-block:: python

   # Define a function to initialize Linear weights.
   # Note that no_grad() is used here to avoid tracking this computation in the autograd graph.
   @torch.no_grad()
   def init_weights(m):
     if isinstance(m, nn.Linear):
       nn.init.xavier_normal_(m.weight)
       m.bias.fill_(0.0)

   # Apply the function recursively on the module and its submodules.
   dynamic_net.apply(init_weights)

توضح هذه الأمثلة كيف يمكن تشكيل شبكات عصبية معقدة من خلال تركيب الوحدات النمطية والتعامل معها بسهولة. للسماح بإنشاء شبكات عصبية سريعًا وبسهولة مع الحد الأدنى من التعليمات البرمجية، يوفر PyTorch مكتبة كبيرة من الوحدات النمطية عالية الأداء داخل مساحة الاسم :mod: `torch.nn` التي تقوم بعمليات الشبكة العصبية الشائعة مثل التجميع، والتحويلات، ووظائف الخسارة، وما إلى ذلك.

في القسم التالي، نقدم مثالًا كاملًا على تدريب شبكة عصبية.

لمزيد من المعلومات، راجع:

* مكتبة الوحدات النمطية التي يوفرها PyTorch: `torch.nn <https://pytorch.org/docs/stable/nn.html>`_
* تحديد وحدات الشبكة العصبية النمطية: https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html

.. _Neural Network Training with Modules:

تدريب الشبكة العصبية باستخدام الوحدات النمطية
بعد بناء الشبكة، يجب تدريبها، ويمكن تحسين معلماتها بسهولة باستخدام إحدى خوارزميات التحسين (Optimizers) من وحدة PyTorch: ``torch.optim``:

.. code-block:: python

   # إنشاء الشبكة (من القسم السابق) وخوارزمية التحسين
   net = Net()
   optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

   # تشغيل حلقة تدريبية تجريبية "تُعلم" الشبكة
   # لإخراج دالة الصفر الثابتة
   for _ in range(10000):
     input = torch.randn(4)
     output = net(input)
     loss = torch.abs(output)
     net.zero_grad()
     loss.backward()
     optimizer.step()

   # بعد التدريب، قم بتبديل وضعية الوحدة إلى الوضع التقييمي لإجراء الاستنتاج، وحساب مقاييس الأداء، وما إلى ذلك.
   # (انظر المناقشة أدناه لوصف أوضاع التدريب والتقييم)
   ...
   net.eval()
   ...

في هذا المثال المبسط، تتعلم الشبكة ببساطة إخراج الصفر، حيث يتم "معاقبة" أي إخراج غير صفري وفقاً لقيمته المطلقة عن طريق استخدام ``torch.abs`` كدالة خسارة. وعلى الرغم من أن هذه ليست مهمة مثيرة للاهتمام، إلا أن الأجزاء الرئيسية من التدريب موجودة:

* يتم إنشاء شبكة.
* يتم إنشاء خوارزمية تحسين (في هذه الحالة، خوارزمية النسبية المتوافقة)، ويتم ربط معلمات الشبكة بها.
* حلقة تدريب...
    * تحصل على إدخال،
    * تشغل الشبكة،
    * تحسب الخسارة،
    * تصفير تدرجات معلمات الشبكة،
    * تستدعي ``loss.backward()`` لتحديث تدرجات المعلمات،
    * تستدعي ``optimizer.step()`` لتطبيق التدرجات على المعلمات.

بعد تشغيل الشفرة أعلاه، لاحظ أن معلمات الشبكة قد تغيرت. على وجه الخصوص، عند فحص قيمة معلمة "الوزن" (weight) للطبقة ``l1``، نجد أن قيمها أصبحت الآن أقرب بكثير من الصفر (كما هو متوقع):

.. code-block:: python

   print(net.l1.weight)
   : Parameter containing:
   tensor([[-0.0013],
           [ 0.0030],
           [-0.0008]], requires_grad=True)

لاحظ أن العملية أعلاه تتم بالكامل أثناء وجود وحدة الشبكة في "وضع التدريب". الوحدات الافتراضية تكون في وضع التدريب ويمكن التبديل بين أوضاع التدريب والتقييم باستخدام ``torch.nn.Module.train`` و ``torch.nn.Module.eval``. ويمكن أن تتصرف الوحدات بشكل مختلف حسب الوضع الذي تكون فيه. على سبيل المثال، تحتفظ وحدة ``torch.nn.BatchNorm`` بمتوسط متحرك وانحراف معياري أثناء التدريب لا يتم تحديثهما عندما تكون الوحدة في وضع التقييم. بشكل عام، يجب أن تكون الوحدات في وضع التدريب أثناء التدريب، ولا يتم التبديل إلى وضع التقييم إلا للاستنتاج أو التقييم. وفيما يلي مثال على وحدة مخصصة تتصرف بشكل مختلف بين الوضعين:

.. code-block:: python

   class ModalModule(nn.Module):
     def __init__(self):
       super().__init__()

     def forward(self, x):
       if self.training:
         # إضافة ثابت فقط في وضع التدريب
         return x + 1.
       else:
         return x


   m = ModalModule()
   x = torch.randn(4)

   print('training mode output: {}'.format(m(x)))
   : tensor([1.6614, 1.2669, 1.0617, 1.6213, 0.5481])

   m.eval()
   print('evaluation mode output: {}'.format(m(x)))
   : tensor([ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519])

يمكن أن يكون تدريب الشبكات العصبية أمراً صعباً في كثير من الأحيان. لمزيد من المعلومات، يمكنك الاطلاع على:

* استخدام خوارزميات التحسين: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html.
* تدريب الشبكات العصبية: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
* مقدمة إلى autograd: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

حالة الوحدة
------------

في القسم السابق، قمنا بتدريب "معلمات" الوحدة، أو الجوانب القابلة للتعلم من الحساب. الآن، إذا أردنا حفظ النموذج المدرب على القرص، فيمكننا القيام بذلك عن طريق حفظ ``state_dict`` الخاص به (أي "قاموس الحالة"):

.. code-block:: python

   # حفظ الوحدة
   torch.save(net.state_dict(), 'net.pt')

   ...

   # تحميل الوحدة لاحقاً
   new_net = Net()
   new_net.load_state_dict(torch.load('net.pt'))
   : <All keys matched successfully>

يحتوي ``state_dict`` للوحدة على الحالة التي تؤثر على حساباتها. وهذا يشمل، على سبيل المثال لا الحصر، معلمات الوحدة. بالنسبة لبعض الوحدات، قد يكون من المفيد وجود حالة تتجاوز المعلمات تؤثر على حسابات الوحدة ولكنها غير قابلة للتعلم. بالنسبة لهذه الحالات، يوفر PyTorch مفهوم "المخازن المؤقتة" (buffers)، سواء "المستمرة" (persistent) أو "غير المستمرة" (non-persistent). فيما يلي نظرة عامة على مختلف أنواع الحالات التي يمكن أن تحتويها الوحدة:

* **المعلمات**: الجوانب القابلة للتعلم من الحساب؛ موجودة ضمن ``state_dict``.
* **المخازن المؤقتة**: الجوانب غير القابلة للتعلم من الحساب

  * **المخازن المؤقتة المستمرة**: موجودة ضمن ``state_dict`` (أي تتم تسويتها عند الحفظ والتحميل)
  * **المخازن المؤقتة غير المستمرة**: غير موجودة ضمن ``state_dict`` (أي يتم استبعادها من التسوية)

كمثال محفز لاستخدام المخازن المؤقتة، ضع في اعتبارك وحدة بسيطة تحتفظ بمتوسط متحرك. نريد أن تكون القيمة الحالية للمتوسط المتحرك جزءاً من ``state_dict`` للوحدة بحيث يتم استعادتها عند تحميل الشكل المسلسل للوحدة، ولكننا لا نريد أن تكون قابلة للتعلم. يوضح هذا المقتطف كيفية استخدام ``torch.nn.Module.register_buffer`` لتحقيق ذلك:

.. code-block:: python

   class RunningMean(nn.Module):
     def __init__(self, num_features, momentum=0.9):
       super().__init__()
       self.momentum = momentum
       self.register_buffer('mean', torch.zeros(num_features))
     def forward(self, x):
       self.mean = self.momentum * self.mean + (1.0 - self.momentum) * x
       return self.mean

الآن، تعتبر القيمة الحالية للمتوسط المتحرك جزءاً من ``state_dict`` للوحدة وسيتم استعادتها بشكل صحيح عند تحميل الوحدة من القرص:

.. code-block:: python

   m = RunningMean(4)
   for _ in range(10):
     input = torch.randn(4)
     m(input)

   print(m.state_dict())
   : OrderedDict([('mean', tensor([ 0.1041, -0.1113, -0.0647,  0.1515]))]))

   # سيحتوي الشكل المسلسل على مصفوفة 'mean'
   torch.save(m.state_dict(), 'mean.pt')

   m_loaded = RunningMean(4)
   m_loaded.load_state_dict(torch.load('mean.pt'))
   assert(torch.all(m.mean == m_loaded.mean))

كما ذكرنا سابقاً، يمكن استبعاد المخازن المؤقتة من ``state_dict`` للوحدة عن طريق وضع علامة عليها كمخازن مؤقتة غير مستمرة:

.. code-block:: python

   self.register_buffer('unserialized_thing', torch.randn(5), persistent=False)

تتأثر كل من المخازن المؤقتة المستمرة وغير المستمرة بالتغييرات على مستوى الوحدة في الجهاز/نوع البيانات المطبقة باستخدام ``torch.nn.Module.to``:

.. code-block:: python

   # ينقل جميع معلمات الوحدة ومخازنها المؤقتة إلى الجهاز/نوع البيانات المحدد
   m.to(device='cuda', dtype=torch.float64)

يمكن تكرار المخازن المؤقتة للوحدة باستخدام ``torch.nn.Module.buffers`` أو ``torch.nn.Module.named_buffers``.

.. code-block:: python

   for buffer in m.named_buffers():
     print(buffer)

توضح الفئة التالية الطرق المختلفة لتسجيل المعلمات والمخازن المؤقتة داخل الوحدة:

.. code-block:: python

   class StatefulModule(nn.Module):
     def __init__(self):
       super().__init__()
       # تعيين معلمة nn.Parameter كسمة للوحدة يسجل تلقائياً المصفوفة كمعلمة للوحدة.
       self.param1 = nn.Parameter(torch.randn(2))

       # طريقة بديلة قائمة على النصوص لتسجيل معلمة.
       self.register_parameter('param2', nn.Parameter(torch.randn(3)))

       # يحجز "param3" كمعلمة، مما يمنع تعيينه لأي شيء
       # باستثناء معلمة. لن تكون الإدخالات "null" مثل هذه موجودة في قاموس حالة الوحدة.
       self.register_parameter('param3', None)

       # يسجل قائمة من المعلمات.
       self.param_list = nn.ParameterList([nn.Parameter(torch.randn(2)) for i in range(3)])

       # يسجل قاموس من المعلمات.
       self.param_dict = nn.ParameterDict({
         'foo': nn.Parameter(torch.randn(3)),
         'bar': nn.Parameter(torch.randn(4))
       })

       # يسجل مخزناً مؤقتاً مستمراً (مخزناً مؤقتاً يظهر في قاموس حالة الوحدة).
       self.register_buffer('buffer1', torch.randn(4), persistent=True)

       # يسجل مخزناً مؤقتاً غير مستمر (مخزناً مؤقتاً لا يظهر في قاموس حالة الوحدة).
       self.register_buffer('buffer2', torch.randn(5), persistent=False)

       # يحجز "buffer3" كمخزن مؤقت، مما يمنع تعيينه لأي شيء
       # باستثناء مخزن مؤقت. لن تكون الإدخالات "null" مثل هذه موجودة في قاموس حالة الوحدة.
       self.register_buffer('buffer3', None)

       # إضافة وحدة فرعية يسجل معلماتها كمعلمات للوحدة.
       self.linear = nn.Linear(2, 3)

   m = StatefulModule()

   # حفظ وتحميل قاموس الحالة.
   torch.save(m.state_dict(), 'state.pt')
   m_loaded = StatefulModule()
   m_loaded.load_state_dict(torch.load('state.pt'))

   # لاحظ أن المخزن المؤقت غير المستمر "buffer2" والسمات المحجوزة "param3" و "buffer3" لا
   # تظهر في قاموس الحالة.
   print(m_loaded.state_dict())
   : OrderedDict([('param1', tensor([-0.0322,  0.9066])),
                  ('param2', tensor([-0.4472,  0.1409,  0.4852])),
                  ('buffer1', tensor([ 0.6949, -0.1944,  1.2911, -2.1044])),
                  ('param_list.0', tensor([ 0.4202, -0.1953])),
                  ('param_list.1', tensor([ 1.5299, -0.8747])),
                  ('param_list.2', tensor([-1.6289,  1.4898])),
                  ('param_dict.bar', tensor([-0.6434,  1.5187,  0.0346, -0.4077])),
                  ('param_dict.foo', tensor([-0.0845, -1.4324,  0.7022])),
                  ('linear.weight', tensor([[-0.3915, -0.6176],
                                            [ 0.6062, -0.5992],
                                            [ 0.4452, -0.2843]])),
                  ('linear.bias', tensor([-0.3710, -0.0795, -0.3947]))])

لمزيد من المعلومات، يمكنك الاطلاع على:

* الحفظ والتحميل: https://pytorch.org/tutorials/beginner/saving_loading_models.html
* دلالات التسلسل: https://pytorch.org/docs/main/notes/serialization.html
* ما هو قاموس الحالة؟ https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html

تهيئة الوحدة
--------
.. default-domain:: torch

بشكل افتراضي، يتم تهيئة المعلمات وذاكرات التخزين العشوائي ذات النقطة العائمة للوحدات النمطية التي يوفرها :mod:`torch.nn` أثناء إنشاء الوحدة النمطية كقيم ذات نقطة عائمة 32 بت على وحدة المعالجة المركزية باستخدام مخطط تهيئة تم تحديده لأداء جيد تاريخيًا لنوع الوحدة النمطية. في بعض حالات الاستخدام، قد يكون من المستحسن إجراء التهيئة باستخدام نوع بيانات مختلف أو جهاز مختلف (مثل وحدة معالجة الرسومات) أو تقنية تهيئة مختلفة.

أمثلة:

.. code-block:: python

   # تهيئة الوحدة النمطية مباشرة على وحدة معالجة الرسومات.
   m = nn.Linear(5, 3, device='cuda')

   # تهيئة الوحدة النمطية بمعلمات ذات دقة نصفية.
   m = nn.Linear(5, 3, dtype=torch.half)

   # تخطي تهيئة المعلمات الافتراضية وإجراء التهيئة المخصصة (مثل التهيئة المتعامدة).
   m = torch.nn.utils.skip_init(nn.Linear, 5, 3)
   nn.init.orthogonal_(m.weight)

لاحظ أن خيارات الجهاز ونوع البيانات الموضحة أعلاه تنطبق أيضًا على أي ذاكرات تخزين عشوائي ذات نقطة عائمة مسجلة للوحدة النمطية:

.. code-block:: python

   m = nn.BatchNorm2d(3, dtype=torch.half)
   print(m.running_mean)
   : tensor([0., 0., 0.], dtype=torch.float16)

بينما يمكن لمؤلفي الوحدات النمطية استخدام أي جهاز أو نوع بيانات لتهيئة المعلمات في وحداتهم النمطية المخصصة، فإن الممارسة الجيدة هي استخدام ``dtype=torch.float`` و ``device='cpu'`` بشكل افتراضي أيضًا. يمكنك أيضًا توفير المرونة الكاملة في هذه المجالات لوحدتك النمطية المخصصة عن طريق الالتزام بالاتفاقية الموضحة أعلاه والتي تتبعها جميع وحدات :mod:`torch.nn` النمطية:

* توفير وسيط ``device`` للمنشئ الذي ينطبق على أي معلمات / ذاكرات تخزين مسجلة بواسطة الوحدة النمطية.
* توفير وسيط ``dtype`` للمنشئ الذي ينطبق على أي معلمات / ذاكرات تخزين عشوائي ذات نقطة عائمة مسجلة بواسطة الوحدة النمطية.
* استخدم فقط دالات التهيئة (أي الدالات من :mod:`torch.nn.init`) على المعلمات وذاكرات التخزين داخل منشئ الوحدة النمطية. لاحظ أن هذا مطلوب فقط لاستخدام :func:`~torch.nn.utils.skip_init`؛ راجع `هذه الصفحة <https://pytorch.org/tutorials/prototype/skip_param_init.html#updating-modules-to-support-skipping-initialization>`_ للحصول على تفسير.

لمزيد من المعلومات، راجع:

* تخطي تهيئة معلمات الوحدة النمطية: https://pytorch.org/tutorials/prototype/skip_param_init.html

خطافات الوحدة النمطية
----------------

في :ref:`تدريب الشبكة العصبية باستخدام الوحدات النمطية <neural-network-training-with-modules>`، قمنا بتوضيح عملية التدريب للوحدة النمطية، والتي تقوم بشكل تكراري بتنفيذ التمريرات الأمامية والخلفية، وتحديث معلمات الوحدة النمطية في كل تكرار. لمزيد من التحكم في هذه العملية، يوفر PyTorch "خطافات" يمكنها تنفيذ حسابات تعسفية أثناء التمرير الأمامي أو الخلفي، وحتى تعديل كيفية إجراء التمرير إذا لزم الأمر. بعض الأمثلة المفيدة لهذه الوظيفة تشمل التصحيح، وتصوير التنشيط، وفحص التدرجات بعمق، وما إلى ذلك. يمكن إضافة الخطافات إلى الوحدات النمطية التي لم تكتبها بنفسك، مما يعني أن هذه الوظيفة يمكن تطبيقها على الوحدات النمطية التابعة لجهات خارجية أو التي يوفرها PyTorch.

يوفر PyTorch نوعين من الخطافات للوحدات النمطية:

* يتم استدعاء **خطافات التمرير الأمامي** أثناء التمرير الأمامي. يمكن تثبيتها لوحدة نمطية معينة باستخدام :func:`~torch.nn.Module.register_forward_pre_hook` و :func:`~torch.nn.Module.register_forward_hook`.
  سيتم استدعاء هذه الخطافات على التوالي مباشرة قبل استدعاء دالة التمرير الأمامي وبعدها مباشرة.
  بدلاً من ذلك، يمكن تثبيت هذه الخطافات عالميًا لجميع الوحدات النمطية باستخدام الدالتين المماثلتين :func:`~torch.nn.modules.module.register_module_forward_pre_hook` و :func:`~torch.nn.modules.module.register_module_forward_hook`.
* يتم استدعاء **خطافات التمرير الخلفي** أثناء التمرير الخلفي. يمكن تثبيتها باستخدام :func:`~torch.nn.Module.register_full_backward_pre_hook` و :func:`~torch.nn.Module.register_full_backward_hook`.
  سيتم استدعاء هذه الخطافات عندما يتم حساب التمرير الخلفي لهذه الوحدة النمطية.
  يسمح :func:`~torch.nn.Module.register_full_backward_pre_hook` للمستخدم بالوصول إلى تدرجات المخرجات في حين أن :func:`~torch.nn.Module.register_full_backward_hook` يسمح للمستخدم بالوصول إلى التدرجات لكل من المدخلات والمخرجات. بدلاً من ذلك، يمكن تثبيتها عالميًا لجميع الوحدات النمطية باستخدام :func:`~torch.nn.modules.module.register_module_full_backward_hook` و :func:`~torch.nn.modules.module.register_module_full_backward_pre_hook`.

تسمح جميع الخطافات للمستخدم بإرجاع قيمة محدّثة سيتم استخدامها في بقية الحساب.
وبالتالي، يمكن استخدام هذه الخطافات لتنفيذ تعليمات برمجية تعسفية إما مع التمرير الأمامي/الخلفي المنتظم للوحدة النمطية أو لتعديل بعض المدخلات/المخرجات دون الحاجة إلى تغيير دالة ``forward()`` للوحدة النمطية.

فيما يلي مثال يوضح استخدام خطافات التمرير الأمامي والخلفي:

.. code-block:: python

   torch.manual_seed(1)

   def forward_pre_hook(m, inputs):
     # يسمح بفحص وتعديل الإدخال قبل التمرير الأمامي.
     # لاحظ أن المدخلات ملفوفة دائمًا في مجموعة.
     input = inputs[0]
     return input + 1.

   def forward_hook(m, inputs, output):
     # يسمح بفحص الإدخالات / المخرجات وتعديل المخرجات
     # بعد التمرير الأمامي. لاحظ أن المدخلات ملفوفة دائمًا في مجموعة في حين يتم تمرير المخرجات
     # كما هي.

     # حساب بقايا على غرار ResNet.
     return output + inputs[0]

   def backward_hook(m, grad_inputs, grad_outputs):
     # يسمح بفحص grad_inputs / grad_outputs وتعديل
     # grad_inputs المستخدمة في بقية التمرير الخلفي. لاحظ أن grad_inputs و
     # يتم دائمًا لف grad_outputs في tuples.
     new_grad_inputs = [torch.ones_like(gi) * 42. for gi in grad_inputs]
     return new_grad_inputs

   # إنشاء وحدة نمطية ومدخلات عينة.
   m = nn.Linear(3, 3)
   x = torch.randn(2, 3, requires_grad=True)

   # ==== توضيح خطافات التمرير الأمامي. ====
   # تشغيل الإدخال من خلال الوحدة النمطية قبل وبعد إضافة الخطافات.
   print('output with no forward hooks: {}'.format(m(x)))
   : output with no forward hooks: tensor([[-0.5059, -0.8158,  0.2390],
                                           [-0.0043,  0.4724, -0.1714]], grad_fn=<AddmmBackward>)

   # لاحظ أن الإدخال المعدل يؤدي إلى إخراج مختلف.
   forward_pre_hook_handle = m.register_forward_pre_hook(forward_pre_hook)
   print('output with forward pre hook: {}'.format(m(x)))
   : output with forward pre hook: tensor([[-0.5752, -0.7421,  0.4942],
                                           [-0.0736,  0.5461,  0.0838]], grad_fn=<AddmmBackward>)

   # لاحظ الإخراج المعدل.
   forward_hook_handle = m.register_forward_hook(forward_hook)
   print('output with both forward hooks: {}'.format(m(x)))
   : output with both forward hooks: tensor([[-1.0980,  0.6396,  0.4666],
                                             [ 0.3634,  0.6538,  1.0256]], grad_fn=<AddBackward0>)

   # إزالة الخطافات؛ لاحظ أن الإخراج هنا يتطابق مع الإخراج قبل إضافة الخطافات.
   forward_pre_hook_handle.remove()
   forward_hook_handle.remove()
   print('output after removing forward hooks: {}'.format(m(x)))
   : output after removing forward hooks: tensor([[-0.5059, -0.8158,  0.2390],
                                                  [-0.0043,  0.4724, -0.1714]], grad_fn=<AddmmBackward>)

   # ==== توضيح خطافات التمرير الخلفي. ====
   m(x).sum().backward()
   print('x.grad with no backwards hook: {}'.format(x.grad))
   : x.grad with no backwards hook: tensor([[ 0.4497, -0.5046,  0.3146],
                                            [ 0.4497, -0.5046,  0.3146]])

   # مسح التدرجات قبل تشغيل التمرير الخلفي مرة أخرى.
   m.zero_grad()
   x.grad.zero_()

   m.register_full_backward_hook(backward_hook)
   m(x).sum().backward()
   print('x.grad with backwards hook: {}'.format(x.grad))
   : x.grad with backwards hook: tensor([[42., 42., 42.],
                                         [42., 42., 42.]])

ميزات متقدمة
----------

يوفر PyTorch أيضًا العديد من الميزات المتقدمة المصممة للعمل مع الوحدات النمطية. جميع هذه الوظائف متاحة للوحدات النمطية المكتوبة مخصصًا، مع التحذير الصغير الذي قد يتطلب من الوحدات النمطية الالتزام بقيود معينة من أجل دعمها. يمكن العثور على مناقشة متعمقة لهذه الميزات والمتطلبات المقابلة لها في الروابط أدناه.

التدريب الموزع
***********

توجد طرق مختلفة للتدريب الموزع داخل PyTorch، لكل من التدريب على نطاق واسع باستخدام وحدات معالجة الرسومات المتعددة وكذلك التدريب عبر أجهزة متعددة. اطلع على صفحة
`نظرة عامة على التدريب الموزع <https://pytorch.org/tutorials/beginner/dist_overview.html>`_ للحصول على معلومات مفصلة حول كيفية استخدامها.

تحليل أداء التوصيف
***************

يمكن أن يكون `محلل أداء PyTorch <https://pytorch.org/tutorials/beginner/profiler.html>`_ مفيدًا لتحديد الاختناقات في الأداء داخل نماذج الخاصة بك. فهو يقيس ويخرج خصائص الأداء لكل من استخدام الذاكرة والوقت المستغرق.

تحسين الأداء باستخدام التقريب
********************

يمكن لتحسين الأداء واستخدام الذاكرة باستخدام تقنيات التقريب للوحدات النمطية باستخدام عرض بت أقل من دقة النقطة العائمة. تحقق من آليات التقريب المختلفة التي يوفرها PyTorch
`هنا <https://pytorch.org/docs/stable/quantization.html>`_.

تحسين استخدام الذاكرة بالتشذيب
*********************

غالبًا ما تكون نماذج التعلم العميق كبيرة الحجم مفرطة في المعلمات، مما يؤدي إلى ارتفاع استخدام الذاكرة. لمكافحة ذلك، يوفر PyTorch آليات لتشذيب النماذج، والتي يمكن أن تساعد في تقليل استخدام الذاكرة مع الحفاظ على دقة المهمة. يصف
`التدريب على التشذيب <https://pytorch.org/tutorials/intermediate/pruning_tutorial.html>`_ كيفية استخدام تقنيات التشذيب التي يوفرها PyTorch أو تحديد تقنيات التشذيب المخصصة حسب الحاجة.

المعلمات
*******

بالنسبة لبعض التطبيقات، قد يكون من المفيد تقييد مساحة المعلمات أثناء تدريب النموذج. على سبيل المثال، يمكن أن يؤدي فرض تقييد متعامد للمعلمات التي يتم تعلمها إلى تحسين التقارب لشبكات RNN. يوفر PyTorch آلية لتطبيق `المعلمات <https://pytorch.org/tutorials/intermediate/parametrizations.html>`_ مثل هذا، كما يسمح بتعريف قيود مخصصة.

تحويل الوحدات النمطية باستخدام FX
****************************

يوفر مكون `FX <https://pytorch.org/docs/stable/fx.html>`_ في PyTorch طريقة مرنة لتحويل الوحدات النمطية عن طريق العمل مباشرة على مخططات حسابات الوحدات النمطية. يمكن استخدامه لإنشاء أو معالجة وحدات نمطية برمجيًا لمجموعة واسعة من حالات الاستخدام. لاستكشاف FX، تحقق من هذه الأمثلة لاستخدام FX للاندماج `التقريبي + معيار الدفعة <https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html>`_ و `تحليل أداء وحدة المعالجة المركزية <https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html>`_.