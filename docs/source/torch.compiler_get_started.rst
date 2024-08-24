.. _torch.compiler_get_started:

البدء
======

قبل قراءة هذا القسم، تأكد من قراءة :ref:`torch.compiler_overview`.

لنبدأ بالنظر في مثال بسيط على ``torch.compile`` يوضح كيفية استخدام ``torch.compile`` للاستنتاج. يوضح هذا المثال ميزتي ``torch.cos()`` و ``torch.sin()`` اللتين تعتبران مثالاً على المشغلين النقطيين لأنهم يعملون عنصرًا تلو الآخر على متجه. قد لا يظهر هذا المثال مكاسب أداء كبيرة، ولكنه يجب أن يساعدك على تكوين فهم بديهي لكيفية استخدام ``torch.compile`` في برامجك الخاصة.

.. note::
   لتشغيل هذا البرنامج النصي، يجب أن يكون لديك وحدة معالجة رسومية واحدة على الأقل على جهازك.
   إذا لم يكن لديك وحدة GPU، فيمكنك إزالة التعليمات البرمجية ``.to(device="cuda:0")``
   في المقتطف أدناه وسيعمل على وحدة المعالجة المركزية. يمكنك أيضًا تعيين الجهاز إلى
   ``xpu:0`` لتشغيله على وحدات معالجة الرسوميات Intel®.

.. code:: python

   import torch
   def fn(x):
      a = torch.cos(x)
      b = torch.sin(a)
      return b
   new_fn = torch.compile(fn, backend="inductor")
   input_tensor = torch.randn(10000).to(device="cuda:0")
   a = new_fn(input_tensor)

هناك مشغل نقطي أكثر شهرة قد ترغب في استخدامه وهو
شيء مثل ``torch.relu()``. تعد العمليات النقطية في وضع الحريص دون المستوى الأمثل لأن كل منها ستحتاج إلى قراءة مصفوفة من الذاكرة، وإجراء بعض التغييرات، ثم إعادة كتابة تلك التغييرات. أهم عملية تحسين يقوم بها المحرك هي الاندماج. في
يمكننا في المثال أعلاه تحويل قراءتين (``x``، ``a``) و
2 كتابات (``a``، ``b``) إلى قراءة واحدة (``x``) وكتابة واحدة (``b``)، وهو أمر بالغ الأهمية خاصة لوحدات معالجة الرسومات الأحدث حيث يكون الاختناق هو عرض النطاق الترددي للذاكرة (مدى السرعة التي يمكنك بها إرسال البيانات إلى وحدة معالجة الرسومات) بدلاً من الحوسبة (مدى سرعة وحدة معالجة الرسومات في معالجة عمليات النقطة العائمة).

وهناك تحسين رئيسي آخر يوفره المحرك هو الدعم التلقائي
لرسوم CUDA.
تساعد رسوم CUDA في القضاء على العبء الزائد الناتج عن إطلاق نوى فردية
من برنامج Python وهو أمر ذو صلة خاصة بوحدات معالجة الرسومات الأحدث.

يدعم TorchDynamo العديد من backends المختلفة، ولكن TorchInductor يعمل على وجه التحديد
من خلال إنشاء نوى Triton <https://github.com/openai/triton>. دعنا نحفظ
مثالنا أعلاه في ملف يسمى ``example.py``. يمكننا فحص التعليمات البرمجية
تم إنشاء نوى Triton عن طريق تشغيل ``TORCH_COMPILE_DEBUG=1 python example.py``.
مع تنفيذ البرنامج النصي، يجب أن ترى رسائل "DEBUG" المطبوعة على
المحطة الطرفية. بالقرب من نهاية السجل، يجب أن ترى مسارًا إلى مجلد
يحتوي على ``torchinductor_ <your_username>``. في هذا المجلد، يمكنك العثور على
يحتوي ملف "output_code.py" على كود النواة المولد المشابه لما يلي:

.. code-block:: python

   @pointwise(size_hints=[16384], filename=__file__, triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
   @triton.jit
   def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
      xnumel = 10000
      xoffset = tl.program_id(0) * XBLOCK
      xindex = xoffset + tl.arange(0, XBLOCK)[:]
      xmask = xindex < xnumel
      x0 = xindex
      tmp0 = tl.load(in_ptr0 + (x0), xmask, other=0.0)
      tmp1 = tl.cos(tmp0)
      tmp2 = tl.sin(tmp1)
      tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)

.. note:: مقتطف الكود أعلاه هو مثال. اعتمادًا على الأجهزة الخاصة بك،
   قد تشاهد كودًا مختلفًا تم إنشاؤه.

ويمكنك التحقق من أن دمج "cos" و "sin" قد حدث بالفعل
لأن عمليات "cos" و "sin" تحدث داخل نواة Triton واحدة
وتتم الاحتفاظ بالمتغيرات المؤقتة في سجلات ذات وصول سريع جدًا.

اقرأ المزيد حول أداء Triton
`هنا <https://openai.com/blog/triton/>`__. نظرًا لأن الكود مكتوب
في Python، فمن السهل إلى حد ما فهمه حتى إذا لم تكن قد كتبت الكثير من نوى CUDA.

بعد ذلك، دعنا نجرب نموذجًا حقيقيًا مثل resnet50 من PyTorch
مركز التطوير.

.. code-block:: python

   import torch
   model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
   opt_model = torch.compile(model, backend="inductor")
   opt_model(torch.randn(1,3,64,64))

وهذه ليست backend الوحيدة المتاحة، يمكنك تشغيلها في REPL
``torch.compiler.list_backends()`` لمعرفة جميع backends المتاحة. جرب التالي
``cudagraphs`` للإلهام.

استخدام نموذج مُدرب مسبقًا
~~~~~~~~~~~~~~~~~~~~~~~~

غالبًا ما يستفيد مستخدمو PyTorch من النماذج المُدربة مسبقًا من
`المحولات <https://github.com/huggingface/transformers>`__ أو
`تيم <https://github.com/rwightman/pytorch-image-models>`__ ويتمثل أحد الأهداف التصميمية في أن يعمل TorchDynamo و TorchInductor خارج الصندوق مع
أي نموذج يرغب الأشخاص في تأليفه.

دعنا نقوم بتنزيل نموذج مُدرب مسبقًا مباشرةً من مركز Hub الخاص بـ HuggingFace وتحسينه:

.. code-block:: python

   import torch
   from transformers import BertTokenizer, BertModel
   # نسخ ولصق من هنا https://huggingface.co/bert-base-uncased
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
   model = torch.compile(model, backend="inductor") # هذا هو سطر الكود الوحيد الذي قمنا بتغييره
   text = "استبدلني بأي نص تريده."
   encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")
   output = model(**encoded_input)

إذا قمت بإزالة ``to(device="cuda:0")`` من النموذج و
``encoded_input``، فستقوم Triton بإنشاء نوى C++ التي سيتم
تحسينها لتشغيلها على وحدة المعالجة المركزية الخاصة بك. يمكنك فحص كل من Triton أو C++
نوى لبيرت. إنها أكثر تعقيدًا من مثال الرياضيات المثلثية الذي جربناه أعلاه ولكن يمكنك بالمثل تصفحها ومعرفة ما إذا كنت تفهم كيفية عمل PyTorch.

وبالمثل، دعنا نجرب مثالًا على TIMM:

.. code-block:: python

   import timm
   import torch
   model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=2)
   opt_model = torch.compile(model, backend="inductor")
   opt_model(torch.randn(64,3,7,7))

الخطوات التالية
~~~~~~~~~~

في هذا القسم، راجعنا بعض أمثلة الاستدلال وتوصلنا إلى فهم أساسي
كيف يعمل torch.compile. إليك ما يمكنك التحقق منه بعد ذلك:

- `تعليمات torch.compile على التدريب <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_
- :ref:`torch.compiler_api`
- :ref:`torchdynamo_fine_grain_tracing`