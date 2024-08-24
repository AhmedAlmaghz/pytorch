# PyTorch 2.4: البدء مع معالج الرسوميات Intel GPU
=================================================

تم إطلاق الدعم لمعالجات الرسوميات Intel GPUs بالتزامن مع إصدار PyTorch v2.4.

يدعم هذا الإصدار فقط البناء من المصدر لمعالجات الرسوميات Intel GPUs.

متطلبات الأجهزة
----------------

.. list-table::
   :header-rows: 1

   * - الأجهزة المدعومة
     - Intel® Data Center GPU Max Series
   * - نظام التشغيل المدعوم
     - Linux

يتوافق PyTorch لمعالجات الرسوميات Intel GPUs مع Intel® Data Center GPU Max Series ويدعم فقط نظام التشغيل Linux مع الإصدار 2.4.

متطلبات البرمجيات
----------------

كمتطلب أساسي، قم بتثبيت التعريف والبرامج المطلوبة باتباع `متطلبات تثبيت PyTorch لمعالجات الرسوميات Intel GPUs <https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html>`_.

إعداد البيئة
--------

قبل البدء، تحتاج إلى إعداد البيئة. يمكن القيام بذلك عن طريق تشغيل نص ``setvars.sh`` المقدم من حزم ``intel-for-pytorch-gpu-dev`` و ``intel-pti-dev``.

.. code-block::

   source ${ONEAPI_ROOT}/setvars.sh

.. note::
   ``ONEAPI_ROOT`` هو المجلد الذي قمت بتثبيت حزم ``intel-for-pytorch-gpu-dev`` و ``intel-pti-dev`` فيه. عادة ما يقع في ``/opt/intel/oneapi/`` أو ``~/intel/oneapi/``.

البناء من المصدر
-----------------

الآن بعد أن قمنا بتثبيت جميع الحزم المطلوبة وتم تفعيل البيئة، استخدم الأوامر التالية لتثبيت ``pytorch`` و ``torchvision`` و ``torchaudio`` عن طريق البناء من المصدر. لمزيد من التفاصيل، يرجى الرجوع إلى الأدلة الرسمية في `بناء PyTorch من المصدر <https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support>`_، و `بناء Vision من المصدر <https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation>`_، و `بناء Audio من المصدر <https://pytorch.org/audio/main/build.linux.html>`_.

.. code-block::

   # الحصول على كود PyTorch المصدري
   git clone --recursive https://github.com/pytorch/pytorch
   cd pytorch
   git checkout main # أو استخدم أمر "git checkout" لإصدار محدد >= v2.4
   git submodule sync
   git submodule update --init --recursive

   # الحصول على الحزم المطلوبة للترجمة
   conda install cmake ninja
   pip install -r requirements.txt

   # PyTorch لمعالجات الرسوميات Intel GPUs يدعم فقط منصة Linux حاليا.
   # تثبيت الحزم المطلوبة لترجمة PyTorch.
   conda install intel::mkl-static intel::mkl-include

   # (اختياري) إذا كنت تستخدم torch.compile مع inductor/triton، قم بتثبيت الإصدار المطابق من triton
   # قم بتشغيل الأمر من مجلد pytorch بعد استنساخه
   # لدعم معالج الرسوميات Intel GPU، يرجى تصدير USE_XPU=1 قبل تشغيل الأمر.
   USE_XPU=1 make triton

   # إذا كنت ترغب في ترجمة PyTorch مع تمكين C++ ABI الجديد، قم بتشغيل هذا الأمر أولاً:
   export _GLIBCXX_USE_CXX11_ABI=1

   # بناء PyTorch من المصدر
   export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
   python setup.py develop
   cd ..

   # (اختياري) إذا كنت تستخدم torchvison.
   # الحصول على كود torchvision
   git clone https://github.com/pytorch/vision.git
   cd vision
   git checkout main # أو استخدم إصدارا محددا
   python setup.py develop
   cd ..

   # (اختياري) إذا كنت تستخدم torchaudio.
   # الحصول على كود torchaudio
   git clone https://github.com/pytorch/audio.git
   cd audio
   pip install -r requirements.txt
   git checkout main # أو استخدم إصدارا محددا
   git submodule sync
   git submodule update --init --recursive
   python setup.py develop
   cd ..

التحقق من توفر معالج الرسوميات Intel GPU
--------------------------------

.. note::
   تأكد من إعداد البيئة بشكل صحيح باتباع قسم `إعداد البيئة <#set-up-environment>`_ قبل تشغيل الكود.

للتحقق مما إذا كان معالج الرسوميات Intel GPU الخاص بك متاحا، يمكنك عادة استخدام الكود التالي:

.. code-block::

   import torch
   torch.xpu.is_available()  # torch.xpu هو واجهة برمجة التطبيقات API لدعم معالج الرسوميات Intel GPU

إذا كانت النتيجة ``False``، فتأكد من وجود معالج رسوميات Intel GPU في نظامك واتبع `متطلبات تثبيت PyTorch لمعالجات الرسوميات Intel GPUs <https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html>`_ بشكل صحيح. بعد ذلك، تحقق من اكتمال ترجمة PyTorch بنجاح.

تغييرات الكود الدنيا
-------------------

إذا كنت تقوم بترحيل الكود من ``cuda``، فستقوم بتغيير المراجع من ``cuda`` إلى ``xpu``. على سبيل المثال:

.. code-block::

   # كود CUDA
   tensor = torch.tensor([1.0, 2.0]).to("cuda")

   # كود معالج الرسوميات Intel GPU
   tensor = torch.tensor([1.0, 2.0]).to("xpu")

توضح النقاط التالية الدعم والقيود لـ PyTorch مع معالج الرسوميات Intel GPU:

#. يتم دعم كل من سير عمل التدريب والاستدلال.
#. يتم دعم كل من الوضع الفوري ``torch.compile``.
#. يتم دعم أنواع البيانات مثل FP32 و BF16 و FP16 والدقة المختلطة التلقائية (AMP).
#. لن يتم دعم النماذج التي تعتمد على المكونات الخارجية حتى إصدار PyTorch v2.5 أو أحدث.

أمثلة
--------

يحتوي هذا القسم على أمثلة للاستخدام لكل من سير عمل الاستدلال والتدريب.

أمثلة الاستدلال
^^^^^^^^^^^^^^^^^^

فيما يلي بعض الأمثلة على سير عمل الاستدلال.

الاستدلال مع FP32
"""""""""""""""""""

.. code-block::

   import torch
   import torchvision.models as models

   model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   ######## تغييرات الكود #######
   model = model.to("xpu")
   data = data.to("xpu")
   ######## تغييرات الكود #######

   with torch.no_grad():
       model(data)

   print("انتهى التنفيذ")

الاستدلال مع AMP
""""""""""""""""""

.. code-block::

   import torch
   import torchvision.models as models

   model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### تغييرات الكود #################
   model = model.to("xpu")
   data = data.to("xpu")
   #################### تغييرات الكود #################

   with torch.no_grad():
       d = torch.rand(1, 3, 224, 224)
       ############################# تغييرات الكود #####################
       d = d.to("xpu")
       # قم بضبط dtype=torch.bfloat16 لنوع البيانات BF16
       with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=True):
       ############################# تغييرات الكود #####################
           model(data)

   print("انتهى التنفيذ")

الاستدلال باستخدام ``torch.compile``
""""""""""""""""""""""""""""""""

.. code-block::

   import torch
   import torchvision.models as models

   model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
   model.eval()
   data = torch.rand(1, 3, 224, 224)
   ITERS = 10

   ######## تغييرات الكود #######
   model = model.to("xpu")
   data = data.to("xpu")
   ######## تغييرات الكود #######

   model = torch.compile(model)
   for i in range(ITERS):
       with torch.no_grad():
           model(data)

   print("انتهى التنفيذ")

أمثلة التدريب
^^^^^^^^^^^^^^^^^

فيما يلي بعض الأمثلة على سير عمل التدريب.

التدريب مع FP32
"""""""""""""""

.. code-block::

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = "datasets/cifar10/"

   transform = torchvision.transforms.Compose(
       [
           torchvision.transforms.Resize((224, 224)),
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
       ]
   )
   train_dataset = torchvision.datasets.CIFAR10(
       root=DATA,
       train=True,
       transform=transform,
       download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
   model.train()
   ######################## تغييرات الكود #######################
   model = model.to("xpu")
   criterion = criterion.to("xpu")
   ######################## تغييرات الكود #######################

   for batch_idx, (data, target) in enumerate(train_loader):
       ########## تغييرات الكود ##########
       data = data.to("xpu")
       target = target.to("xpu")
       ########## تغييرات الكود ##########
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       print(batch_idx)
   torch.save(
       {
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
       },
       "checkpoint.pth",
   )

   print("انتهى التنفيذ")

التدريب مع AMP
""""""""""""""

.. code-block::

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = "datasets/cifar10/"

   use_amp=True

   transform = torchvision.transforms.Compose(
       [
           torchvision.transforms.Resize((224, 224)),
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
       ]
   )
   train_dataset = torchvision.datasets.CIFAR10(
       root=DATA,
       train=True,
       transform=transform,
       download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
   scaler = torch.amp.GradScaler(enabled=use_amp)

   model.train()
   ######################## تغييرات الكود #######################
   model = model.to("xpu")
   criterion = criterion.to("xpu")
   ######################## تغييرات الكود #######################

   for batch_idx, (data, target) in enumerate(train_loader):
       ########## تغييرات الكود ##########
       data = data.to("xpu")
       target = target.to("xpu")
       ########## تغييرات الكود ##########
       # قم بضبط dtype=torch.bfloat16 لنوع البيانات BF16
       with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=use_amp):
           output = model(data)
           loss = criterion(output, target)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad()
       print(batch_idx)

   torch.save(
       {
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
       },
       "checkpoint.pth",
   )

   print("انتهى التنفيذ")