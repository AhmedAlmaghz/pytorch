.. _torch.utils.data:

torch.utils.data
=================

يحتوي هذا على فئات مفيدة للعمل مع مجموعات البيانات.

.. currentmodule:: torch.utils.data

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Dataset
   IterableDataset
   DataLoader
   RandomSampler
   SequentialSampler
   BatchSampler
   DistributedSampler
   WeightedRandomSampler

.. _torch.utils.data.dataloader:

torch.utils.data.DataLoader
--------------------------

.. autoclass:: DataLoader
   :members:
   :inherited-members:

.. _torch.utils.data.dataset:

torch.utils.data.Dataset
------------------------

.. autoclass:: Dataset
   :members:

.. _torch.utils.data.iterable:

torch.utils.data.IterableDataset
--------------------------------

.. autoclass:: IterableDataset
   :members:

.. _torch.utils.data.sampler:

torch.utils.data.Sampler
------------------------

.. autoclass:: Sampler
   :members:

.. _torch.utils.data.random_sampler:

torch.utils.data.RandomSampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomSampler
   :members:

.. _torch.utils.data.sequential_sampler:

torch.utils.data.SequentialSampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SequentialSampler
   :members:

.. _torch.utils.data.distributed.distributed_sampler:

torch.utils.data.distributed.DistributedSampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: distributed.DistributedSampler
   :members:

.. _torch.utils.data.batch_sampler:

torch.utils.data.BatchSampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BatchSampler
   :members:

.. _torch.utils.data.weighted_random_sampler:

torch.utils.data.WeightedRandomSampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: WeightedRandomSampler
   :members:

.. _torch.utils.data.dataloader_worker_process:

torch.utils.data.DataLoader Worker Process
-----------------------------------------

.. currentmodule:: torch.utils.data._utils.worker

.. autoclass:: DataLoaderIter
   :members:

.. autofunction:: _worker_loop

.. autofunction:: _mp_sample_worker

.. autofunction:: _spawn_workers

.. autofunction:: _init_process

.. autofunction:: _worker_loop_wrapper

.. autofunction:: _worker_loop_wrapper_tpu

.. autofunction:: _worker_loop_tpu

.. autofunction ._worker_loop_gpu

.. autofunction:: _pin_memory_tpu

.. autofunction:: _pin_memory_gpu

.. _torch.utils.data.dataloader_options:

torch.utils.data.DataLoader Options
----------------------------------

.. currentmodule:: torch.utils.data

.. autoclass:: DataLoader
   :members: num_workers, pin_memory, timeout, batch_size, collate_fn, worker_init_fn, multiprocessing_context, generator, persistent_workers

.. _torch.utils.data.dataloader_notes:

torch.utils.data.DataLoader ملاحظات
---------------------------------

.. currentmodule:: torch.utils.data

.. _torch.utils.data.dataloader.pin_memory:

pin_memory
^^^^^^^^^

عند تعيينها إلى ``True`` ، سيتم استخدام الذاكرة المثبتة في عملية العامل.
يؤدي ذلك إلى نسخ أقل في بعض الحالات.

.. note::

   لا تتوفر الذاكرة المثبتة إلا في PyTorch الذي تم بناؤه مع CUDA.

.. _torch.utils.data.dataloader.num_workers:

num_workers
^^^^^^^^^^

عدد العمليات الفرعية المستخدمة للبيانات الموازية. لا يتم استخدام العمليات الفرعية في
الوضع الافتراضي.

.. warning::

   قد لا تعمل العمليات الفرعية بشكل جيد مع TensorFlow. إذا واجهت مشكلات، يرجى
   استخدام ``num_workers=0``.

.. _torch.utils.data.dataloader.timeout:

timeout
^^^^^^^

عدد الثواني التي ينتظرها DataLoader قبل إيقاف عملية العامل. القيمة الافتراضية هي 0،
أي لا يوجد حد زمني.

.. _torchMultiplier:

.. _torch.utils.data.dataloader.persistent_workers:

persistent_workers
^^^^^^^^^^^^^^^^^^

إذا كان ``True`` ، فإن العامل لن يتم إيقافه بعد دورة واحدة. هذا سيحافظ على
الذاكرة المثبتة في العامل، مما قد يؤدي إلى تقليل النسخ. ومع ذلك، فإن العمال
الذين يستمرون في العمل قد يستهلكون المزيد من الذاكرة. القيمة الافتراضية هي
``False``.

.. _torch.utils.data.dataloader.multiprocessing_context:

multiprocessing_context
^^^^^^^^^^^^^^^^^^^^^^^^

سياق المعالجة المتعددة الذي سيتم استخدامه لتشغيل العمال. القيمة الافتراضية هي
``None`` ، والتي ستستخدم السياق الافتراضي (عادة ``fork`` في Unix و ``spawn`` في
Windows). يمكنك أيضًا استخدام ``multiprocessing.get_context('spawn')`` أو
``multiprocessing.get_context('fork')`` أو ``multiprocessing.get_context('forkserver')``.

.. note::

   لا يتم دعم ``forkserver`` في Windows.

.. _torch.utils.data.dataloader.worker_init_fn:

worker_init_fn
^^^^^^^^^^^^^^

إذا تم تعيينها، فسيتم استدعاء هذه الدالة في كل عامل قبل بدء دورة العمل.
يمكن استخدامها لمكافحة بعض مشكلات التهيئة في DataLoader.

.. _torch.utils.data.dataloader.generator:

generator
^^^^^^^^^

عند تعيينها، سيتم استخدام هذا المولد لإنشاء الأرقام العشوائية في DataLoader.

.. _torch.utils.data.dataloader.collate_fn:

collate_fn
^^^^^^^^^^

عند تعيينها، سيتم استخدام هذه الدالة لتجميع عينات البيانات في الدُفعات.

.. _torch.utils.data.dataloader.iter_per_load:

iter_per_load
^^^^^^^^^^^^^^

عدد المرات التي يجب أن يمر بها كل عامل على مجموعة البيانات. القيمة الافتراضية هي 1.

.. _torch.utils.data.dataloader.default_collate:

torch.utils.data.dataloader.default_collate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: default_collate

.. _torch.utils.data.dataloader.exceptions:

torch.utils.data.dataloader استثناءات
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torch.utils.data

.. autoclass:: DataLoaderWarning

.. _torch.utils.data.dataloader.best_practices:

أفضل الممارسات لـ DataLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torch.utils.data

.. _torch.utils.data.dataloader.best_practices.pinned_memory_benchmark:

اختبار الذاكرة المثبتة
'''''''''''''''''''''''''

إذا كنت تستخدم الذاكرة المثبتة، فيجب عليك التأكد من أنها توفر بالفعل زيادة في
الأداء. يمكنك القيام بذلك عن طريق مقارنة سرعة DataLoader مع وبدون الذاكرة المثبتة.

.. code:: python

   # بدون ذاكرة مثبتة
   dataloader = DataLoader(dataset, num_workers=4)

   # مع ذاكرة مثبتة
   dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)

.. _torch.utils.data.dataloader.best_practices.num_workers_benchmark:

اختبار عدد العمليات الفرعية
'''''''''''''''''''

يمكن أن يكون لعدد العمليات الفرعية تأثير كبير على أداء DataLoader. يجب عليك
اختبار عدد العمال لمعرفة العدد الأمثل.

.. code:: python

   # بدون عمال
   dataloader = DataLoader(dataset)

   # مع 4 عمال
   dataloMultiplier = DataLoader(dataset, num_workers=4)

.. _torch.utils.data.dataloader.best_practices.large_vs_small_batches:

الدُفعات الكبيرة مقابل الدُفعات الصغيرة
'''''''''''''''''''''''''

قد يكون حجم الدُفعة مهمًا أيضًا. إذا كان حجم الدُفعة كبيرًا، فقد لا يكون لديك
ذاكرة كافية. إذا كان حجم الدُفعة صغيرًا، فقد لا تستفيد بشكل كافٍ من وحدة معالجة
الرسوميات.

.. _torch.utils.data.dataloader.best_practices.single_vs_multi_process:

عملية واحدة مقابل عمليات متعددة
''''''''''''''''''''

إذا كنت تستخدم عملية واحدة، فيمكنك استخدام ``torch.no_grad()`` لتجنب
تسجيل العمليات الحسابية.

.. code:: python

   for data in dataloader:
       with torch.no_grad():
           # قم بعملك

إذا كنت تستخدم عمليات متعددة، فيجب عليك استخدام ``torch.enable_grad()`` و
``torch.disable_grad()`` لتجنب تسجيل العمليات الحسابية.

.. code:: python

   for data in dataloader:
       torch.set_grad_enabled(True)
       # قم بعملك

.. _torch.utils.data.dataloader.best_practices.multiprocessing_in_jupyter:

المعالجة المتعددة في Jupyter
'''''''''''''''''''''

إذا كنت تستخدم Jupyter، فيجب عليك استخدام ``start_method='spawn'`` في
``torch.utils.data.DataLoader`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_queue:

المعالجة المتعددة مع Queue
'''''''''''''''''''

إذا كنت تستخدم Queue في عملية رئيسية، فيجب عليك استخدام ``start_method='spawn'``
في ``torch.utils.data.DataLoader`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_tpu:

المعالجة المتعددة مع TPU
'''''''''''''''''

إذا كنت تستخدم TPU، فيجب عليك استخدام ``start_method='fork'`` في
``torch.utils.data.DataLoader`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=8, multiprocessing_context=get_context('fork'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_forkserver:

المعالجة المتعددة مع ForkServer
''''''''''''''''''''''''

إذا كنت تستخدم ``forkserver`` ، فيجب عليك استخدام ``forkserver_preload`` لتجنب
الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('forkserver'), forkserver_preload=True)

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_fork:

المعالجة المتعددة مع Fork
''''''''''''''''''

إذا كنت تستخدم ``fork`` ، فيجب عليك استخدام ``fork_args`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('fork'), fork_args=(('RLIMIT_NOFILE', 1024), ))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_spawn:

المعالجة المتعددة مع Spawn
''''''''''''''''''''

إذا كنت تستخدم ``spawn`` ، فيجب عليك استخدام ``spawn_args`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'), spawn_args=(('RLIMIT_NOFILE', 1024), ))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_gloo:

المعالجة المتعددة مع Gloo
''''''''''''''''''

إذا كنت تستخدم Gloo، فيجب عليك استخدام ``spawn`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_mpi:

المعالجة المتعددة مع MPI
'''''''''''''''''

إذا كنت تستخدم MPI، فيجب عليك استخدام ``spawn`` لتجنب الأخطاء.

.. code:: python

   datalo

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_torch_distributed:

المعالجة المتعددة مع torch.distributed
''''''''''''''''''''''''''''''''

إذا كنت تستخدم ``torch.distributed`` ، فيجب عليك استخدام ``spawn`` لتجنب
الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_horovod:

المعالجة المتعددة مع Horovod
'''''''''''''''''''''

إذا كنت تستخدم Horovod، فيجب عليك استخدام ``spawn`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_xla:

المعالجة المتعددة مع XLA
'''''''''''''''''

إذا كنت تستخدم XLA، فيجب عليك استخدام ``spawn`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_torch_cuda:

المعالجة المتعددة مع torch.cuda
''''''''''''''''''''''''

إذا كنت تستخدم ``torch.cuda`` ، فيجب عليك استخدام ``spawn`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_torch_cuda_async:

المعالجة المتعددة مع torch.cuda.async
''''''''''''''''''''''''''''''

إذا كنت تستخدم ``torch.cuda.async`` ، فيجب عليك استخدام ``spawn`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_torch_cuda_comm:

المعالجة المتعددة مع torch.cuda.comm
'''''''''''''''''''''''''''''

إذا كنت تستخدم ``torch.cuda.comm`` ، فيجب عليك استخدام ``spawn`` لتجنب الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_torch_distributed_launch:

المعالجة المتعددة مع torch.distributed.launch
''''''''''''''''''''''''''''''''''''''''

إذا كنت تستخدم ``torch.distributed.launch`` ، فيجب عليك استخدام ``spawn`` لتجنب
الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_torch_distributed_run:

المعالجة المتعددة مع torch.distributed.run
''''''''''''''''''''''''''''''''''''

إذا كنت تستخدم ``torch.distributed.run`` ، فيجب عليك استخدام ``spawn`` لتجنب
الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_torch_distributed_rpc:

المعالجة المتعددة مع torch.distributed.rpc
'''''''''''''''''''''''''''''''''''

إذا كنت تستخدم ``torch.distributed.rpc`` ، فيجب عليك استخدام ``spawn`` لتجنب
الأخطاء.

.. code:: python

   dataloader = DataLoader(dataset, num_workers=4, multiprocessing_context=get_context('spawn'))

.. _torch.utils.data.dataloader.best_practices.multiprocessing_with_torch_distributed_elastic:

المعالجة المتعددة مع torch.distributed
================================

في قلب أداة تحميل البيانات PyTorch يوجد فئة :class:`torch.utils.data.DataLoader`. تمثل هذه الفئة كائنًا قابلًا للتنفيذ في Python عبر مجموعة بيانات، مع دعم لما يلي:

* أنواع مجموعات البيانات على طراز الخرائط والطراز القابل للتنفيذ.
* تخصيص ترتيب تحميل البيانات.
* التجميع التلقائي.
* تحميل البيانات أحادية ومتعددة العمليات.
* تثبيت الذاكرة التلقائي.

تتم تهيئة هذه الخيارات بواسطة وسائط البناء لـ :class:`~torch.utils.data.DataLoader`، والتي لها التوقيع التالي::

    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, *, prefetch_factor=2,
               persistent_workers=False)

تصف الأقسام أدناه بالتفصيل تأثيرات هذه الخيارات واستخداماتها.

أنواع مجموعات البيانات
-----------------

وسيط البناء الأكثر أهمية لـ :class:`~torch.utils.data.DataLoader` هو وسيط :attr:`dataset`، والذي يشير إلى كائن مجموعة بيانات لتحميل البيانات منه. يدعم PyTorch نوعين مختلفين من مجموعات البيانات:

* مجموعات البيانات على طراز الخرائط.
* مجموعات البيانات على الطراز القابل للتنفيذ.

مجموعات البيانات على طراز الخرائط
^^^^^^^^^^^^^^^^^^^^^^^^

مجموعة البيانات على طراز الخرائط هي مجموعة بيانات تقوم بتنفيذ بروتوكولات :meth:`__getitem__` و :meth:`__len__`، وتمثل خريطة من المؤشرات/المفاتيح (غير الصحيحة) إلى عينات البيانات.

على سبيل المثال، يمكن لمجموعة بيانات مثل هذه، عند الوصول إليها باستخدام ``dataset[idx]``، قراءة الصورة التي تحمل المؤشر ``idx`` وتصنيفها المقابل من مجلد على القرص.

راجع :class:`~torch.utils.data.Dataset` لمزيد من التفاصيل.

مجموعات البيانات على الطراز القابل للتنفيذ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

مجموعة البيانات على الطراز القابل للتنفيذ هي مثيل لفئة فرعية من :class:`~torch.utils.data.IterableDataset` تقوم بتنفيذ بروتوكول :meth:`__iter__`، وتمثل كائنًا قابلًا للتنفيذ عبر عينات البيانات. وهذا النوع من مجموعات البيانات مناسب بشكل خاص للحالات التي تكون فيها القراءات العشوائية مكلفة أو حتى مستحيلة، وحيث يعتمد حجم الدفعة على البيانات المستردة.

على سبيل المثال، يمكن لمجموعة بيانات مثل هذه، عند استدعائها ``iter(dataset)``، أن تعيد دفق بيانات القراءة من قاعدة بيانات أو خادم بعيد أو حتى سجلات يتم إنشاؤها في الوقت الفعلي.

راجع :class:`~torch.utils.data.IterableDataset` لمزيد من التفاصيل.

.. note:: عند استخدام :class:`~torch.utils.data.IterableDataset` مع تحميل البيانات متعددة العمليات. يتم استنساخ نفس كائن مجموعة البيانات على كل عملية عامل، وبالتالي يجب تكوين النسخ المتماثلة بشكل مختلف لتجنب البيانات المكررة. راجع وثائق :class:`~torch.utils.data.IterableDataset` لمعرفة كيفية تحقيق ذلك.

ترتيب تحميل البيانات وفئة :class:`~torch.utils.data.Sampler`
-----------------------------------------------------

بالنسبة لمجموعات البيانات على الطراز القابل للتنفيذ، يتحكم المستخدم في ترتيب تحميل البيانات بالكامل. يسمح هذا بتنفيذ أسهل لقراءة الجزء والدفعات الديناميكية (على سبيل المثال، عن طريق إنتاج عينة مجمعة في كل مرة).

يتعلق بقية هذا القسم بحالة مجموعات البيانات على طراز الخرائط. تستخدم فئات :class:`torch.utils.data.Sampler` لتحديد تسلسل المؤشرات/المفاتيح المستخدمة في تحميل البيانات. تمثل كائنات قابلة للتنفيذ عبر المؤشرات إلى مجموعات البيانات. على سبيل المثال، في الحالة الشائعة مع الانحدار التدريجي العشوائي (SGD)، يمكن لـ :class:`~torch.utils.data.Sampler` أن يقوم بترتيب عشوائي لقائمة من المؤشرات وإنتاج كل منها في وقت واحد، أو إنتاج عدد صغير منها لتنفيذ SGD على دفعات صغيرة.

سيتم بناء عينة تسلسلية أو مختلطة تلقائيًا بناءً على وسيط :attr:`shuffle` إلى :class:`~torch.utils.data.DataLoader`. بدلاً من ذلك، يمكن للمستخدمين استخدام وسيط :attr:`sampler` لتحديد كائن :class:`~torch.utils.data.Sampler` مخصص يقوم في كل مرة بإنتاج مؤشر/مفتاح الاسترجاع التالي.

يمكن تمرير :class:`~torch.utils.data.Sampler` مخصص ينتج قائمة بمؤشرات الدفعات كوسيط :attr:`batch_sampler`. يمكن أيضًا تمكين التجميع التلقائي عبر وسيطي :attr:`batch_size` و :attr:`drop_last`. راجع "القسم التالي" <تحميل البيانات المجمعة وغير المجمعة> لمزيد من التفاصيل حول هذا الموضوع.

.. note::
  لا يتوافق أي من :attr:`sampler` أو :attr:`batch_sampler` مع مجموعات البيانات القابلة للتنفيذ، حيث لا يوجد لدى هذه المجموعات أي مفهوم للمفتاح أو المؤشر.

تحميل البيانات المجمعة وغير المجمعة
-------------------------

يدعم :class:`~torch.utils.data.DataLoader` التجميع التلقائي لعينات البيانات الفردية المستردة إلى دفعات عبر وسائط :attr:`batch_size` و :attr:`drop_last` و :attr:`batch_sampler` و :attr:`collate_fn` (والذي له دالة افتراضية).

التجميع التلقائي (افتراضي)
^^^^^^^^^^^^^^^^^^^^^

هذه هي الحالة الأكثر شيوعًا، وتتوافق مع استرداد دفعة مصغرة من البيانات وتجميعها في عينات مجمعة، أي تحتوي على مصفوفات ذات بُعد واحد يكون البعد الدفعي (عادةً الأول).

عندما لا يكون :attr:`batch_size` (افتراضيًا ``1``) ``None``، فإن محمل البيانات ينتج عينات مجمعة بدلاً من العينات الفردية. تُستخدم وسائط :attr:`batch_size` و :attr:`drop_last` لتحديد كيفية حصول محمل البيانات على دفعات من مفاتيح مجموعة البيانات. بالنسبة لمجموعات البيانات على طراز الخرائط، يمكن للمستخدمين بدلاً من ذلك تحديد :attr:`batch_sampler`، والذي ينتج قائمة بالمفاتيح في كل مرة.

.. note::
  يتم استخدام وسيطي :attr:`batch_size` و :attr:`drop_last` بشكل أساسي لبناء :attr:`batch_sampler` من :attr:`sampler`. بالنسبة لمجموعات البيانات على طراز الخرائط، يتم توفير :attr:`sampler` إما من قبل المستخدم أو يتم بناؤه بناءً على وسيط :attr:`shuffle`. بالنسبة لمجموعات البيانات القابلة للتنفيذ، فإن :attr:`sampler` هو عينة وهمية لا نهائية. راجع "هذا القسم" <ترتيب تحميل البيانات والعينة> لمزيد من التفاصيل حول العينات.

.. note::
  عند الاسترداد من مجموعات البيانات على الطراز القابل للتنفيذ باستخدام "تحميل البيانات متعددة العمليات" <تحميل البيانات متعددة العمليات>، فإن وسيط :attr:`drop_last` يسقط الدفعة الأخيرة غير الكاملة لكل نسخة من مجموعة بيانات العامل.

بعد استرداد قائمة العينات باستخدام المؤشرات من العينة، يتم استخدام الدالة التي تم تمريرها كوسيط :attr:`collate_fn` لتجميع قوائم العينات في دفعات.

في هذه الحالة، يكون تحميل البيانات من مجموعة بيانات على طراز الخرائط مكافئًا تقريبًا لما يلي::

    for indices in batch_sampler:
        yield collate_fn([dataset[i] for i in indices])

وتحميل البيانات من مجموعة بيانات على الطراز القابل للتنفيذ مكافئ تقريبًا لما يلي::

    dataset_iter = iter(dataset)
    for indices in batch_sampler:
        yield collate_fn([next(dataset_iter) for _ in indices])

يمكن استخدام :attr:`collate_fn` مخصص لتخصيص التجميع، على سبيل المثال، وسادة البيانات التسلسلية إلى طول الدفعة الأقصى. راجع "هذا القسم" <dataloader-collate_fn_> لمزيد من المعلومات حول :attr:`collate_fn`.

تعطيل التجميع التلقائي
^^^^^^^^^^^^^^^

في حالات معينة، قد يرغب المستخدمون في التعامل مع التجميع يدويًا في كود مجموعة البيانات، أو ببساطة تحميل عينات فردية. على سبيل المثال، قد يكون من الأرخص تحميل بيانات مجمعة مباشرة (مثل القراءات المجمعة من قاعدة بيانات أو قراءة قطع متجاورة من الذاكرة)، أو يعتمد حجم الدفعة على البيانات، أو تم تصميم البرنامج للعمل على عينات فردية. في ظل هذه السيناريوهات، من الأفضل عدم استخدام التجميع التلقائي (حيث يتم استخدام :attr:`collate_fn` لتجميع العينات)، ولكن السماح لمحمل البيانات بإرجاع كل عضو في كائن :attr:`dataset` مباشرة.

عندما يكون كل من :attr:`batch_size` و :attr:`batch_sampler` ``None`` (قيمة افتراضية لـ :attr:`batch_sampler` هي بالفعل ``None``)، يتم تعطيل التجميع التلقائي. تتم معالجة كل عينة يتم الحصول عليها من :attr:`dataset` باستخدام الدالة التي تم تمريرها كوسيط :attr:`collate_fn`.

**عندما يتم تعطيل التجميع التلقائي**، فإن :attr:`collate_fn` الافتراضي يحول ببساطة صفيفات NumPy إلى مصفوفات PyTorch، ويترك كل شيء آخر دون تغيير.

في هذه الحالة، يكون تحميل البيانات من مجموعة بيانات على طراز الخرائط مكافئًا تقريبًا لما يلي::

    for index in sampler:
        yield collate_fn(dataset[index])

وتحميل البيانات من مجموعة بيانات على الطراز القابل للتنفيذ مكافئ تقريبًا لما يلي::

    for data in iter(dataset):
        yield collate_fn(data)

راجع "هذا القسم" <dataloader-collate_fn_> لمزيد من المعلومات حول :attr:`collate_fn`.

.. _dataloader-collate_fn:

العمل مع :attr:`collate_fn`
^^^^^^^^^^^^^^^^^^^^^^^^^^

يختلف استخدام :attr:`collate_fn` قليلاً عندما يكون التجميع التلقائي ممكّنًا أو معطلاً.

**عندما يتم تعطيل التجميع التلقائي**، يتم استدعاء :attr:`collate_fn` مع كل عينة بيانات فردية، ويتم إنتاج الإخراج من مؤشر ترابط محمل البيانات. في هذه الحالة، يقوم :attr:`collate_fn` الافتراضي بتحويل صفيفات NumPy إلى مصفوفات PyTorch.

**عندما يتم تمكين التجميع التلقائي**، يتم استدعاء :attr:`collate_fn` بقائمة من عينات البيانات في كل مرة. من المتوقع أن يقوم بتجميع عينات الإدخال في دفعة لإنتاجها من مؤشر ترابط محمل البيانات. يصف باقي هذا القسم سلوك :attr:`collate_fn` الافتراضي (:func:`~torch.utils.data.default_collate`).

على سبيل المثال، إذا كانت كل عينة بيانات تتكون من صورة ثلاثية القنوات وتصنيف رقمي، أي أن كل عنصر في مجموعة البيانات يعيد زوجًا ``(image، class_index)``، فإن :attr:`collate_fn` الافتراضي يقوم بتجميع قائمة من هذه الأزواج في زوج واحد من صورة مجمعة ومصنف مجمع. على وجه الخصوص، فإن لـ :attr:`collate_fn` الافتراضي الخصائص التالية:

* دائمًا ما يضيف بُعدًا جديدًا كبعد دفعي.

* يقوم تلقائيًا بتحويل صفيفات NumPy والقيم الرقمية في Python إلى مصفوفات PyTorch.

* يحافظ على بنية البيانات، على سبيل المثال، إذا كانت كل عينة عبارة عن قاموس، فإنه ينتج قاموسًا بنفس مجموعة المفاتيح ولكن مع مصفوفات مجمعة كقيم (أو قوائم إذا لم يكن من الممكن تحويل القيم إلى مصفوفات). نفس الشيء بالنسبة للقوائم، والمصفوفات، والأزواج المسماة، وما إلى ذلك.

يمكن للمستخدمين استخدام :attr:`collate_fn` مخصص لتحقيق التجميع المخصص، على سبيل المثال، التجميع على طول بُعد آخر غير الأول، أو وسادة التسلسلات ذات الأطوال المختلفة، أو إضافة دعم لأنواع البيانات المخصصة.

إذا صادفت حالة يكون فيها إخراج :class:`~torch.utils.data.DataLoader` بأبعاد أو نوع مختلف عما تتوقعه، فقد تحتاج إلى التحقق من :attr:`collate_fn`.

تحميل البيانات أحادية ومتعددة العمليات
-------------------------

يستخدم :class:`~torch.utils.data.DataLoader` تحميل البيانات أحادية العملية بشكل افتراضي.

داخل عملية Python، يمنع 'Global Interpreter Lock (GIL) <https://wiki.python.org/moin/GlobalInterpreterLock>`_
من تحقيق الموازاة الكاملة الحقيقية لرمز Python عبر الخيوط. لتجنب حظر
رمز الحساب مع تحميل البيانات، توفر PyTorch مفتاحًا سهلاً لأداء
تحميل البيانات متعدد العمليات ببساطة عن طريق تعيين وسيط :attr:`num_workers`
إلى عدد صحيح موجب.

تحميل البيانات أحادية العملية (افتراضي)
^^^^^^^^^^^^^^^^^^^^^^^^^^

في هذا الوضع، يتم جلب البيانات في نفس العملية التي تم فيها تهيئة
:class:`~torch.utils.data.DataLoader`. لذلك، قد يؤدي تحميل البيانات إلى حظر الحساب.
ومع ذلك، قد يكون هذا الوضع مفضلًا عندما تكون الموارد المستخدمة
لمشاركة البيانات بين العمليات (مثل الذاكرة المشتركة، أو مؤشرات الملفات) محدودة،
أو عندما تكون مجموعة البيانات بأكملها صغيرة ويمكن تحميلها بالكامل في
الذاكرة. بالإضافة إلى ذلك، غالبًا ما يُظهر تحميل البيانات أحادية العملية
آثار الأخطاء الأكثر قابلية للقراءة، وبالتالي فهو مفيد للتصحيح.

تحميل البيانات متعدد العمليات
^^^^^^^^^^^^^^^^^^^

يؤدي تعيين وسيط :attr:`num_workers` كعدد صحيح موجب إلى
تشغيل تحميل البيانات متعدد العمليات مع العدد المحدد من عمليات عامل التحميل.

.. تحذير ::
   بعد عدة تكرارات، ستستهلك عمليات عامل التحميل نفس كمية ذاكرة وحدة المعالجة المركزية مثل العملية الأصلية
   لجميع كائنات Python في العملية الأصلية التي يتم الوصول إليها من عمليات العامل.
   يمكن أن يمثل ذلك مشكلة إذا كانت مجموعة البيانات تحتوي على الكثير من
   البيانات (على سبيل المثال، تقوم بتحميل قائمة كبيرة جدًا من أسماء الملفات في وقت إنشاء مجموعة البيانات)
   و/أو تستخدم الكثير من العمال (إجمالي
   استخدام الذاكرة هو "عدد العمال * حجم العملية الأصلية"). أبسط
   حل بديل هو استبدال كائنات Python بتمثيلات غير مرجعية مثل Pandas أو Numpy أو PyArrow
   الكائنات. تحقق من
   'issue #13246
   <https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662>`_
   لمزيد من التفاصيل حول سبب حدوث ذلك ورموز المثال لكيفية
   حل هذه المشكلات.

في هذا الوضع، في كل مرة يتم فيها إنشاء مؤشر لـ :class:`~torch.utils.data.DataLoader`
(على سبيل المثال، عند استدعاء ``enumerate(dataloader)``)، يتم إنشاء :attr:`num_workers`
عمليات عامل. في هذه المرحلة، يتم تمرير :attr:`dataset`،
:attr:`collate_fn`، و:attr:`worker_init_fn` إلى كل
عامل، حيث يتم استخدامها لتهيئة وجلب البيانات. وهذا يعني أن الوصول إلى مجموعة البيانات مع
إدخال/إخراج داخلي، والتحويلات
(بما في ذلك :attr:`collate_fn`) يتم تشغيله في عملية العامل.

:func:`torch.utils.data.get_worker_info()` يعيد معلومات مفيدة مختلفة
في عملية عامل (بما في ذلك معرف العامل، نسخة مجموعة البيانات، البذرة الأولية،
إلخ.)، ويعيد ``None`` في العملية الرئيسية. قد يستخدم المستخدمون هذه الدالة في
رمز مجموعة البيانات و/أو :attr:`worker_init_fn` لتكوين كل
نسخة مجموعة بيانات بشكل فردي، ولتحديد ما إذا كان الرمز يعمل في عملية عامل. على سبيل المثال، قد يكون هذا
مفيدًا بشكل خاص في تجزئة مجموعة البيانات.

بالنسبة لمجموعات البيانات على الطراز الخرائطي، تقوم العملية الرئيسية بتوليد المؤشرات باستخدام
:attr:`sampler` وإرسالها إلى العمال. لذا فإن أي تعشيق يتم في
العملية الرئيسية التي توجه التحميل عن طريق تعيين المؤشرات لتحميلها.

بالنسبة لمجموعات البيانات على الطراز القابل للتحديد، نظرًا لأن كل عملية عامل تحصل على نسخة من
كائن :attr:`dataset`، فإن التحميل متعدد العمليات البسيط غالبًا ما يؤدي إلى
بيانات مكررة. باستخدام :func:`torch.utils.data.get_worker_info()` و/أو
:attr:`worker_init_fn`، يمكن للمستخدمين تكوين كل نسخة بشكل مستقل. (راجع
وثائق :class:`~torch.utils.data.IterableDataset` لمعرفة كيفية تحقيق ذلك. ) لنفس الأسباب، في
التحميل متعدد العمليات، يسقط وسيط :attr:`drop_last`
الحجة آخر دفعة غير مكتملة من مجموعة البيانات القابلة للتحديد لكل عامل.

يتم إيقاف تشغيل العمال بمجرد الوصول إلى نهاية التكرار، أو عندما
يتم جمع القمامة في المؤشر.

.. تحذير ::
  بشكل عام، لا يوصى بإرجاع Tensor CUDA في التحميل متعدد العمليات
  بسبب العديد من الدقائق في استخدام CUDA ومشاركة Tensor CUDA في
  المعالجة المتعددة (راجع: ref: multiprocessing-cuda-note`). بدلاً من ذلك، نوصي
  باستخدام "تثبيت الذاكرة التلقائي <Memory Pinning_>`_ (أي تعيين
  :attr:`pin_memory=True`)، والذي يمكّن النقل السريع للبيانات إلى GPUs الممكّنة من CUDA.

السلوكيات الخاصة بالمنصة
"""""""""""""""""

نظرًا لاعتماد العمال على Python :py:mod:`multiprocessing`، يختلف سلوك بدء العامل على Windows عن Unix.

* على Unix، :func:`fork()` هي طريقة البدء الافتراضية لـ :py:mod:`multiprocessing`.
  باستخدام :func:`fork`، يمكن لعمال الأطفال عادةً الوصول إلى :attr:`dataset` و
  وظائف وسيط Python مباشرة من خلال مساحة العنوان المستنسخة.

* على Windows أو MacOS، :func:`spawn()` هي طريقة البدء الافتراضية لـ :py:mod:`multiprocessing`.
  باستخدام :func:`spawn`، يتم إطلاق مفسر آخر والذي يقوم بتشغيل النص البرمجي الرئيسي،
  تليها وظيفة العامل الداخلية التي تتلقى :attr:`dataset`،
  :attr:`collate_fn` والحجج الأخرى من خلال :py:mod:`pickle` التسلسل.

يعني هذا التسلسل المنفصل أنه يجب عليك اتخاذ خطوتين لضمان التوافق مع Windows أثناء استخدام
تحميل البيانات متعدد العمليات:

- قم بتغليف معظم كود النص البرمجي الرئيسي الخاص بك داخل كتلة "if __name__ == '__main__':"،
  للتأكد من أنه لا يتم تشغيله مرة أخرى (الأكثر احتمالًا لإنشاء خطأ) عندما يتم إطلاق كل عملية عامل. يمكنك وضع منطق إنشاء مجموعة البيانات و
  :class:`~torch.utils.data.DataLoader`
  هنا، حيث لا تحتاج إلى إعادة التنفيذ في العمال.

- تأكد من إعلان أي مخصص :attr:`collate_fn`، :attr:`worker_init_fn`
  أو رمز :attr:`dataset` كتعريفات على مستوى عالٍ، خارج
  فحص "main". يضمن هذا توفرها في عمليات العامل.
  (هذا مطلوب لأن الوظائف يتم تسلسلها كمراجع فقط، وليس "bytecode".)

.. _data-loading-randomness:

العشوائية في تحميل البيانات متعدد العمليات
""""""""""""""""""""""""""

بشكل افتراضي، سيكون لكل عامل بذرة PyTorch الخاصة به والتي تم تعيينها إلى "base_seed + worker_id"،
حيث "base_seed" هي عدد طويل تم إنشاؤه بواسطة العملية الرئيسية باستخدام RNG (وبالتالي،
استهلاك حالة RNG بشكل إلزامي) أو :attr:`generator` محدد. ومع ذلك، قد يتم تكرار البذور للمكتبات الأخرى عند تهيئة العمال،
مما يتسبب في إرجاع كل عامل لأرقام عشوائية متطابقة. (راجع: ref: هذا القسم <dataloader-workers-random-seed>` في الأسئلة الشائعة.).

في :attr:`worker_init_fn`، يمكنك الوصول إلى بذرة PyTorch المحددة لكل عامل
مع إما :func:`torch.utils.data.get_worker_info().seed <torch.utils.data.get_worker_info>`
أو :func:`torch.initial_seed()`، واستخدامه لزرع مكتبات أخرى قبل تحميل البيانات.

تثبيت الذاكرة
---------

تكون النسخ من المضيف إلى GPU أسرع بكثير عندما تنشأ من ذاكرة مثبتة (صفحة مؤمنة). راجع
:ref:`cuda-memory-pinning` لمزيد من التفاصيل حول متى وكيفية استخدام
الذاكرة المثبتة بشكل عام.

بالنسبة لتحميل البيانات، يؤدي تمرير :attr:`pin_memory=True` إلى
:class:`~torch.utils.data.DataLoader` سيضع تلقائيًا بيانات Tensor المستردة في الذاكرة المثبتة،
وبالتالي تمكين نقل البيانات بشكل أسرع إلى GPUs الممكّنة من CUDA.

تعترف منطق تثبيت الذاكرة الافتراضي فقط بـ Tensors والخرائط والمحددات التي تحتوي على Tensors.  بشكل افتراضي، إذا رأت منطق التثبيت
دفعة تكون نوعًا مخصصًا (والذي سيحدث إذا كان لديك :attr:`collate_fn` الذي يعيد نوع دفعة مخصص)، أو إذا كان كل
عنصر في دفعتك هو نوع مخصص، فلن تعترف منطق التثبيت بها، وستعيد تلك الدفعة (أو تلك
العناصر) دون تثبيت الذاكرة. لتمكين تثبيت الذاكرة لأنواع الدفعات أو البيانات المخصصة،
قم بتعريف طريقة :meth:`pin_memory` على نوع (أنواع) المخصص (المخصصين).

انظر المثال أدناه.

مثال ::

    class SimpleCustomBatch:
        def __init__(self، data):
            transposed_data = list(zip(*data))
            self.inp = torch.stack(transposed_data[0]، 0)
            self.tgt = torch.stack(transposed_data[1]، 0)

        # طريقة تثبيت الذاكرة المخصصة على النوع المخصص
        def pin_memory(self):
            self.inp = self.inp.pin_memory()
            self.tgt = self.tgt.pin_memory()
            return self

    def collate_wrapper(batch):
        return SimpleCustomBatch(batch)

    inps = torch.arange(10 * 5، dtype=torch.float32).view(10، 5)
    tgts = torch.arange(10 * 5، dtype=torch.float32).view(10، 5)
    dataset = TensorDataset(inps، tgts)

    loader = DataLoader(dataset، batch_size=2، collate_fn=collate_wrapper،
                        pin_memory=True)

    for batch_ndx، sample in enumerate(loader):
        print(sample.inp.is_pinned())
        print(sample.tgt.is_pinned())


.. autoclass:: DataLoader
.. autoclass:: Dataset
.. autoclass:: IterableDataset
.. autoclass:: TensorDataset
.. autoclass:: StackDataset
.. autoclass:: ConcatDataset
.. autoclass:: ChainDataset
.. autoclass:: Subset
.. autofunction:: torch.utils.data._utils.collate.collate
.. autofunction:: torch.utils.data.default_collate
.. autofunction:: torch.utils.data.default_convert
.. autofunction:: torch.utils.data.get_worker_info
.. autofunction:: torch.utils.data.random_split
.. autoclass:: torch.utils.data.Sampler
.. autoclass:: torch.utils.data.SequentialSampler
.. autoclass:: torch.utils.data.RandomSampler
.. autoclass:: torch.utils.data.SubsetRandomSampler
.. autoclass:: torch.utils.data.WeightedRandomSampler
.. autoclass:: torch.utils.data.BatchSampler
.. autoclass:: torch.utils.data.distributed.DistributedSampler


.. يتم توثيق هذه الوحدات النمطية كجزء من قائمة بيانات torch / يتم إدراجها هنا الآن حتى
.. لدينا إصلاح أوضح
.. py:module:: torch.utils.data.datapipes
.. py:module:: torch.utils.data.datapipes.dataframe
.. py:module:: torch.utils.data.datapipes.iter
.. py:module:: torch.utils.data.datapipes.map
.. py:module:: torch.utils.data.datapipes.utils