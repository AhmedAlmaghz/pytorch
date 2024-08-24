.. role:: hidden
    :class: hidden-section

حزمة الاتصال الموزع - torch.distributed

.. note ::
   يرجى الرجوع إلى `نظرة عامة على PyTorch Distributed <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
   للتعرف على جميع الميزات المتعلقة بالتدريب الموزع.

.. automodule:: torch.distributed
.. currentmodule:: torch.distributed

Backends
--------

تدعم ``torch.distributed`` ثلاث backends مدمجة، لكل منها قدرات مختلفة. يوضح الجدول أدناه الوظائف المتاحة
للاستخدام مع CPU / CUDA tensors. يدعم MPI CUDA فقط إذا كان التنفيذ المستخدم لبناء PyTorch يدعمه.

+----------------+-----------+-----------+-----------+
| Backend        | ``gloo``  | ``mpi``   | ``nccl``  |
+----------------+-----+-----+-----+-----+-----+-----+
| Device         | CPU | GPU | CPU | GPU | CPU | GPU |
+================+=====+=====+=====+=====+=====+=====+
| send           | ✓   | ✘   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| recv           | ✓   | ✘   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| broadcast      | ✓   | ✓   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| all_reduce     | ✓   | ✓   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| reduce         | ✓   | ✘   | ✓  Multiplier | ✘   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| all_gather     | ✓   | ✘   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| gather         | ✓   | ✘   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| scatter        | ✓   | ✘   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| reduce_scatter | ✘   | ✘   | ✘   | ✘   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| all_to_all     | ✘   | ✘   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| barrier        | ✓   | ✘   | ✓   | ؟   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+

Backends التي تأتي مع PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

تدعم حزمة PyTorch distributed أنظمة Linux (مستقرة) وMacOS (مستقرة) وWindows (تجريبية).
بشكل افتراضي لنظام Linux، يتم بناء backends Gloo وNCCL وتضمينها في PyTorch
distributed (NCCL فقط عند البناء باستخدام CUDA). MPI هو backend اختياري لا يمكن تضمينه إلا إذا قمت ببناء PyTorch من المصدر. (على سبيل المثال، بناء PyTorch على مضيف يحتوي على MPI
مثبت.)

.. note ::
   اعتبارًا من PyTorch v1.8، تدعم Windows جميع backends الاتصالات الجماعية باستثناء NCCL،
   إذا أشارت حجة `init_method` لـ :func:`init_process_group` إلى ملف، فيجب أن يلتزم
   بالمخطط التالي:

   - نظام الملفات المحلي، ``init_method="file:///d:/tmp/some_file"``
   - نظام الملفات المشترك، ``init_method="file://////{machine_name}/{share_folder_name}/some_file"``

   مثل منصة Linux، يمكنك تمكين TcpStore عن طريق تعيين متغيرات البيئة،
   MASTER_ADDR وMASTER_PORT.

أي backend يجب استخدامه؟
^^^^^^^^^^^^^^^^^^^^^

في الماضي، كنا نسأل في كثير من الأحيان: "أي backend يجب أن أستخدم؟".

- قاعدة عامة

  - استخدم backend NCCL للتدريب الموزع **GPU**
  - استخدم backend Gloo للتدريب الموزع **CPU**.

- مضيفي GPU مع اتصال InfiniBand

  - استخدم NCCL، لأنه backend الوحيد الذي يدعم حاليًا
    InfiniBand وGPUDirect.

- مضيفي GPU مع اتصال Ethernet

  - استخدم NCCL، لأنه يوفر حاليًا أفضل أداء للتدريب الموزع لـ GPU
    خاصة للتدريب الموزع أحادي العقدة أو متعدد العقد. إذا واجهتك أي مشكلة مع
    NCCL، استخدم Gloo كخيار احتياطي. (ملاحظة: Gloo أبطأ حاليًا من NCCL لـ GPUs.)

- مضيفي CPU مع اتصال InfiniBand

  - إذا كان InfiniBand الخاص بك قد مكن IP عبر IB، فاستخدم Gloo، وإلا
    استخدم MPI بدلاً من ذلك. نخطط لإضافة دعم InfiniBand لـ
    Gloo في الإصدارات القادمة.

- مضيفي CPU مع اتصال Ethernet

  - استخدم Gloo، ما لم يكن لديك أسباب محددة لاستخدام MPI.

متغيرات البيئة الشائعة
^^^^^^^^^^^^^^

اختيار واجهة الشبكة لاستخدامها
"""""""""""""""""""

بشكل افتراضي، سيحاول كل من backends NCCL وGloo العثور على واجهة الشبكة الصحيحة لاستخدامها.
إذا كانت الواجهة التي تم اكتشافها تلقائيًا غير صحيحة، فيمكنك تجاوزها باستخدام متغيرات البيئة التالية
(قابلة للتطبيق على backend المقابل):

* **NCCL_SOCKET_IFNAME**، على سبيل المثال ``export NCCL_SOCKET_IFNAME=eth0``
* **GLOO_SOCKET_IFNAME**، على سبيل المثال ``export GLOO_SOCKET_IFNAME=eth0``

إذا كنت تستخدم backend Gloo، فيمكنك تحديد واجهات متعددة عن طريق الفصل
عنهم بفاصلة، مثل هذا: ``export GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3``.
سيقوم backend بتنفيذ العمليات بطريقة مستديرة روبين عبر هذه الواجهات. من الضروري أن تحدد جميع العمليات نفس عدد الواجهات في هذا المتغير.

متغيرات بيئة NCCL الأخرى
""""""""""""""""""

**التصحيح** - في حالة فشل NCCL، يمكنك تعيين ``NCCL_DEBUG=INFO`` لطباعة رسالة تحذير صريحة
وكذلك معلومات التهيئة الأساسية لـ NCCL.

يمكنك أيضًا استخدام ``NCCL_DEBUG_SUBSYS`` للحصول على مزيد من التفاصيل حول جانب محدد
من NCCL. على سبيل المثال، ستطبع ``NCCL_DEBUG_SUBSYS=COLL`` سجلات
المكالمات الجماعية، والتي قد تكون مفيدة عند تصحيح الأعطال، خاصة تلك
التي تسببها عدم تطابق نوع الرسالة الجماعية أو حجمها. في حالة فشل اكتشاف الطوبولوجيا، سيكون من المفيد تعيين ``NCCL_DEBUG_SUBSYS=GRAPH``
للتفتيش على نتيجة الكشف التفصيلية وحفظها كمرجع إذا كانت هناك حاجة إلى مزيد من المساعدة
من فريق NCCL.

**ضبط الأداء** - يقوم NCCL بضبط تلقائي بناءً على اكتشاف الطوبولوجيا الخاصة به لتوفير جهد الضبط للمستخدمين
. على بعض الأنظمة المستندة إلى المقبس، قد يحاول المستخدمون ضبطها على الرغم من ذلك
``NCCL_SOCKET_NTHREADS`` و ``NCCL_NSOCKS_PERTHREAD`` لزيادة عرض النطاق الترددي للشبكة المقبس. تم ضبط هذين متغيري البيئة مسبقًا بواسطة NCCL
لمزودي الخدمات السحابية مثل AWS أو GCP.

للاطلاع على القائمة الكاملة لمتغيرات بيئة NCCL، يرجى الرجوع إلى
`الوثائق الرسمية لـ NVIDIA NCCL <https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html>`_


.. _distributed-basics:

أساسيات
------

توفر حزمة `torch.distributed` دعم PyTorch وبدائيات الاتصال
للتعددية المتوازية عبر عدة عقد حسابية تعمل على جهاز واحد أو أكثر
الآلات. تبني فئة :func:`torch.nn.parallel.DistributedDataParallel` على هذا
الوظائف لتوفير التدريب الموزع المتزامن كغلاف حول أي
نموذج PyTorch. يختلف هذا عن أنواع التوازي التي يوفرها
:doc:`multiprocessing` و :func:`torch.nn.DataParallel` في أنه يدعم
العديد من الأجهزة المتصلة بالشبكة وفي أن المستخدم يجب أن يبدأ صراحةً نسخة منفصلة
من النص الرئيسي لنص البرنامج النصي للتدريب لكل عملية.

في حالة العقدة الواحدة المتزامنة، قد يكون لـ `torch.distributed` أو
:func:`torch.nn.parallel.DistributedDataParallel` مزايا على أساليب أخرى
للتعددية، بما في ذلك :func:`torch.nn.DataParallel`:

* تحتفظ كل عملية بمثبتها الخاص وتنفذ خطوة تحسين كاملة مع كل
  تكرار. في حين أن هذا قد يبدو مكرراً، لأن التدرجات قد تم جمعها بالفعل
  معًا وتمت متوسطها عبر العمليات وهي نفسها لكل عملية، وهذا يعني
  أن خطوة بث المعلمة غير مطلوبة، مما يقلل من الوقت المستغرق في نقل المنسوجات بين
  العقد.
* تحتوي كل عملية على مترجم Python مستقل، مما يؤدي إلى القضاء على النفقات العامة الإضافية للمترجم
  وعرقلة "GIL" التي تأتي من تشغيل عدة خيوط تنفيذ أو نماذج
  المكررات أو GPUs من عملية Python واحدة. هذا مهم بشكل خاص للنماذج التي
  الاستخدام المكثف لوقت تشغيل Python، بما في ذلك النماذج ذات الطبقات المتكررة أو العديد من المكونات الصغيرة.

التهيئة
-----

يجب تهيئة الحزمة باستخدام دالة :func:`torch.distributed.init_process_group`
أو :func:`torch.distributed.device_mesh.init_device_mesh` قبل استدعاء أي طرق أخرى.
كلاهما يمنع حتى تنضم جميع العمليات.

.. autofunction:: is_available

.. autofunction:: init_process_group

.. autofunction:: torch.distributed.device_mesh.init_device_mesh

.. autofunction:: is_initialized

.. autofunction:: is_mpi_available

.. autofunction:: is_nccl_available

.. autofunction:: is_gloo_available

.. autofunction:: is_torchelastic_launched

--------------------------------------------------------------------------------

حاليا، هناك ثلاث طرق للتهيئة مدعومة:

التهيئة باستخدام بروتوكول TCP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

هناك طريقتان للتهيئة باستخدام بروتوكول TCP، وكلتاهما تتطلب عنوان شبكة يمكن الوصول إليه من جميع العمليات (processes) وحجم "world_size" المرغوب فيه. الطريقة الأولى تتطلب تحديد عنوان ينتمي إلى العملية ذات الرتبة 0 (rank 0). تتطلب طريقة التهيئة هذه أن يكون لدى جميع العمليات رتب محددة يدويًا.

ملاحظة: لم يعد عنوان البث المجموعاتي (multicast address) مدعومًا في الحزمة الموزعة (distributed package) الأحدث. كما أن "group_name" مهملة أيضًا.

::

    import torch.distributed as dist

    # استخدام عنوان إحدى الآلات
    dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                            rank=args.rank, world_size=4)

التهيئة باستخدام نظام ملفات مشترك
^^^^^^^^^^^^^^^^^^^^^^^^

تستخدم طريقة التهيئة الأخرى نظام ملفات مشتركًا ومرئيًا من جميع الآلات في مجموعة، إلى جانب حجم "world_size" المرغوب فيه. يجب أن يبدأ عنوان URL (محدد موقع الموارد الموحد) بـ "file://" وأن يحتوي على مسار إلى ملف غير موجود (في دليل موجود) على نظام ملفات مشترك. ستعمل التهيئة باستخدام نظام الملفات على إنشاء هذا الملف تلقائيًا إذا لم يكن موجودًا، ولكنها لن تقوم بحذفه. لذلك، من مسؤوليتك التأكد من تنظيف الملف قبل إجراء المكالمة التالية لـ :func:`init_process_group` على نفس مسار/اسم الملف.

ملاحظة: لم يعد تعيين الرتبة التلقائي (automatic rank assignment) مدعومًا في الحزمة الموزعة الأحدث، كما أن "group_name" مهملة أيضًا.

.. warning::
    تفترض هذه الطريقة أن نظام الملفات يدعم القفل باستخدام "fcntl" - معظم الأنظمة المحلية وأنظمة ملفات الشبكة (NFS) تدعمه.

.. warning::
    ستعمل هذه الطريقة دائمًا على إنشاء الملف وتحاول بذل قصارى جهدها لتنظيفه وإزالته في نهاية البرنامج. وبعبارة أخرى، فإن كل تهيئة باستخدام طريقة التهيئة باستخدام الملف تتطلب ملفًا جديدًا فارغًا لكي تنجح عملية التهيئة. إذا تم استخدام نفس الملف الذي استخدمته عملية التهيئة السابقة (والذي لم يتم تنظيفه) مرة أخرى، فقد يؤدي ذلك إلى حدوث سلوك غير متوقع ويمكن أن يتسبب في حدوث توقف تام (deadlocks) وأخطاء. لذلك، على الرغم من أن هذه الطريقة ستحاول بذل قصارى جهدها لتنظيف الملف، إذا حدث أن فشلت عملية الحذف التلقائي، فمن مسؤوليتك التأكد من إزالة الملف في نهاية التدريب لمنع إعادة استخدام نفس الملف مرة أخرى في المرة التالية. هذا مهم بشكل خاص إذا كنت تخطط لاستدعاء :func:`init_process_group` عدة مرات على نفس اسم الملف. وبعبارة أخرى، إذا لم يتم إزالة/تنظيف الملف واستدعاء :func:`init_process_group` مرة أخرى على ذلك الملف، فمن المتوقع حدوث أخطاء. القاعدة العامة هنا هي التأكد من أن الملف غير موجود أو فارغ في كل مرة يتم فيها استدعاء :func:`init_process_group`.

::

    import torch.distributed as dist

    # يجب تحديد الرتبة دائمًا
    dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                            world_size=4, rank=args.rank)

التهيئة باستخدام متغيرات البيئة
^^^^^^^^^^^^^^^^^^^

ستقوم هذه الطريقة بقراءة التهيئة من متغيرات البيئة، مما يسمح بتخصيص كيفية الحصول على المعلومات بشكل كامل. المتغيرات التي يجب تعيينها هي:

* ``MASTER_PORT`` - مطلوب؛ يجب أن يكون منفذًا متاحًا على الآلة ذات الرتبة 0
* ``MASTER_ADDR`` - مطلوب (باستثناء الرتبة 0)؛ عنوان العقدة ذات الرتبة 0
* ``WORLD_SIZE`` - مطلوب؛ يمكن تعيينه إما هنا، أو في مكالمة دالة التهيئة
* ``RANK`` - مطلوب؛ يمكن تعيينه إما هنا، أو في مكالمة دالة التهيئة

سيتم استخدام الآلة ذات الرتبة 0 لإعداد جميع الاتصالات.

هذه هي الطريقة الافتراضية، مما يعني أنه لا يلزم تحديد "init_method" (أو يمكن أن يكون "env://").

ما بعد التهيئة
----------

بمجرد تشغيل :func:`torch.distributed.init_process_group`، يمكن استخدام الوظائف التالية. للتحقق مما إذا كان قد تم تهيئة مجموعة العمليات بالفعل، استخدم الدالة :func:`torch.distributed.is_initialized`.

.. autoclass:: Backend
    :members:

.. autofunction:: get_backend

.. autofunction:: get_rank

.. autofunction:: get_world_size

الإيقاف
-----

من المهم تنظيف الموارد عند الخروج عن طريق استدعاء :func:`destroy_process_group`.

أبسط نمط يمكن اتباعه هو تدمير كل مجموعات العمليات والخلفيات (backends) عن طريق استدعاء :func:`destroy_process_group()` مع القيمة الافتراضية "None" لحجة "group"، في نقطة في نص التدريب حيث لم تعد الاتصالات مطلوبة، وعادة ما تكون بالقرب من نهاية الدالة "main()". يجب إجراء المكالمة مرة واحدة لكل عملية مدرب، وليس على مستوى مطلق العملية الخارجي.

إذا لم يتم استدعاء :func:`destroy_process_group` من قبل جميع الرتب في مجموعة عمليات ضمن مدة المهلة الزمنية، خاصة عند وجود مجموعات عمليات متعددة في التطبيق، على سبيل المثال، للموازاة متعددة الأبعاد، فقد تحدث توقفات عند الخروج. ويرجع ذلك إلى أن دالة الهدم لـ "ProcessGroupNCCL" تستدعي "ncclCommAbort"، والتي يجب استدعاؤها بشكل جماعي، ولكن ترتيب استدعاء دالة الهدم لـ "ProcessGroupNCCL" إذا تم استدعاؤها بواسطة جامع القمامة (GC) في بايثون غير محدد. يساعد استدعاء :func:`destroy_process_group` من خلال ضمان استدعاء "ncclCommAbort" بترتيب متسق عبر الرتب، وتجنب استدعاء "ncclCommAbort" أثناء دالة الهدم لـ "ProcessGroupNCCL".

إعادة التهيئة
^^^^^^^^

يمكن أيضًا استخدام "destroy_process_group" لتدمير مجموعات العمليات الفردية. إحدى حالات الاستخدام قد تكون التدريب المتسامح مع الأخطاء (fault-tolerant training)، حيث قد يتم تدمير مجموعة عمليات ثم تهيئة مجموعة جديدة أثناء وقت التشغيل. في هذه الحالة، من الضروري مزامنة عمليات التدريب باستخدام بعض الوسائل الأخرى غير بدائيات "torch.distributed" _بعد_ استدعاء التدمير وقبل التهيئة اللاحقة. هذا السلوك غير مدعوم/غير مجرب حاليًا، بسبب صعوبة تحقيق هذه المزامنة، ويعتبر مشكلة معروفة. يرجى إرسال طلب سحب (pull request) أو طلب ميزة (feature request) على جيثب (GitHub) إذا كان هذا الاستخدام يمنعك.

--------------------------------------------------------------------------------

التخزين الموزع القائم على القيمة-المفتاح
--------------------------

تأتي الحزمة الموزعة مع متجر قيمة-مفتاح موزع، والذي يمكن استخدامه لمشاركة المعلومات بين العمليات في المجموعة وكذلك لتهيئة الحزمة الموزعة في :func:`torch.distributed.init_process_group` (عن طريق إنشاء المتجر بشكل صريح كبديل لتحديد "init_method"). هناك 3 خيارات لمتاجر القيمة-المفتاح: :class:`~torch.distributed.TCPStore`، و:class:`~torch.distributed.FileStore`، و:class:`~torch.distributed.HashStore`.

.. autoclass:: Store
.. autoclass:: TCPStore
.. autoclass:: HashStore
.. autoclass:: FileStore
.. autoclass:: PrefixStore

.. autofunction:: torch.distributed.Store.set
.. autofunction:: torch.distributed.Store.get
.. autofunction:: torch.distributed.Store.add
.. autofunction:: torch.distributed.Store.compare_set
.. autofunction:: torch.distributed.Store.wait
.. autofunction:: torch.distributed.Store.num_keys
.. autofunction:: torch.distributed.Store.delete_key
.. autofunction:: torch.distributed.Store.set_timeout

المجموعات
------

بشكل افتراضي، تعمل العمليات الجماعية على المجموعة الافتراضية (المعروفة أيضًا باسم العالم) وتتطلب من جميع العمليات الدخول في مكالمة الدالة الموزعة. ومع ذلك، يمكن لبعض أعباء العمل الاستفادة من الاتصال الأكثر دقة. وهنا تأتي مجموعات التوزيع للعب. يمكن استخدام الدالة :func:`~torch.distributed.new_group` لإنشاء مجموعات جديدة، مع مجموعات فرعية تعسفية من جميع العمليات. تقوم الدالة بإرجاع مقبض مجموعة غير شفاف يمكن تمريره كحجة "group" إلى جميع العمليات الجماعية (العمليات الجماعية هي وظائف موزعة لتبادل المعلومات في أنماط برمجة معينة معروفة).


.. autofunction:: new_group

.. autofunction:: get_group_rank

.. autofunction:: get_global_rank

.. autofunction:: get_process_group_ranks


شبكة الأجهزة
----------

شبكة الأجهزة (DeviceMesh) هي طبقة تجريد أعلى تقوم بإدارة مجموعات العمليات (أو موصلات NCCL). تسمح للمستخدم بإنشاء مجموعات عمليات بين العقد وداخل العقد بسهولة دون القلق بشأن كيفية تعيين الرتب بشكل صحيح لمجموعات العمليات الفرعية المختلفة، كما تساعد في إدارة مجموعات العمليات الموزعة بسهولة. يمكن استخدام الدالة :func:`~torch.distributed.device_mesh.init_device_mesh` لإنشاء شبكة أجهزة جديدة، مع شكل شبكة يصف طوبولوجيا الجهاز.

.. autoclass:: torch.distributed.device_mesh.DeviceMesh

الاتصال من نقطة إلى نقطة
-------------------

.. autofunction:: send

.. autofunction:: recv

تعيد الدالتان :func:`~torch.distributed.isend` و:func:`~torch.distributed.irecv` كائنات طلب موزعة عند استخدامها. بشكل عام، نوع هذا الكائن غير محدد لأنه لا ينبغي أبدًا إنشاؤه يدويًا، ولكنه يدعم طريقتين مضمونتين:

* ``is_completed()`` - تعيد القيمة "True" إذا تم الانتهاء من العملية
* ``wait()`` - ستؤدي إلى توقف العملية حتى تنتهي العملية.
  يتم ضمان إعادة "is_completed()" للقيمة "True" بمجرد إعادتها.

.. autofunction:: isend

.. autofunction:: irecv

.. autofunction:: send_object_list

.. autofunction:: recv_object_list

.. autofunction:: batch_isend_irecv

.. autoclass:: P2POp

العمليات الجماعية المتزامنة وغير المتزامنة
-----------------------------
تدعم كل دالة من دوال العمليات الجماعية النوعين التاليين من العمليات، وذلك حسب إعداد مؤشر "async_op" الذي يتم تمريره إلى العملية الجماعية:

**العملية المتزامنة** - الوضع الافتراضي، عندما يكون "async_op" مضبوطًا على "False". عندما تعيد الدالة النتيجة، يكون من المضمون أن العملية الجماعية قد تم تنفيذها. في حالة عمليات CUDA، لا يمكن ضمان إتمام عملية CUDA، لأن عمليات CUDA غير متزامنة. وبالنسبة للعمليات الجماعية على وحدة المعالجة المركزية (CPU)، فإن أي استدعاءات لاحقة للدوال التي تستخدم ناتج الاستدعاء الجماعي ستتصرف كما هو متوقع. وبالنسبة للعمليات الجماعية على CUDA، فإن استدعاءات الدوال التي تستخدم النتيجة على نفس تدفق CUDA ستتصرف كما هو متوقع. ويجب على المستخدمين مراعاة المزامنة في حالة التشغيل على تدفقات مختلفة. لمزيد من التفاصيل حول دلاليات CUDA مثل مزامنة التدفق، راجع `دلاليات CUDA <https://pytorch.org/docs/stable/notes/cuda.html>`__.
راجع النص البرمجي أدناه لمشاهدة أمثلة على الاختلافات في هذه الدلالات لعمليات وحدة المعالجة المركزية (CPU) وCUDA.

**العملية غير المتزامنة** - عندما يكون "async_op" مضبوطًا على "True". تعيد دالة العملية الجماعية كائن طلب موزع. بشكل عام، لا تحتاج إلى إنشائه يدويًا ومن المضمون أنه يدعم طريقتين:

* ``is_completed()`` - في حالة العمليات الجماعية على وحدة المعالجة المركزية (CPU)، تعيد القيمة "True" إذا تم إتمام العملية. وفي حالة عمليات CUDA، تعيد القيمة "True" إذا تم وضع العملية بنجاح في طابور انتظار تدفق CUDA وأصبح من الممكن استخدام النتيجة على التدفق الافتراضي دون مزامنة إضافية.
* ``wait()`` - في حالة العمليات الجماعية على وحدة المعالجة المركزية (CPU)، ستؤدي إلى تعليق العملية حتى يتم إتمام العملية. وفي حالة العمليات الجماعية على CUDA، ستؤدي إلى التعليق حتى يتم وضع العملية بنجاح في طابور انتظار تدفق CUDA وأصبح من الممكن استخدام النتيجة على التدفق الافتراضي دون مزامنة إضافية.
* ``get_future()`` - تعيد كائن "torch._C.Future". مدعوم لـ NCCL، كما أنه مدعوم لمعظم العمليات على GLOO و MPI، باستثناء العمليات من ند لند.
  ملاحظة: مع استمرارنا في اعتماد Futures ودمج واجهات برمجة التطبيقات (APIs)، قد تصبح مكالمة "get_future()" زائدة عن الحاجة.

**مثال**

يمكن استخدام الكود التالي كمرجع فيما يتعلق بدلالات عمليات CUDA عند استخدام العمليات الجماعية الموزعة.
فهو يوضح الحاجة الصريحة إلى المزامنة عند استخدام النواتج الجماعية على تدفقات CUDA مختلفة:

::

    # يتم تشغيل الكود على كل رتبة.
    dist.init_process_group("nccl", rank=rank, world_size=2)
    output = torch.tensor([rank]).cuda(rank)
    s = torch.cuda.Stream()
    handle = dist.all_reduce(output, async_op=True)
    # تضمن عملية الانتظار وضع العملية في طابور الانتظار، ولكن ليس بالضرورة إتمامها.
    handle.wait()
    # استخدام النتيجة على تدفق غير افتراضي.
    with torch.cuda.stream(s):
        s.wait_stream(torch.cuda.default_stream())
        output.add_(100)
    if rank == 0:
        # إذا تم حذف الاستدعاء الصريح لـ wait_stream، فإن النتيجة أدناه ستكون
        # بشكل غير محدد إما 1 أو 101، اعتمادًا على ما إذا كانت عملية الجمع من جميع المصادر (all_reduce) قد استبدلت
        # القيمة بعد إتمام عملية الإضافة.
        print(output)


دوال العمليات الجماعية
----------------

.. autofunction:: broadcast

.. autofunction:: broadcast_object_list

.. autofunction:: all_reduce

.. autofunction:: reduce

.. autofunction:: all_gather

.. autofunction:: all_gather_into_tensor

.. autofunction:: all_gather_object

.. autofunction:: gather

.. autofunction:: gather_object

.. autofunction:: scatter

.. autofunction:: scatter_object_list

.. autofunction:: reduce_scatter

.. autofunction:: reduce_scatter_tensor

.. autofunction:: all_to_all_single

.. autofunction:: all_to_all

.. autofunction:: barrier

.. autofunction:: monitored_barrier

.. autoclass:: Work

.. autoclass:: ReduceOp

.. class:: reduce_op

    فئة منتهية الصلاحية تشبه الفئات التعدادية لعمليات التخفيض: ``SUM``، ``PRODUCT``،
    ``MIN``، و ``MAX``.

    يوصى باستخدام الفئة :class:`~torch.distributed.ReduceOp` بدلاً من ذلك.

تحليل ملفات تعريف عمليات الاتصال الجماعي
------------------------------

يرجى ملاحظة أنه يمكنك استخدام ``torch.profiler`` (موصى به، متوفر فقط بعد الإصدار 1.8.1) أو ``torch.autograd.profiler`` لتحليل ملفات تعريف عمليات الاتصال الجماعي والاتصال من نقطة إلى نقطة المذكورة هنا. جميع المكتبات الخلفية المدمجة (``gloo``،
``nccl``، ``mpi``) مدعومة، وسيتم عرض استخدام الاتصال الجماعي كما هو متوقع في ملفات تعريف الإخراج/التتبع. ويتم تحليل الكود الخاص بك بنفس طريقة تحليل أي مشغل Torch عادي:

::

    import torch
    import torch.distributed as dist
    with torch.profiler():
        tensor = torch.randn(20, 10)
        dist.all_reduce(tensor)

يرجى الرجوع إلى `توثيق المحلل <https://pytorch.org/docs/main/profiler.html>`__ للحصول على نظرة عامة كاملة على ميزات المحلل.


دوال العمليات الجماعية متعددة وحدات معالجة الرسوميات (GPU)
-----------------------------------------

.. warning::
    تم إيقاف استخدام دوال العمليات الجماعية متعددة وحدات معالجة الرسوميات (GPU) (والتي تعني استخدام وحدات معالجة رسوميات متعددة لكل خيط وحدة المعالجة المركزية). واعتبارًا من اليوم، فإن نموذج البرمجة المفضل في PyTorch Distributed هو استخدام وحدة واحدة لكل خيط، كما هو موضح في واجهات برمجة التطبيقات (APIs) في هذه الوثيقة. إذا كنت مطورًا لمكتبة خلفية وترغب في دعم أجهزة متعددة لكل خيط، يرجى التواصل مع القائمين على صيانة PyTorch Distributed.


.. _distributed-launch:

المكتبات الخلفية من أطراف أخرى
--------------------

بالإضافة إلى المكتبات الخلفية المدمجة GLOO/MPI/NCCL، تدعم PyTorch Distributed
المكتبات الخلفية من أطراف أخرى من خلال آلية تسجيل وقت التشغيل.
للاطلاع على المراجع حول كيفية تطوير مكتبة خلفية من طرف آخر من خلال ملحق C++،
يرجى الرجوع إلى `الدروس - ملحقات C++ و CUDA المخصصة <https://pytorch.org/
tutorials/advanced/cpp_extension.html>`_ و
``test/cpp_extensions/cpp_c10d_extension.cpp``. تعتمد قدرات المكتبات الخلفية من أطراف أخرى على تنفيذها الخاص.

تشتق المكتبة الخلفية الجديدة من الفئة ``c10d::ProcessGroup`` وتسجل اسم المكتبة الخلفية والواجهة المنشئة لها من خلال الدالة :func:`torch.distributed.Backend.register_backend`
عند استيرادها.

عند استيراد هذه المكتبة الخلفية يدويًا واستدعاء الدالة :func:`torch.distributed.init_process_group`
مع اسم المكتبة الخلفية المقابل، تعمل حزمة "torch.distributed" على المكتبة الخلفية الجديدة.

.. warning::
    إن دعم المكتبات الخلفية من أطراف أخرى هو دعم تجريبي وقد يخضع للتغيير.

أداة الإطلاق
--------

توفر حزمة `torch.distributed` أيضًا أداة مساعدة للإطلاق في
`torch.distributed.launch`. يمكن استخدام أداة المساعدة هذه لإطلاق
عمليات متعددة لكل عقدة في التدريب الموزع.


.. automodule:: torch.distributed.launch


أداة الاستدعاء
---------

توفر حزمة :ref:`multiprocessing-doc` أيضًا دالة "spawn"
في :func:`torch.multiprocessing.spawn`. يمكن استخدام دالة المساعدة
هذه لاستدعاء عمليات متعددة. تعمل من خلال تمرير الدالة التي تريد تشغيلها،
وتقوم باستدعاء N عملية لتشغيلها. يمكن استخدام ذلك أيضًا في التدريب الموزع.

للاطلاع على المراجع حول كيفية استخدامها، يرجى الرجوع إلى `مثال PyTorch - تنفيذ ImageNet
<https://github.com/pytorch/examples/tree/master/imagenet>`_

يرجى ملاحظة أن هذه الدالة تتطلب الإصدار 3.4 من Python أو أعلى.

تصحيح أخطاء تطبيقات ``torch.distributed``
تحتوي مكتبة "تورتش" على مجموعة من الأدوات لمساعدتك في تصحيح تطبيقاتك الموزعة. يمكن أن يكون تصحيح التطبيقات الموزعة أمرًا صعبًا بسبب صعوبة فهم التعليق أو التوقف أو السلوك غير المتسق عبر الرتب. فيما يلي بعض الأدوات التي يمكن أن تساعدك في تصحيح تطبيقاتك:

------------------------------------------------------

**نقطة توقف بايثون (Python Breakpoint)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

من المريح للغاية استخدام أداة تصحيح أخطاء بايثون في بيئة موزعة، ولكن نظرًا لأنها لا تعمل بشكل افتراضي، فإن الكثير من المستخدمين لا يستخدمونها على الإطلاق. تقدم مكتبة "تورتش" غلافًا مخصصًا حول أداة "بي دي بي" (pdb) التي تسهل عملية التصحيح.

تجعل ``torch.distributed.breakpoint`` هذه العملية سهلة. داخليًا، يقوم بتخصيص سلوك نقطة التوقف في ``pdb`` بطريقتين، ولكنه يعمل بشكل طبيعي مثل أداة "بي دي بي".

1. يقوم بتشغيل أداة التصحيح فقط على رتبة واحدة (يحددها المستخدم).
2. يضمن توقف جميع الرتب الأخرى، وذلك باستخدام ``torch.distributed.barrier()`` التي ستطلق بمجرد إصدار الرتبة التي يتم تصحيحها أمر "متابعة".
3. يعيد توجيه الإدخال المعياري من العملية الفرعية بحيث يتصل بنهاية طرفيتك.

لاستخدامه، ما عليك سوى تنفيذ ``torch.distributed.breakpoint(rank)`` على جميع الرتب، باستخدام نفس القيمة لـ ``rank`` في كل حالة.

**حاجز مراقب (Monitored Barrier)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

اعتبارًا من الإصدار 1.10، توجد وظيفة ``torch.distributed.monitored_barrier`` كبديل لـ ``torch.distributed.barrier`` والتي تفشل بمعلومات مفيدة حول الرتبة التي قد تكون معطلة عند التعطل، أي عندما لا تقوم جميع الرتب بالاتصال بوظيفة ``torch.distributed.monitored_barrier`` ضمن المهلة الزمنية المحددة. تقوم وظيفة ``torch.distributed.monitored_barrier`` بتنفيذ حاجز على جانب المضيف باستخدام أوامر الاتصال الأساسية "إرسال" و"استقبال" بطريقة مشابهة للتأكيدات، مما يسمح للرتبة 0 بالإبلاغ عن الرتب التي فشلت في التأكيد على الحاجز في الوقت المحدد. على سبيل المثال، ضع في اعتبارك الوظيفة التالية حيث تفشل الرتبة 1 في الاتصال بوظيفة ``torch.distributed.monitored_barrier`` (في الممارسة العملية، قد يكون ذلك بسبب وجود خلل في التطبيق أو تعليق في عملية جماعية سابقة):

::

    import os
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


    def worker(rank):
        dist.init_process_group("nccl", rank=rank, world_size=2)
        # monitored barrier requires gloo process group to perform host-side sync.
        group_gloo = dist.new_group(backend="gloo")
        if rank not in [1]:
            dist.monitored_barrier(group=group_gloo, timeout=timedelta(seconds=2))


    if __name__ == "__main__":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        mp.spawn(worker, nprocs=2, args=())

يتم إنتاج رسالة الخطأ التالية على الرتبة 0، مما يسمح للمستخدم بتحديد الرتب المعطلة المحتملة وإجراء المزيد من التحقيق:

::

  RuntimeError: Rank 1 failed to pass monitoredBarrier in 2000 ms
   Original exception:
  [gloo/transport/tcp/pair.cc:598] Connection closed by peer [2401:db00:eef0:1100:3560:0:1c05:25d]:8594


**TORCH_DISTRIBUTED_DEBUG**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

عند تعيين ``TORCH_CPP_LOG_LEVEL=INFO``، يمكن استخدام متغير البيئة ``TORCH_DISTRIBUTED_DEBUG`` لتشغيل تسجيل المعلومات المفيدة وفحوصات المزامنة الجماعية الإضافية للتأكد من أن جميع الرتب متزامنة بشكل مناسب. يمكن تعيين ``TORCH_DISTRIBUTED_DEBUG`` إلى إما ``OFF`` (افتراضي)، أو ``INFO``، أو ``DETAIL`` اعتمادًا على مستوى التصحيح المطلوب. يرجى ملاحظة أن الخيار الأكثر تفصيلاً، وهو ``DETAIL``، قد يؤثر على أداء التطبيق، وبالتالي يجب استخدامه فقط عند تصحيح المشكلات.

سيؤدي تعيين ``TORCH_DISTRIBUTED_DEBUG=INFO`` إلى تسجيل معلومات التصحيح الإضافية عند تهيئة النماذج التي تم تدريبها باستخدام ``torch.nn.parallel.DistributedDataParallel``. وسيؤدي تعيين ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` أيضًا إلى تسجيل إحصائيات الأداء في الوقت الفعلي لعدد محدد من الحلقات. تتضمن إحصائيات وقت التشغيل هذه بيانات مثل وقت الحساب للأمام، ووقت الحساب للخلف، ووقت اتصال التدرج، وما إلى ذلك. على سبيل المثال، بالنظر إلى التطبيق التالي:

::

    import os

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


    class TwoLinLayerNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(10, 10, bias=False)
            self.b = torch.nn.Linear(10, 1, bias=False)

        def forward(self, x):
            a = self.a(x)
            b = self.b(x)
            return (a, b)


    def worker(rank):
        dist.init_process_group("nccl", rank=rank, world_size=2)
        torch.cuda.set_device(rank)
        print("init model")
        model = TwoLinLayerNet().cuda()
        print("init ddp")
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        inp = torch.randn(10, 10).cuda()
        print("train")

        for _ in range(20):
            output = ddp_model(inp)
            loss = output[0] + output[1]
            loss.sum().backward()


    if __name__ == "__main__":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ[
            "TORCH_DISTRIBUTED_DEBUG"
        ] = "DETAIL"  # set to DETAIL for runtime logging.
        mp.spawn(worker, nprocs=2, args=())

يتم تقديم سجلات التصحيح التالية في وقت التهيئة:

::

  I0607 16:10:35.739390 515217 logger.cpp:173] [Rank 0]: DDP Initialized with:
  broadcast_buffers: 1
  bucket_cap_bytes: 26214400
  find_unused_parameters: 0
  gradient_as_bucket_view: 0
  is_multi_device_module: 0
  iteration: 0
  num_parameter_tensors: 2
  output_device: 0
  rank: 0
  total_parameter_size_bytes: 440
  world_size: 2
  backend_name: nccl
  bucket_sizes: 440
  cuda_visible_devices: N/A
  device_ids: 0
  dtypes: float
  master_addr: localhost
  master_port: 29501
  module_name: TwoLinLayerNet
  nccl_async_error_handling: N/A
  nccl_blocking_wait: N/A
  nccl_debug: WARN
  nccl_ib_timeout: N/A
  nccl_nthreads: N/A
  nccl_socket_ifname: N/A
  torch_distributed_debug: INFO


يتم تقديم سجلات التصحيح التالية أثناء وقت التشغيل (عند تعيين ``TORCH_DISTRIBUTED_DEBUG=DETAIL``):

::

  I0607 16:18:58.085681 544067 logger.cpp:344] [Rank 1 / 2] Training TwoLinLayerNet unused_parameter_size=0
   Avg forward compute time: 40838608
   Avg backward compute time: 5983335
  Avg backward comm. time: 4326421
   Avg backward comm/comp overlap time: 4207652
  I0607 16:18:58.085693 544066 logger.cpp:344] [Rank 0 / 2] Training TwoLinLayerNet unused_parameter_size=0
   Avg forward compute time: 42850427
   Avg backward compute time: 3885553
  Avg backward comm. time: 2357981
   Avg backward comm/comp overlap time: 2234674


بالإضافة إلى ذلك، يعزز ``TORCH_DISTRIBUTED_DEBUG=INFO`` تسجيل الأخطاء عند حدوث تعطل في ``torch.nn.parallel.DistributedDataParallel`` بسبب وجود معلمات غير مستخدمة في النموذج. حاليًا، يجب تمرير ``find_unused_parameters=True`` إلى تهيئة ``torch.nn.parallel.DistributedDataParallel`` إذا كانت هناك معلمات قد لا تستخدم في عملية الحساب للأمام، واعتبارًا من الإصدار 1.10، يجب استخدام جميع نتائج النموذج في حساب الخسارة حيث أن ``torch.nn.parallel.DistributedDataParallel`` لا تدعم المعلمات غير المستخدمة في عملية الحساب للخلف. هذه القيود صعبة خاصة بالنسبة للنماذج الأكبر، لذا عند حدوث خطأ، يقوم ``torch.nn.parallel.DistributedDataParallel`` بتسجيل اسم المؤهل الكامل لجميع المعلمات التي لم تستخدم. على سبيل المثال، في التطبيق أعلاه، إذا قمنا بتعديل حساب الخسارة ليكون ``loss = output[1]``، فإن ``TwoLinLayerNet.a`` لا تتلقى تدرجًا في عملية الحساب للخلف، مما يؤدي إلى فشل ``DDP``. في حالة حدوث خطأ، يتم تمرير معلومات إلى المستخدم حول المعلمات التي لم تستخدم، والتي قد يكون من الصعب العثور عليها يدويًا في النماذج الكبيرة:


::

  RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing
   the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
  making sure all `forward` function outputs participate in calculating loss.
  If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return va
  lue of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
  Parameters which did not receive grad for rank 0: a.weight
  Parameter indices which did not receive grad for rank 0: 0


سيؤدي تعيين ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` إلى تشغيل فحوصات الاتساق والمزامنة الإضافية لكل مكالمة جماعية يصدرها المستخدم، سواء بشكل مباشر أو غير مباشر (مثل DDP ``allreduce``). يتم ذلك من خلال إنشاء مجموعة عمليات مغلفة تقوم بلف جميع مجموعات العمليات التي تم إرجاعها بواسطة واجهات برمجة التطبيقات ``torch.distributed.init_process_group`` و``torch.distributed.new_group``. ونتيجة لذلك، ستعيد واجهات برمجة التطبيقات هذه مجموعة عمليات مغلفة يمكن استخدامها تمامًا مثل مجموعة عمليات عادية، ولكنها تقوم بفحوصات الاتساق قبل إرسال الجماعي إلى مجموعة عمليات أساسية. تشمل هذه الفحوصات حاليًا حاجزًا مراقبًا ``torch.distributed.monitored_barrier``، والذي يضمن إكمال جميع الرتب لمكالماتها الجماعية المعلقة والإبلاغ عن الرتب التي تعلق. بعد ذلك، يتم فحص الجماعي نفسه للاتساق من خلال التأكد من تطابق جميع وظائف الجماعي واستدعائها بأشكال متسقة من التنسيقات. إذا لم يكن الأمر كذلك، يتم تضمين تقرير خطأ مفصل عند تعطل التطبيق، بدلاً من التعليق أو رسالة خطأ غير مفيدة. على سبيل المثال، ضع في اعتبارك الوظيفة التالية التي تحتوي على أشكال إدخال غير متطابقة في ``torch.distributed.all_reduce``:

::

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


    def worker(rank):
        dist.init_process_group("nccl", rank=rank, world_size=2)
        torch.cuda.set_device(rank)
        tensor = torch.randn(10 if rank == 0 else 20).cuda()
        dist.all_reduce(tensor)
        torch.cuda.synchronize(device=rank)


    if __name__ == "__main__":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        os seedu["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        mp.spawn(worker, nprocs=2, args=())

مع دعم خلفية "إن سي سي إل" (NCCL)، من المحتمل أن يؤدي مثل هذا التطبيق إلى تعليق قد يكون من الصعب تحديد سببه في السيناريوهات غير البديهية. إذا قام المستخدم بتمكين ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` وأعاد تشغيل التطبيق، فإن رسالة الخطأ التالية تكشف عن السبب الجذري:

::

    work = default_pg.allreduce([tensor], opts)
    RuntimeError: Error when verifying shape tensors for collective ALLREDUCE on rank 0. This likely indicates that input shapes into the collective are mismatched across ranks. Got shapes:  10
    [ torch.LongTensor{1} ]

.. note::
    للحصول على تحكم دقيق في مستوى التصحيح أثناء وقت التشغيل، يمكن أيضًا استخدام وظائف ``torch.distributed.set_debug_level`` و``torch.distributed.set_debug_level_from_env`` و``torch.distributed.get_debug_level``.

بالإضافة إلى ذلك، يمكن استخدام ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` بالتزامن مع ``TORCH_SHOW_CPP_STACKTRACES=1`` لتسجيل المكدس الكامل عند اكتشاف عدم تزامن جماعي. ستعمل فحوصات عدم التزامن الجماعي هذه لجميع التطبيقات التي تستخدم مكالمات جماعية "سي10دي" (c10d) المدعومة بواسطة مجموعات العمليات التي تم إنشاؤها باستخدام واجهات برمجة التطبيقات ``torch.distributed.init_process_group`` و``torch.distributed.new_group``.

تسجيل
بالإضافة إلى الدعم الصريح للتصحيح من خلال ``torch.distributed.monitored_barrier`` و ``TORCH_DISTRIBUTED_DEBUG``، فإن مكتبة C++ الأساسية لـ ``torch.distributed`` تقوم أيضًا بإخراج رسائل السجل بمستويات مختلفة. يمكن أن تكون هذه الرسائل مفيدة لفهم حالة التنفيذ لمهمة تدريب موزعة ولاستكشاف أخطاء مشكلات مثل أخطاء اتصال الشبكة.

+-------------------------+-----------------------------+------------------------+
| ``TORCH_CPP_LOG_LEVEL`` | ``TORCH_DISTRIBUTED_DEBUG`` |   مستوى السجل الفعّال  |
+=========================+=============================+========================+
| ``ERROR``               | يتم تجاهله                     | خطأ                  |
+-------------------------+-----------------------------+------------------------+
| ``WARNING``             | يتم تجاهله                     | تحذير                |
+-------------------------+-----------------------------+------------------------+
| ``INFO``                | يتم تجاهله                     | معلومات               |
+-------------------------+-----------------------------+------------------------+
| ``INFO``                | ``INFO``                    | تصحيح                  |
+-------------------------+-----------------------------+------------------------+
| ``INFO``                | ``DETAIL``                  | تتبع (المعروف أيضًا باسم الكل)     |
+-------------------------+-----------------------------+------------------------+

ترفع المكونات الموزعة أنواع استثناء مخصصة مشتقة من `RuntimeError`:

- `torch.distributed.DistError`: هذا هو النوع الأساسي لجميع الاستثناءات الموزعة.
- `torch.distributed.DistBackendError`: يتم إلقاء هذا الاستثناء عندما يحدث خطأ محدد للخلفية. على سبيل المثال، إذا
  تم استخدام خلفية `NCCL` ويحاول المستخدم استخدام وحدة معالجة رسومية (GPU) غير متوفرة لمكتبة `NCCL`.
- `torch.distributed.DistNetworkError`: يتم إلقاء هذا الاستثناء عندما تواجه مكتبات الشبكات
  أخطاء (مثال: إعادة تعيين الاتصال بواسطة النظراء)
- `torch.distributed.DistStoreError`: يتم إلقاء هذا الاستثناء عندما يواجه المتجر
  خطأ (مثال: انتهاء مهلة TCPStore)

.. autoclass:: torch.distributed.DistError
.. autoclass:: torch.distributed.DistBackendError
.. autoclass:: torch.distributed.DistNetworkError
.. autoclass:: torch.distributed.DistStoreError

إذا كنت تقوم بالتدريب على عقدة واحدة، فقد يكون من الملائم وضع نقطة توقف في البرنامج النصي الخاص بك. نوفر طريقة ملائمة لوضع نقطة توقف في رتبة واحدة:

.. autofunction:: torch.distributed.breakpoint

.. الوحدات الموزعة الناقصة لإدخالات محددة.
.. نقوم بإضافتها هنا لأغراض التتبع حتى يتم إصلاحها بشكل دائم.
.. py:module:: torch.distributed.algorithms
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks
.. py:module:: torch.distributed.algorithms.model_averaging
.. py:module:: torch.distributed.elastic
.. py:module:: torch.distributed.elastic.utils
.. py:module:: torch.distributed.elastic.utils.data
.. py:module:: torch.distributed.launcher
.. py:module:: torch.distributed.nn
.. py:module:: torch.distributed.nn.api
.. py:module:: torch.distributed.nn.jit
.. py:module:: torch.distributed.nn.jit.templates
.. py:module:: torch.distributed.tensor
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.default_hooks
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.mixed_precision_hooks
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks
.. py:module:: torch.distributed.algorithms.join
.. py:module:: torch.distributed.algorithms.model_averaging.averagers
.. py:module:: torch.distributed.algorithms.model_averaging.hierarchical_model_averager
.. py:module:: torch.distributed.algorithms.model_averaging.utils
.. py:module:: torch.distributed.argparse_util
.. py:module:: torch.distributed.c10d_logger
.. py:module:: torch.distributed.checkpoint.api
.. py:module:: torch.distributed.checkpoint.default_planner
.. py:module:: torch.distributed.checkpoint.filesystem
.. py:module:: torch.distributed.checkpoint.metadata
.. py:module:: torch.distributed.checkpoint.optimizer
.. py:module:: torch.distributed.checkpoint.planner
.. py:module:: torch.distributed.checkpoint.planner_helpers
.. py:module:: torch.distributed.checkpoint.resharding
.. py:module:: torch.distributed.checkpoint.state_dict_loader
.. py:module:: torch.distributed.checkpoint.state_dict_saver
.. py:module:: torch.distributed.checkpoint.stateful
.. py:module:: torch.distributed.checkpoint.storage
.. py:module:: torch.distributed.checkpoint.utils
.. py:module:: torch.distributed.collective_utils
.. py:module:: torch.distributed.constants
.. py:module:: torch.distributed.device_mesh
.. py:module:: torch.distributed.distributed_c10d
.. py:module:: torch.distributed.elastic.agent.server.api
.. py:module:: torch.distributed.elastic.agent.server.local_elastic_agent
.. py:module:: torch.distributed.elastic.events.api
.. py:module:: torch.distributed.elastic.events.handlers
.. py:module:: torch.distributed.elastic.metrics.api
.. py:module:: torch.distributed.elastic.multiprocessing.api
.. py:module:: torch.distributed.elastic.multiprocessing.errors.error_handler
.. py:module:: torch.distributed.elastic.multiprocessing.errors.handlers
.. py:module:: torch.distributed.elastic.multiprocessing.redirects
.. py:module:: torch.distributed.elastic.multiprocessing.tail_log
.. py:module:: torch.distributed.elastic.rendezvous.api
.. py:module:: torch.distributed.elastic.rendezvous.c10d_rendezvous_backend
.. py:module:: torch.distributed.elastic.rendezvous.dynamic_rendezvous
.. py:module:: torch.distributed.elastic.rendezvous.etcd_rendezvous
.. py:module:: torch.distributed.elastic.rendezvous.etcd_rendezvous_backend
.. py:module:: torch.distributed.elastic.rendezvous.etcd_server
.. py:module:: torch.distributed.elastic.rendezvous.etcd_store
.. py:module:: torch.distributed.elastic.rendezvous.static_tcp_rendezvous
.. py:module:: torch.distributed.elastic.rendezvous.utils
.. py:module:: torch.distributed.elastic.timer.api
.. py:module:: torch.distributed.elastic.timer.file_based_local_timer
.. py:module:: torch.distributed.elastic.timer.local_timer
.. py:module:: torch.distributed.elastic.utils.api
.. py:module:: torch.distributed.elastic.utils.data.cycling_iterator
.. py:module:: torch.distributed.elastic.utils.data.elastic_distributed_sampler
.. py:module:: torch.distributed.elastic.utils.distributed
.. py:module:: torch.distributed.elastic.utils.log_level
.. py:module:: torch.distributed.elastic.utils.logging
.. py:module:: torch.distributed.elastic.utils.store
.. py:module:: torch.distributed.fsdp.api
.. py:module:: torch.distributed.fsdp.fully_sharded_data_parallel
.. py:module:: torch.distributed.fsdp.sharded_grad_scaler
.. py:module:: torch.distributed.fsdp.wrap
.. py:module:: torch.distributed.launcher.api
.. py:module:: torch.distributed.logging_handlers
.. py:module:: torch.distributed.nn.api.remote_module
.. py:module:: torch.distributed.nn.functional
.. py:module:: torch.distributed.nn.jit.instantiator
.. py:module:: torch.distributed.nn.jit.templates.remote_module_template
.. py:module:: torch.distributed.optim.apply_optimizer_in_backward
.. py:module:: torch.distributed.optim.functional_adadelta
.. py:module:: torch.distributed.optim.functional_adagrad
.. py:module:: torch.distributed.optim.functional_adam
.. py:module:: torch.distributed.optim.functional_adamax
.. py:module:: torch.distributed.optim.functional_adamw
.. py:module:: torch.distributed.optim.functional_rmsprop
.. py:module:: torch.distributed.optim.functional_rprop
.. py:module:: torch.distributed.optim.functional_sgd
.. py:module:: torch.distributed.optim.named_optimizer
.. py:module:: torch.distributed.optim.optimizer
.. py:module:: torch.distributed.optim.post_localSGD_optimizer
.. py:module:: torch.distributed.optim.utils
.. py:module:: torch.distributed.optim.zero_redundancy_optimizer
.. py:module:: torch.distributed.remote_device
.. py:module:: torch.distributed.rendezvous
.. py:module:: torch.distributed.rpc.api
.. py:module:: torch.distributed.rpc.backend_registry
.. py:module:: torch.distributed.rpc.constants
.. py:module:: torch.distributed.rpc.functions
.. py:module:: torch.distributed.rpc.internal
.. py:module:: torch.distributed.rpc.options
.. py:module:: torch.distributed.rpc.rref_proxy
.. py:module:: torch.distributed.rpc.server_process_global_profiler
.. py:module:: torch.distributed.tensor.parallel.api
.. py:module:: torch.distributed.tensor.parallel.ddp
.. py:module:: torch.distributed.tensor.parallel.fsdp
.. py:module:: torch.distributed.tensor.parallel.input_reshard
.. py:module:: torch.distributed.tensor.parallel.loss
.. py:module:: torch.distributed.tensor.parallel.style
.. py:module:: torch.distributed.utils
.. py:module:: torch.distributed.checkpoint.state_dict