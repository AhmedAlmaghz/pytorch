.. automodule:: torch.package
.. py:module:: torch.package.analyze

.. currentmodule:: torch.package.analyze

حزمة الشعلة
أضاف ``torch.package`` الدعم لإنشاء حزم تحتوي على كل من الآثار ورمز PyTorch التعسفي. يمكن حفظ هذه الحزم ومشاركتها واستخدامها لتحميل نماذج وتشغيلها في تاريخ لاحق أو على جهاز مختلف، بل ويمكن نشرها في الإنتاج باستخدام ``torch::deploy``.

تحتوي هذه الوثيقة على دروس تعليمية وأدلة إرشادية وتوضيحات ومرجع API سيساعدك على معرفة المزيد حول ``torch.package`` وكيفية استخدامه.

.. تحذير ::

تعتمد هذه الوحدة على وحدة "pickle" غير الآمنة. لا تفك حزم البيانات إلا إذا كنت تثق بها.

من الممكن إنشاء بيانات "pickle" ضارة والتي ستنفذ **رمزًا تعسفيًا أثناء فك التقطيع**. لا تقم مطلقًا بفك حزم البيانات التي قد تكون جاءت من مصدر غير موثوق أو قد تم العبث بها.

لمزيد من المعلومات، راجع وثائق وحدة "pickle" <https://docs.python.org/3/library/pickle.html>.

.. محتويات:: :local:
:depth: 2

الدروس التعليمية
---------
تعبئة نموذجك الأول
^^^^^^^^^^^^^^^^^^^^^^^^^^
يتوفر درس يوجهك خلال تعبئة وفك تعبئة نموذج بسيط `على Colab <https://colab.research.google.com/drive/1lFZkLyViGfXxB-m3jqlyTQuYToo3XLo->`_.
بعد الانتهاء من هذا التمرين، ستكون على دراية بواجهة برمجة التطبيقات الأساسية لإنشاء حزم Torch واستخدامها.

كيف يمكنني...
-----------
مشاهدة ما بداخل حزمة؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
تعامل مع الحزمة مثل أرشيف ZIP
""""""""""""""""""""""""""""""""""""
تنسيق الحاوية لحزمة ``torch.package`` هو ZIP، لذلك يجب أن تعمل أي أدوات تعمل مع ملفات ZIP القياسية
يجب أن يعمل لاستكشاف المحتويات. بعض الطرق الشائعة للتفاعل مع ملفات ZIP:

* ``unzip my_package.pt`` سيقوم بفك ضغط أرشيف ``torch.package`` إلى القرص، حيث يمكنك فحص محتوياته بحرية.


::

    $ unzip my_package.pt && tree my_package
    my_package
    ├── .data
    │   ├── 94304870911616.storage
    │   ├── 94304900784016.storage
    │   ├── extern_modules
    │   └── version
    ├── models
    │   └── model_1.pkl
    └── torchvision
        └── models
            ├── resnet.py
            └── utils.py
    ~ cd my_package && cat torchvision/models/resnet.py
    ...


* توفر وحدة ``zipfile`` في Python طريقة قياسية لقراءة محتويات أرشيف ZIP وكتابتها.


::

    from zipfile import ZipFile
    with ZipFile("my_package.pt") as myzip:
        file_bytes = myzip.read("torchvision/models/resnet.py")
        # edit file_bytes in some way
        myzip.writestr("torchvision/models/resnet.py", new_file_bytes)


* يمكن لـ vim قراءة أرشيفات ZIP بشكل أصلي. يمكنك حتى تحرير الملفات وإعادتها إلى الأرشيف!


::

    # أضف هذا إلى ملفك `.vimrc` لتعامل مع ملفات `*.pt` كملفات zip
    au BufReadCmd *.pt call zip#Browse(expand("<amatch>"))

    ~ vi my_package.pt


استخدم طريقة ``file_structure()`` API
""""""""""""""""""""""""""""""""
توفر :class:`PackageImporter` طريقة ``file_structure()``، والتي ستعيد كائنًا قابلًا للطباعة
وكائن :class:`Directory` يمكن الاستعلام عنه. كائن :class:`Directory` هو هيكل دليل بسيط يمكنك استخدامه لاستكشاف
محتويات حزمة ``torch.package`` الحالية.

يمكن طباعة كائن :class:`Directory` نفسه، وسيقوم بطباعة تمثيل شجرة الملفات. لتصفية ما يتم إرجاعه،
استخدم حجج التصفية "include" و"exclude" بأسلوب glob.


::

    with PackageExporter('my_package.pt') as pe:
        pe.save_pickle('models', 'model_1.pkl', mod)

    importer = PackageImporter('my_package.pt')
    # يمكن الحد من العناصر المطبوعة باستخدام حجج include/exclude
    print(importer.file_structure(include=["**/utils.py", "**/*.pkl"], exclude="**/*.storage"))
    print(importer.file_structure()) # ستتم طباعة جميع الملفات


الناتج:


::

    # تم التصفية باستخدام نمط glob:
    # تشمل ["**/utils.py"، "**/*.pkl"]، واستبعاد "**/*.storage"
    ─── my_package.pt
        ├── models
        │   └── model_1.pkl
        └── torchvision
            └── models
                └── utils.py

    # جميع الملفات
    ─── my_package.pt
        ├── .data
        │   ├── 94304870911616.storage
        │   ├── 94304900784016.storage
        │   ├── extern_modules
        │   └── version
        ├── models
        │   └── model_1.pkl
        └── torchvision
            └── models
                ├── resnet.py
                └── utils.py


يمكنك أيضًا استعلام كائنات :class:`Directory` باستخدام طريقة ``has_file()``.


::

    importer_file_structure = importer.file_structure()
    found: bool = importer_file_structure.has_file("package_a/subpackage.py")

لماذا تم تضمين وحدة نمطية معينة كاعتماد؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

لنفترض أن هناك وحدة نمطية معينة تسمى ``foo``، وتريد أن تعرف سبب قيام :class:`PackageExporter` بسحب ``foo`` كاعتماد.

ستقوم طريقة :meth:`PackageExporter.get_rdeps` بإرجاع جميع الوحدات النمطية التي تعتمد مباشرة على ``foo``.

إذا كنت تريد أن ترى كيف تعتمد وحدة نمطية معينة ``src`` على ``foo``، فإن طريقة :meth:`PackageExporter.all_paths`
ستعيد رسمًا بيانيًا بتنسيق DOT يُظهر جميع مسارات الاعتماد بين ``src`` و ``foo``.

إذا كنت تريد فقط أن ترى الرسم البياني الكامل للاعتماد لحزمة :class:`PackageExporter`، فيمكنك استخدام طريقة :meth:`PackageExporter.dependency_graph_string`.


تضمين موارد عشوائية مع الحزمة الخاصة بي والوصول إليها لاحقًا؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
تعرض :class:`PackageExporter` ثلاث طرق، ``save_pickle``، و``save_text``، و``save_binary`` تسمح لك بحفظ
الكائنات النصية، وبيانات Python، والبيانات الثنائية في الحزمة.


::

    with torch.PackageExporter("package.pt") as exporter:
        # يحفظ الكائن كملف Pickled ويحفظه في `my_resources/tensor.pkl` في الأرشيف.
        exporter.save_pickle("my_resources", "tensor.pkl", torch.randn(4))
        exporter.save_text("config_stuff", "words.txt", "a sample string")
        exporter.save_binary("raw_data", "binary", my_bytes)


تعرض :class:`PackageImporter` طرقًا تكميلية تسمى ``load_pickle``، و``load_text``، و``load_binary`` تسمح لك بتحميل
كائنات Python، والنصوص، والبيانات الثنائية من الحزمة.


::

    importer = torch.PackageImporter("package.pt")
    my_tensor = importer.load_pickle("my_resources", "tensor.pkl")
    text = importer.load_text("config_stuff", "words.txt")
    binary = importer.load_binary("raw_data", "binary")


تخصيص كيفية تغليف فئة؟
^^^^^^^^^^^^^^^^^^^^
يسمح ``torch.package`` بتخصيص كيفية تغليف الفئات. يتم الوصول إلى هذا السلوك من خلال تعريف طريقة
``__reduce_package__`` في الفئة ومن خلال تعريف دالة فك التغليف المقابلة. هذا مشابه لتعريف ``__reduce__`` لـ
عملية التخليل القياسية في Python.

الخطوات:

1. قم بتعريف طريقة ``__reduce_package__(self, exporter: PackageExporter)`` في الفئة المستهدفة. يجب أن تقوم هذه الطريقة بتنفيذ العمل لحفظ مثيل الفئة داخل الحزمة، ويجب أن تعيد زوجًا من دالة فك التغليف مع وسائط استدعاء دالة فك التغليف. يتم استدعاء هذه الطريقة بواسطة ``PackageExporter`` عند مصادفة مثيل للفئة المستهدفة.
2. قم بتعريف دالة فك التغليف للفئة. يجب أن تقوم دالة فك التغليف هذه بتنفيذ العمل اللازم لإعادة بناء مثيل الفئة وإعادته. يجب أن يكون أول وسيط في توقيع الدالة عبارة عن مثيل لـ ``PackageImporter``، ويجب أن تكون بقية الوسائط معرفة من قبل المستخدم.


::

    # foo.py [مثال على تخصيص كيفية تغليف فئة Foo]
    from torch.package import PackageExporter, PackageImporter
    import time


    class Foo:
        def __init__(self, my_string: str):
            super().__init__()
            self.my_string = my_string
            self.time_imported = 0
            self.time_exported = 0

        def __reduce_package__(self, exporter: PackageExporter):
            """
            يتم استدعاء هذه الطريقة بواسطة ``torch.package.PackageExporter``'s Pickler's ``persistent_id`` عند
            حفظ مثيل من هذا الكائن. يجب أن تقوم هذه الطريقة بتنفيذ العمل لحفظ هذا
            الكائن داخل أرشيف ``torch.package``.

            تعيد الدالة مع وسائط لإعادة تحميل الكائن من
            ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function.
            """

            # استخدم هذا النمط لضمان عدم وجود تعارضات في التسمية مع التبعيات العادية،
            # يجب ألا تتعارض أي عناصر محفوظة ضمن اسم الوحدة النمطية هذا مع العناصر الأخرى
            # في الحزمة
            generated_module_name = f"foo-generated._{exporter.get_unique_id()}"
            exporter.save_text(
                generated_module_partumodule_name,
                "foo.txt",
                self.my_string + ", with exporter modification!",
            )
            time_exported = time.clock_gettime(1)

            # تعيد دالة فك التغليف مع وسائط لاستدعائها
            return (unpackage_foo, (generated_module_name, time_exported,))


    def unpackage_foo(
        importer: PackageImporter, generated_module_name: str, time_exported: float
    ) -> Foo:
        """
        يتم استدعاء هذه الدالة بواسطة ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function
        عند فك تغليف كائن Foo.
        تقوم بتنفيذ العمل اللازم لتحميل وإعادة مثيل من فئة Foo من أرشيف ``torch.package``.
        """
        time_imported = time.clock_gettime(1)
        foo = Foo(importer.load_text(generated_module_name, "foo.txt"))
        foo.time_imported = time_imported
        foo.time_exported = time_exported
        return foo


::

    # مثال على حفظ مثيلات من فئة Foo

    import torch
    from torch.package import PackageImporter, PackageExporter
    import foo

    foo_1 = foo.Foo("foo_1 initial string")
    foo_2 = foo.Foo("foo_2 initial string")
    with PackageExporter('foo_package.pt') as pe:
        # احفظه بشكل طبيعي، لا يلزم عمل إضافي
        pe.save_pickle('foo_collection', 'foo1.pkl', foo_1)
        pe.save_pickle('foo_collection', 'foo2.pkl', foo_2)

    pi = PackageImporter('foo_package.pt')
    print(pi.file_structure())
    imported_foo = pi.load_pickle('foo_collection', 'foo1.pkl')
    print(f"foo_1 string: '{imported_foo.my_string}'")
    print(f"foo_1 export time: {imported_foo.time_exported}")
    print(f"foo_1 import time: {imported_foo.time_imported}")


::

    # إخراج تشغيل البرنامج النصي أعلاه
    ─── foo_package
        ├── foo-generated
        │   ├── _0
        │   │   └── foo.txt
        │   └── _1
        │       └── foo.txt
        ├── foo_collection
        │   ├── foo1.pkl
        │   └── foo2.pkl
        └── foo.py

    foo_1 string: 'foo_1 initial string, with reduction modification!'
    foo_1 export time: 9857706.650140837
    foo_1 import time: 9857706.652698385


اختبار في شفرة المصدر الخاصة بي ما إذا كان يتم التنفيذ داخل حزمة أم لا؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
سيضيف :class:`PackageImporter` السمة ``__torch_package__`` إلى كل وحدة نمطية يقوم بتهيئتها. يمكن لشفرة التحقق من وجود
هذه السمة لتحديد ما إذا كان يتم تنفيذها في سياق الحزمة أم لا.


::

    # في foo/bar.py:

    if "__torch_package__" in dir():  # true إذا كان يتم تحميل الكود من حزمة
        def is_in_package():
            return True

        UserException = Exception
    else:
        def is_in_package():
            return False

        UserException = UnpackageableException


الآن، سيتصرف الكود بشكل مختلف اعتمادًا على ما إذا كان يتم استيراده بشكل طبيعي من خلال بيئة Python الخاصة بك أو استيراده من
``torch.package``.


::

    from foo.bar import is_in_package

    print(is_in_package())  # False

    loaded_module = PackageImporter(my_package).import_module("foo.bar")
    loaded_module.is_in_package()  # True


**تحذير**: بشكل عام، من السيئ أن يكون لديك كود يتصرف بشكل مختلف اعتمادًا على ما إذا كان معبأً أم لا. يمكن أن يؤدي ذلك إلى
مشكلات يصعب تصحيحها والتي تتأثر بكيفية استيرادك لشفرة المصدر الخاصة بك. إذا كانت حزمتك مصممة للاستخدام المكثف، ففكر في إعادة هيكلة
شفرة المصدر الخاصة بك بحيث تتصرف بنفس الطريقة بغض النظر عن كيفية تحميلها.


تصحيح رمز في حزمة؟
^^^^^^^^^^^^^^^
تقدم :class:`PackageExporter` طريقة ``save_source_string()`` تسمح لك بحفظ شفرة مصدر Python عشوائية إلى وحدة نمطية من اختيارك.


::

    with PackageExporter(f) as exporter:
        # احفظ my_module.foo المتاحة في بيئة Python الحالية الخاصة بك.
        exporter.save_module("my_module.foo")

        # يحفظ هذا السلسلة المقدمة إلى my_module/foo.py في أرشيف الحزمة.
        # سوف يلغي ما تم حفظه مسبقًا في my_module.foo
        exporter.save_source_string("my_module.foo", textwrap.dedent(
            """\
            def my_function():
                print('hello world')
            """
        ))

        # إذا كنت تريد التعامل مع my_module.bar كحزمة
        # (على سبيل المثال، قم بالتخزين في `my_module/bar/__init__.py` بدلاً من `my_module/bar.py)
        # قم بتمرير is_package=True،
        exporter.save_source_string("my_module.bar",
                                    "def foo(): print('hello')\n",
                                    is_package=True)

    importer = PackageImporter(f)
    importer.import_module("my_module.foo").my_function()  # يطبع 'مرحبًا بالعالم'


الوصول إلى محتويات الحزمة من الكود المعلب؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ينفذ :class:`PackageImporter` واجهة برمجة التطبيقات
`importlib.resources <https://docs.python.org/3/library/importlib.html#module-importlib.resources>`_
للوصول إلى الموارد من داخل حزمة.
::

    with PackageExporter(f) as exporter:
        # يحفظ النص في my_resource/a.txt داخل الأرشيف
        exporter.save_text("my_resource", "a.txt", "hello world!")
        # يحفظ التنسور في my_pickle/obj.pkl
        exporter.save_pickle("my_pickle", "obj.pkl", torch.ones(2, 2))

        # شاهد أدناه لمحتويات الوحدة النمطية
        exporter.save_module("foo")
        exporter.save_module("bar")


يتيح واجهة برمجة التطبيقات (API) ``importlib.resources`` الوصول إلى الموارد من داخل التعليمات البرمجية المعلبة.


::

    # foo.py:
    import importlib.resources
    import my_resource

    # يعيد "hello world!"
    def get_my_resource():
        return importlib.resources.read_text(my_resource, "a.txt")


يعد استخدام ``importlib.resources`` الطريقة الموصى بها للوصول إلى محتويات الحزمة من داخل التعليمات البرمجية المعلبة، حيث يتوافق
مع معيار بايثون. ومع ذلك، من الممكن أيضًا الوصول إلى مثيل :class:`PackageImporter` الأساسي نفسه من داخل التعليمات البرمجية المعلبة.


::

    # bar.py:
    import torch_package_importer # هذا هو PackageImporter الذي قام باستيراد هذه الوحدة النمطية.

    # يطبع "hello world!"، وهو ما يعادل importlib.resources.read_text
    def get_my_resource():
        return torch_package_importer.load_text("my_resource", "a.txt")

    # يمكنك أيضًا القيام بأشياء لا تدعمها واجهة برمجة تطبيقات importlib.resources، مثل تحميل
    # كائن مخلل من الحزمة.
    def get_my_pickle():
        return torch_package_importer.load_pickle("my_pickle", "obj.pkl")


التمييز بين التعليمات البرمجية المعلبة وغير المعلبة؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
للتحقق مما إذا كانت التعليمات البرمجية لكائن ما من ``torch.package``، استخدم دالة ``torch.package.is_from_package()``.
ملاحظة: إذا كان الكائن من حزمة ولكن تعريفه من وحدة نمطية موسومة بـ ``extern`` أو من ``stdlib``،
فستعيد هذه الدالة ``False``.


::

    importer = PackageImporter(f)
    mod = importer.import_module('foo')
    obj = importer.load_pickle('model', 'model.pkl')
    txt = importer.load_text('text', 'my_test.txt')

    assert is_from_package(mod)
    assert is_from_package(obj)
    assert not is_from_package(txt) # str هو من stdlib، لذلك ستعيد هذه الدالة False


إعادة تصدير كائن مستورد؟
^^^^^^^^^^^^^^^^^^^
لإعادة تصدير كائن تم استيراده مسبقًا بواسطة :class:`PackageImporter`، يجب جعل :class:`PackageExporter` الجديد
على علم بـ :class:`PackageImporter` الأصلي حتى يتمكن من العثور على شفرة المصدر لتبعيات الكائن الخاص بك.


::

    importer = PackageImporter(f)
    obj = importer.load_pickle("model", "model.pkl")

    # إعادة تصدير الكائن في حزمة جديدة
    with PackageExporter(f2, importer=(importer, sys_importer)) as exporter:
        exporter.save_pickle("model", "model.pkl", obj)


تعبئة وحدة نمطية TorchScript؟
^^^^^^^^^^^^^^^^^^^^^^^^^
لتعبئة نموذج TorchScript، استخدم نفس واجهات برمجة التطبيقات (APIs) ``save_pickle`` و ``load_pickle`` كما تفعل مع أي كائن آخر.
كما يتم دعم حفظ كائنات TorchScript التي تكون سمات أو وحدات نمطية فرعية دون أي جهد إضافي.


::

    # حفظ TorchScript مثل أي كائن آخر
    with PackageExporter(file_name) as e:
        e.save_pickle("res", "script_model.pkl", scripted_model)
        e.save_pickle("res", "mixed_model.pkl", python_model_with_scripted_submodule)
    # تحميل كالمعتاد
    importer = PackageImporter(file_name)
    loaded_script = importer.load_pickle("res", "script_model.pkl")
    loaded_mixed = importer.load_pickle("res", "mixed_model.pkl"


شرح
``torch.package`` نظرة عامة على التنسيق
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ملف ``torch.package`` هو أرشيف ZIP الذي يستخدم بشكل تقليدي امتداد ``.pt``. داخل أرشيف ZIP، هناك نوعان من الملفات:

* ملفات الإطار، والتي يتم وضعها في ``.data/``.
* ملفات المستخدم، والتي هي كل شيء آخر.

على سبيل المثال، هذا هو الشكل الذي يبدو عليه نموذج ResNet مُعبأ بالكامل من ``torchvision``:

::

    resnet
    ├── .data  # يتم تخزين جميع البيانات الخاصة بالإطار هنا.
    │   │      # تمت تسميته لتجنب التضارب مع التعليمات البرمجية المُسلسلة من قبل المستخدم.
    │   ├── 94286146172688.storage  # بيانات tensor
    │   ├── 94286146172784.storage
    │   ├── extern_modules  # ملف نصي بأسماء الوحدات النمطية الخارجية (مثل 'torch')
    │   ├── version         # بيانات وصفية للإصدار
    │   ├── ...
    ├── model  # النموذج المُخلل
    │   └── model.pkl
    └── torchvision  # يتم التقاط جميع تبعيات التعليمات البرمجية كملفات مصدر
        └── models
            ├── resnet.py
            └── utils.py

ملفات الإطار
"""""""""""""""
تعتبر ملفات الدليل ``.data/`` مملوكة لـ torch.package، ويتم اعتبار محتوياتها تفاصيل تنفيذ خاصة.
لا يضمن تنسيق ``torch.package`` أي شيء حول محتويات ``.data/``، ولكن أي تغييرات يتم إجراؤها ستكون متوافقة مع الإصدارات السابقة
(بمعنى أن الإصدارات الأحدث من PyTorch ستتمكن دائمًا من تحميل حزم ``torch.packages`` القديمة).

حاليًا، يحتوي دليل ``.data/`` على العناصر التالية:

* ``version``: رقم إصدار لتنسيق المُسلسل، بحيث يعرف البنية الأساسية لاستيراد ``torch.package`` كيفية تحميل هذه الحزمة.
* ``extern_modules``: قائمة بالوحدات النمطية التي تعتبر "خارجية". سيتم استيراد الوحدات النمطية "الخارجية" باستخدام مُحمل النظام البيئي للتحميل.
* ``*.storage``: بيانات tensor المُسلسلة.

::

    .data
    ├── 94286146172688.storage
    ├── 94286146172784.storage
    ├── extern_modules
    ├── version
    ├── ...

ملفات المستخدم
""""""""""""
جميع الملفات الأخرى في الأرشيف قام المستخدم بوضعها هناك. التخطيط مطابق تمامًا لحزمة Python
`العادية <https://docs.python.org/3/reference/import.html#regular-packages>`_. للحصول على نظرة أكثر تعمقًا في كيفية عمل التعبئة في Python،
يرجى الرجوع إلى `هذه المقالة <https://www.python.org/doc/essays/packages/>`_ (إنها قديمة بعض الشيء، لذا تحقق من تفاصيل التنفيذ
مع `وثائق مرجع Python <https://docs.python.org/3/library/importlib.html>`_).

::

    <package root>
    ├── model  # النموذج المخلل
    │   └── model.pkl
    ├── another_package
    │   ├── __init__.py
    │   ├── foo.txt         # ملف مورد، راجع importlib.resources
    │   └── ...
    └── torchvision
        └── models
            ├── resnet.py   # torchvision.models.resnet
            └── utils.py    # torchvision.models.utils

كيف يجد ``torch.package`` تبعيات كودك
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
تحليل تبعيات كائن
""""""""""""
عندما تصدر أمر ``save_pickle(obj, ...)``، سيقوم :class:`PackageExporter` بتخليل الكائن بشكل طبيعي. بعد ذلك، يستخدم وحدة ``pickletools`` النمطية في المكتبة القياسية لتحليل بايت كود التخليل.

في التخليل، يتم حفظ الكائن مع رمز تشغيل ``GLOBAL`` الذي يصف مكان العثور على تنفيذ نوع الكائن، مثل:

::

    GLOBAL 'torchvision.models.resnet Resnet`

سيقوم محلل التبعيات بتجميع جميع عمليات ``GLOBAL`` ووضع علامة عليها كتبعيات للكائن المخلل.
للحصول على مزيد من المعلومات حول التخليل وتنسيق التخليل، يرجى الرجوع إلى `وثائق Python <https://docs.python.org/3/library/pickle.html>`_.

تحليل تبعيات وحدة نمطية
"""""""""""""""""""
عندما يتم تحديد وحدة نمطية Python كتبعيات، يقوم ``torch.package`` بالانتقال عبر التمثيل AST للوحدة النمطية والبحث عن عبارات الاستيراد مع
الدعم الكامل للأشكال القياسية: ``from x import y``، ``import z``، ``from w import v as u``، إلخ. عندما يتم العثور على إحدى عبارات الاستيراد هذه،
يسجل ``torch.package`` الوحدات النمطية المستوردة كتبعيات يتم بعد ذلك تحليلها بنفس طريقة المشي AST.

**ملاحظة**: التحليل النحوي AST له دعم محدود لبناء الجملة ``__import__(...)`` ولا يدعم استدعاءات ``importlib.import_module``. بشكل عام، لا يجب أن تتوقع أن يكتشف ``torch.package`` الاستيرادات الديناميكية.

إدارة التبعيات
^^^^^^^^^^
يقوم ``torch.package`` تلقائيًا بالعثور على وحدات Python النمطية التي تعتمد عليها التعليمات البرمجية والكائنات الخاصة بك. تُعرف هذه العملية باسم حل التبعيات.
بالنسبة لكل وحدة نمطية يعثر عليها محلل التبعيات، يجب عليك تحديد *إجراء* لاتخاذه.

الإجراءات المسموح بها هي:

* ``intern``: ضع هذه الوحدة النمطية في الحزمة.
* ``extern``: إعلان هذه الوحدة النمطية كتبعيات خارجية للحزمة.
* ``mock``: إنشاء وحدة نمطية وهمية لهذه الوحدة النمطية.
* ``deny``: سيؤدي الاعتماد على هذه الوحدة النمطية إلى حدوث خطأ أثناء تصدير الحزمة.

أخيرًا، هناك إجراء آخر مهم لا يعتبر جزءًا تقنيًا من ``torch.package``:

* إعادة العامل: إزالة تبعيات التعليمات البرمجية أو تغييرها.

لاحظ أن الإجراءات محددة فقط لوحدات Python النمطية بأكملها. لا توجد طريقة لتعبئة "مجرد" دالة أو فئة من وحدة نمطية وترك الباقي.
هذا عن قصد. لا تقدم Python حدودًا واضحة بين الكائنات المحددة في وحدة نمطية. الوحدة النمطية المحددة الوحيدة لوحدة التنظيم هي وحدة نمطية، لذا فهذا ما يستخدمه ``torch.package``.

يتم تطبيق الإجراءات على الوحدات النمطية باستخدام الأنماط. يمكن أن تكون الأنماط إما أسماء وحدات نمطية (``"foo.bar"``) أو أنماطًا فرعية (مثل ``"foo.**"``). يمكنك ربط نمط بإجراء باستخدام طرق على :class:`PackageExporter`، على سبيل المثال

::

    my_exporter.intern("torchvision.**")
    my_exporter.extern("numpy")

إذا تطابقت وحدة نمطية مع نمط، فسيتم تطبيق الإجراء المقابل عليها. بالنسبة لوحدة نمطية معينة، سيتم التحقق من الأنماط بالترتيب الذي تم تحديدها به،
وسيتم اتخاذ الإجراء الأول.

``intern``
""""""""""
إذا تم "intern" وحدة نمطية، فسيتم وضعها في الحزمة.

هذا الإجراء هو رمز نموذجك، أو أي رمز ذي صلة تريد تعبئته. على سبيل المثال، إذا كنت تحاول تعبئة ResNet من ``torchvision``،
فستحتاج إلى "intern" وحدة نمطية torchvision.models.resnet.

عند استيراد الحزمة، عندما تحاول التعليمات البرمجية المعبأة استيراد وحدة نمطية "intern"، فسوف يبحث PackageImporter داخل حزمتك عن تلك الوحدة النمطية.
إذا لم يتمكن من العثور على الوحدة النمطية، فسيتم إثارة خطأ. يضمن ذلك أن يتم عزل كل :class:`PackageImporter` عن بيئة التحميل - حتى إذا كان لديك ``my_interned_module`` متاحًا في كل من حزمتك وبيئة التحميل،
فسيستخدم :class:`PackageImporter` الإصدار الموجود في حزمتك فقط.

**ملاحظة**: يمكن "intern" فقط وحدات Python النمطية المصدرية. سيؤدي أنواع أخرى من الوحدات النمطية، مثل وحدات نمطية ملحقة C ووحدات نمطية بايت كود، إلى حدوث خطأ إذا
حاولت "intern" بها. يجب "mock" هذه الأنواع من الوحدات النمطية أو "extern"ها.

``extern``
""""""""""
إذا تم "extern" وحدة نمطية، فلن يتم تعبئتها. بدلاً من ذلك، سيتم إضافته إلى قائمة التبعيات الخارجية لهذه الحزمة. يمكنك العثور على هذه
القائمة في ``package_exporter.extern_modules``.

عند استيراد الحزمة، عندما تحاول التعليمات البرمجية المعبأة استيراد وحدة نمطية "extern"، سيستخدم :class:`PackageImporter` المستورد الافتراضي لـ Python للعثور
على تلك الوحدة النمطية، كما لو قمت بتشغيل ``importlib.import_module("my_externed_module")``. إذا لم يتمكن من العثور على تلك الوحدة النمطية، فسيتم إثارة خطأ.

بهذه الطريقة، يمكنك الاعتماد على مكتبات الطرف الثالث مثل ``numpy`` و ``scipy`` من داخل حزمتك دون الحاجة إلى تعبئتها أيضًا.

**تحذير**: إذا تم تغيير أي مكتبة خارجية بطريقة غير متوافقة مع الإصدارات السابقة، فقد تفشل حزمتك في التحميل. إذا كنت بحاجة إلى قابلية إعادة إنتاج طويلة الأجل
لحزمتك، فحاول الحد من استخدام "extern".

``mock``
""""""""
إذا تم "mock" وحدة نمطية، فلن يتم تعبئتها. بدلاً من ذلك، سيتم تعبئة وحدة نمطية وهمية في مكانها. ستسمح لك وحدة نمطية وهمية باسترداد
الكائنات منها (حتى لا يؤدي تشغيل "from my_mocked_module import foo" إلى حدوث خطأ)، ولكن أي استخدام لهذا الكائن سيرفع ``NotImplementedError``.

يجب استخدام "mock" للرمز الذي "تعرف" أنه لن يكون مطلوبًا في الحزمة المحملة، ولكنك تريد توفره للاستخدام في المحتويات غير المعبأة.
على سبيل المثال، رمز التهيئة/التكوين، أو الرمز المستخدم فقط للتصحيح/التدريب.

**تحذير**: بشكل عام، يجب استخدام "mock" كملاذ أخير. فهو يقدم اختلافات في السلوك بين التعليمات البرمجية المعبأة وغير المعبأة،
والتي قد تؤدي إلى حدوث ارتباك لاحقًا. يُفضل بدلاً من ذلك إعادة عاملي التعليمات البرمجية الخاصة بك لإزالة التبعيات غير المرغوب فيها.

إعادة العامل
"""""""""""
أفضل طريقة لإدارة التبعيات هي عدم وجود تبعيات على الإطلاق! غالبًا ما يمكن إعادة عاملي التعليمات البرمجية لإزالة التبعيات غير الضرورية. فيما يلي بعض
المبادئ التوجيهية لكتابة التعليمات البرمجية بتبعيات نظيفة (والتي تعد أيضًا ممارسات جيدة بشكل عام!):

**تضمين ما تستخدمه فقط**. لا تترك استيرادات غير مستخدمة في التعليمات البرمجية الخاصة بك. محلل التبعيات ليس ذكيًا بدرجة كافية لمعرفة أنها غير مستخدمة بالفعل،
وسيحاول معالجتها.

**قم بتأهيل استيراداتك**. على سبيل المثال، بدلاً من كتابة استيراد foo واستخدام ``foo.bar.baz`` لاحقًا، يُفضل كتابة ``from foo.bar import baz``. هذا يحدد تبعيتك الحقيقية بشكل أكثر دقة (``foo.bar``) ويتيح لمحلل التبعيات معرفة أنك لا تحتاج إلى كل شيء من ``foo``.

**قم بتقسيم الملفات الكبيرة ذات الوظائف غير ذات الصلة إلى ملفات أصغر**. إذا كانت وحدة "utils" الخاصة بك تحتوي على مجموعة من الوظائف غير ذات الصلة، فستحتاج أي وحدة نمطية تعتمد على "utils" إلى سحب العديد من التبعيات غير ذات الصلة، حتى إذا كنت بحاجة فقط إلى جزء صغير منها. يُفضل بدلاً من ذلك تحديد وحدات نمطية أحادية الغرض يمكن تعبئتها بشكل مستقل عن بعضها البعض.

الأنماط
""""""""
تسمح الأنماط بتحديد مجموعات من الوحدات النمطية باستخدام بناء جملة مناسب. يتبع بناء الجملة والسلوك للأنماط وظيفة Bazel/Buck
`glob() <https://docs.bazel.build/versions/master/be/functions.html#glob>`_.

تتكون الوحدة النمطية المرشحة التي نحاول مطابقتها مع نمط من قائمة من المقاطع مفصولة بسلسلة فاصلة، على سبيل المثال ``foo.bar.baz``.

يحتوي النمط على مقطع واحد أو أكثر. يمكن أن تكون المقاطع:

* سلسلة حرفية (مثل ``foo``)، والتي تتطابق تمامًا.
* سلسلة تحتوي على حرف البدل (مثل ``torch``، أو ``foo*baz*``). يتطابق حرف البدل مع أي سلسلة، بما في ذلك السلسلة الفارغة.
* نجمتان مزدوجتان (``**``). يتطابق هذا مع صفر أو أكثر من المقاطع الكاملة.

أمثلة:

* ``torch.**``: يتطابق مع ``torch`` وجميع وحداته الفرعية، مثل ``torch.nn`` و ``torch.nn.functional``.
* ``torch.*``: يتطابق مع ``torch.nn`` أو ``torch.functional``، ولكن ليس مع ``torch.nn.functional`` أو ``torch``.
* ``torch*.**``: يتطابق مع ``torch``، و ``torchvision``، وجميع وحداتها الفرعية

عند تحديد الإجراءات، يمكنك تمرير أنماط متعددة، على سبيل المثال

::

    exporter.intern(["torchvision.models.**", "torchvision.utils.**"])

ستتطابق الوحدة النمطية مع هذا الإجراء إذا تطابقت مع أي من الأنماط.

يمكنك أيضًا تحديد أنماط للاستبعاد، على سبيل المثال
::

    exporter.mock("**", exclude=["torchvision.**"])


لن تتطابق وحدة البرنامج مع إجراء الاستبعاد هذا إذا تطابقت مع أي من أنماط الاستبعاد. في هذا المثال، نقوم بمحاكاة جميع الوحدات النمطية باستثناء
``torchvision`` والوحدات الفرعية التابعة لها.

عندما يمكن لوحدة البرنامج أن تتطابق مع إجراءات متعددة، يتم تنفيذ الإجراء الأول المحدد.


``torch.package`` الحواف الحادة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
تجنب حالة التخزين العالمية في وحداتك النمطية
""""""""""""""""""""""""""""""""
تسهل Python ربط الأشياء وتشغيل التعليمات البرمجية على مستوى نطاق الوحدة النمطية. هذا جيد بشكل عام - بعد كل شيء، يتم ربط الدوال والصفوف بالأسماء بهذه الطريقة. ومع ذلك، تصبح الأمور أكثر تعقيدًا عندما تقوم بتعريف كائن على مستوى الوحدة النمطية بهدف تغييره، مما يقدم حالة تخزين عالمية قابلة للتغيير.

حالة التخزين العالمية القابلة للتغيير مفيدة جدًا - فيمكنها تقليل التعليمات البرمجية النمطية، والسماح بالتسجيل المفتوح في الجداول، وما إلى ذلك. ولكن ما لم يتم توظيفها بعناية، فقد تسبب تعقيدات عند استخدامها مع ``torch.package``.

يخلق كل :class: 'PackageImporter' بيئة مستقلة لمحتوياته. هذا أمر رائع لأنه يعني أننا نقوم بتحميل العديد من الحزم والتأكد من أنها معزولة عن بعضها البعض، ولكن عندما يتم كتابة الوحدات بطريقة تفترض حالة تخزين عالمية مشتركة، يمكن لهذا السلوك إنشاء أخطاء يصعب تصحيحها.

الأنواع غير مشتركة بين الحزم وبيئة التحميل
""""""""""""""""""""""""""""""
أي فئة تقوم باستيرادها من :class: 'PackageImporter' ستكون إصدارًا من الفئة الخاصة بمستورد الاستيراد هذا. على سبيل المثال:


::

    from foo import MyClass

    my_class_instance = MyClass()

    with PackageExporter(f) as exporter:
        exporter.save_module("foo")

    importer = PackageImporter(f)
    imported_MyClass = importer.import_module("foo").MyClass

    assert isinstance(my_class_instance, MyClass)  # يعمل
    assert isinstance(my_class_instance, imported_MyClass)  # خطأ!


في هذا المثال، ``MyClass`` و ``imported_MyClass`` *ليسوا من نفس النوع*. في هذا المثال المحدد، لدى ``MyClass`` و ``imported_MyClass`` نفس التنفيذ بالضبط، لذلك قد تعتقد أنه من الآمن اعتبارها من نفس الفئة. ولكن ضع في اعتبارك الموقف الذي يأتي فيه ``imported_MyClass`` من حزمة أقدم مع تنفيذ مختلف تمامًا لـ ``MyClass`` - في هذه الحالة، من غير الآمن اعتبارها من نفس الفئة.

تحت الغطاء، لكل مستورد بادئة تسمح له بتحديد الفئات بشكل فريد:


::

    print(MyClass.__name__)  # يطبع "foo.MyClass"
    print(imported_MyClass.__name__)  # يطبع <torch_package_0>.foo.MyClass


هذا يعني أنه لا ينبغي لك توقع نجاح فحوصات "isinstance" عندما يكون أحد الوسيطين من حزمة والآخر لا. إذا كنت بحاجة إلى هذه الوظيفة، فكر في الخيارات التالية:

* قم بالكتابة باستخدام أسلوب البط (duck typing) (مجرد استخدام الفئة بدلاً من التحقق الصريح من أنها من نوع معين).
* اجعل علاقة الكتابة جزءًا صريحًا من عقد الفئة. على سبيل المثال، يمكنك إضافة علامة تبويب للصفة ``self.handler = "handle_me_this_way"`` ولتتحقق شفرة العميل من قيمة ``handler`` بدلاً من التحقق من النوع مباشرة.


كيف يحافظ ``torch.package`` على عزل الحزم عن بعضها البعض
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ينشئ كل مثيل لـ :class: 'PackageImporter' بيئة مستقلة ومعزولة لوحداته النمطية وكائناته. لا يمكن للوحدات النمطية في الحزمة استيراد سوى الوحدات النمطية الأخرى المعلبة، أو الوحدات النمطية التي تم وضع علامة عليها على أنها "خارجية". إذا كنت تستخدم العديد من مثيلات :class: 'PackageImporter' لتحميل حزمة واحدة، فستحصل على بيئات مستقلة متعددة لا تتفاعل مع بعضها البعض.

يتم تحقيق ذلك عن طريق توسيع البنية الأساسية لاستيراد Python باستخدام مستورد مخصص. يوفر :class: 'PackageImporter' نفس واجهة برمجة التطبيقات الأساسية كمستورد "importlib"؛ أي أنه ينفذ أساليب "import_module" و "``__import__``".

عند استدعاء :meth: 'PackageImporter.import_module'، يقوم :class: 'PackageImporter' ببناء وإرجاع وحدة نمطية جديدة، تمامًا كما يفعل المستورد النظامي. ومع ذلك، يقوم :class: 'PackageImporter' بتصحيح الوحدة النمطية التي تمت إعادتها لاستخدام "self" (أي مثيل :class: 'PackageImporter' هذا) لتلبية طلبات الاستيراد المستقبلية عن طريق البحث في الحزمة بدلاً من البحث في بيئة Python الخاصة بالمستخدم.

التحريف
""""""""
لتجنب الالتباس ("هل هذا الكائن "foo.bar" من حزمتي، أم من بيئة Python الخاصة بي؟")، يقوم :class: 'PackageImporter' بتحريف "``__name__``" و "``__file__``" لجميع الوحدات المستوردة، عن طريق إضافة بادئة تحريف إليها.

بالنسبة لـ "``__name__``"، يصبح اسم مثل "torchvision.models.resnet18" "``<torch_package_0>.torchvision.models.resnet18``".

بالنسبة لـ "``__file__``"، يصبح اسم مثل "torchvision/models/resnet18.py" "``<torch_package_0>.torchvision/modules/resnet18.py``".

يساعد تحريف الاسم على تجنب التلاعب غير المقصود لأسماء الوحدات النمطية بين الحزم المختلفة، ويساعدك في تصحيح الأخطاء من خلال جعل آثار المكدس والبيانات المطبوعة توضح بشكل أكبر ما إذا كانت تشير إلى كود معلب أم لا. للحصول على تفاصيل حول التحريف، راجع الملاحظات الموجهة للمطورين في "mangling.md" في "torch/package/".


مرجع API
---------
.. autoclass:: torch.package.خطأ_التعبئة

.. autoclass:: torch.package.خطأ_المطابقة_الفارغة

.. autoclass:: torch.package.مصدر_التعبئة
  :members:

  .. automethod:: __init__

.. autoclass:: torch.package.مُستورد_التعبئة
  :members:

  .. automethod:: __init__

.. autoclass:: torch.package.الدليل
  :members:


.. تحتاج هذه الوحدة إلى توثيق. نقوم بإضافتها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.package.analyze.find_first_use_of_broken_modules
.. py:module:: torch.package.analyze.is_from_package
.. py:module:: torch.package.analyze.trace_dependencies
.. py:module:: torch.package.file_structure_representation
.. py:module:: torch.package.find_file_dependencies
.. py:module:: torch.package.glob_group
.. py:module:: torch.package.importer
.. py:module:: torch.package.package_exporter
.. py:module:: torch.package.package_importer