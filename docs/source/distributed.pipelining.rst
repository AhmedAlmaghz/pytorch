.. role:: hidden
    :class: hidden-section

توازي الأنابيب
##########

.. note::
   ``torch.distributed.pipelining`` هو حاليًا في حالة ألفا وتحت التطوير. قد تكون تغييرات واجهة برمجة التطبيقات (API) ممكنة. تم نقله من مشروع `PiPPy <https://github.com/pytorch/PiPPy>`_ .

لماذا التوازي الأنبوبي؟
***************

توازي الأنابيب هو أحد التوازيات **البدائية** للتعلم العميق. يسمح بتقسيم **تنفيذ** نموذج بحيث يمكن تنفيذ **ميكروباتشات** متعددة لأجزاء مختلفة من كود النموذج في نفس الوقت. يمكن أن يكون توازي الأنابيب تقنية فعالة لـ:

* التدريب واسع النطاق
* مجموعات محدودة النطاق الترددي
* استنتاج النماذج الكبيرة

تتشارك السيناريوهات المذكورة أعلاه في خاصية مشتركة تتمثل في أن الحساب لكل جهاز لا يمكنه إخفاء التواصل للتوازي التقليدي، على سبيل المثال، تجميع الأوزان الكاملة لـ FSDP.

ما هو ``torch.distributed.pipelining``؟
*****************************************

بينما يعد التوصيل الواعد للقياس، إلا أنه غالبًا ما يكون من الصعب تنفيذه لأنه يحتاج إلى **تقسيم تنفيذ** نموذج بالإضافة إلى أوزان النموذج. غالبًا ما يتطلب تقسيم التنفيذ إجراء تغييرات في كود النموذج. يأتي جانب آخر من التعقيد من **جدولة الميكروبتشات في بيئة موزعة**، مع مراعاة **اعتماد تدفق البيانات**.

توفر حزمة "pipelining" مجموعة أدوات تقوم تلقائيًا بالأشياء المذكورة أعلاه، مما يسمح بتنفيذ سهل لتوازي الأنابيب في النماذج **العامة**.

يتكون من جزأين: **واجهة أمامية مقسمة** و **وقت تشغيل موزع**. تأخذ الواجهة الأمامية المقسمة كود النموذج كما هو، وتقسمه إلى "تقسيمات نموذج"، وتلتقط علاقة تدفق البيانات. يقوم وقت التشغيل الموزع بتنفيذ مراحل الأنابيب على أجهزة مختلفة بالتوازي، والتعامل مع أشياء مثل تقسيم الميكروبتشات، والجدولة، والاتصال، وانتشار التدرجات، وما إلى ذلك.

بشكل عام، توفر حزمة "pipelining" الميزات التالية:

* تقسيم كود النموذج بناءً على المواصفات البسيطة.
* الدعم الغني لجداول الأنابيب، بما في ذلك GPipe و 1F1B و Interleaved 1F1B و Looped BFS، وتوفير البنية الأساسية لكتابة الجداول المخصصة.
* دعم من الدرجة الأولى لتوازي الأنابيب عبر المضيف، حيث يتم استخدام PP عادةً (عبر وصلات الاتصال البطيئة).
* قابلية التركيب مع تقنيات PyTorch المتوازية الأخرى مثل الموازي للبيانات (DDP و FSDP) أو الموازي للموتر. يوضح مشروع `TorchTitan <https://github.com/pytorch/torchtitan>`_ تطبيق "3D Parallel" على نموذج Llama.

الخطوة 1: إنشاء ``PipelineStage``
*******************************

قبل أن نتمكن من استخدام ``PipelineSchedule``، نحتاج إلى إنشاء كائنات ``PipelineStage`` التي تلتف حول الجزء من النموذج الذي يتم تشغيله في تلك المرحلة. ``PipelineStage`` مسؤول عن تخصيص مخازن مؤقتة للاتصال وإنشاء عمليات الإرسال/الاستقبال للتواصل مع الأقران. يدير المخازن المؤقتة الوسيطة، على سبيل المثال، لنتائج الإرسال الأمامي التي لم يتم استهلاكها بعد، ويوفر أداة لتشغيل الخلفيات لمرحلة النموذج.

تحتاج "PipelineStage" إلى معرفة أشكال الإدخال والإخراج لمرحلة النموذج، حتى تتمكن من تخصيص المخازن المؤقتة للاتصال بشكل صحيح. يجب أن تكون الأشكال ثابتة، على سبيل المثال، لا يمكن أن تتغير الأشكال أثناء التشغيل من خطوة إلى أخرى. سيتم رفع فئة "PipeliningShapeError" إذا لم تتطابق الأشكال أثناء التشغيل مع الأشكال المتوقعة. عند التركيب مع تقنيات الموازاة الأخرى أو تطبيق الدقة المختلطة، يجب مراعاة هذه التقنيات حتى تتمكن "PipelineStage" من معرفة الشكل (ونوع البيانات) الصحيح لنتيجة وحدة المرحلة في وقت التشغيل.

يمكن للمستخدمين إنشاء مثيل "PipelineStage" مباشرة، عن طريق تمرير "nn.Module" الذي يمثل الجزء من النموذج الذي يجب تشغيله في المرحلة. قد يتطلب ذلك إجراء تغييرات على كود النموذج الأصلي. راجع المثال في: ref: `option_1_manual` .

بدلاً من ذلك، يمكن أن تستخدم الواجهة الأمامية المقسمة تقسيم الرسم البياني لتقسيم نموذجك إلى سلسلة من "nn.Module" تلقائيًا. تتطلب هذه التقنية إمكانية تتبع النموذج باستخدام "torch.Export". قابلية تركيب "nn.Module" الناتجة مع تقنيات الموازاة الأخرى تجريبية، وقد تتطلب بعض الحلول البديلة. قد يكون استخدام هذه الواجهة الأمامية أكثر جاذبية إذا لم يتمكن المستخدم من تغيير كود النموذج بسهولة. راجع: ref: `option_2_tracer` لمزيد من المعلومات.

الخطوة 2: استخدام ``PipelineSchedule`` للتنفيذ
*****************************************

الآن يمكننا توصيل "PipelineStage" بجدول الأنابيب، وتشغيل الجدول باستخدام بيانات الإدخال. إليك مثال على GPipe:

.. code-block:: python

   من torch.distributed.pipelining import ScheduleGPipe

   # إنشاء جدول
   الجدول = ScheduleGPipe(stage، n_microbatches)

   # بيانات الإدخال (الدفعة الكاملة)
   x = torch.randn(batch_size، in_dim، device=device)

   # تشغيل الأنبوب باستخدام الإدخال `x`
   # سيتم تقسيم `x` إلى ميكروبتشات تلقائيًا
   إذا كان الرتبة == 0:
       الجدول.الخطوة (x)
   آخر:
       الإخراج = الجدول.الخطوة ()

لاحظ أن الكود أعلاه يحتاج إلى إطلاقه لكل عامل، لذا نستخدم خدمة التشغيل لإطلاق عمليات متعددة:

.. code-block:: bash

   torchrun --nproc_per_node=2 example.py

خيارات لتقسيم نموذج
**************

.. _option_1_manual:

الخيار 1: تقسيم نموذج يدويًا
====================

لإنشاء مثيل "PipelineStage" مباشرةً، يكون المستخدم مسؤولاً عن توفير مثيل "nn.Module" واحد يمتلك "nn.Parameters" و "nn.Buffers" ذات الصلة، ويحدد طريقة "forward()" التي تنفذ العمليات ذات الصلة بتلك المرحلة. على سبيل المثال، تعرض نسخة مختصرة من فئة Transformer المحددة في Torchtitan نمطًا لبناء نموذج قابل للتقسيم بسهولة.

.. code-block:: python

   class Transformer(nn.Module):
       def __init__(self، model_args: ModelArgs):
           super().__init__()

           self.tok_embeddings = nn.Embedding(...)

           # باستخدام ModuleDict، يمكننا حذف الطبقات دون التأثير على الأسماء،
           # ضمان حفظ نقاط التفتيش وتحميلها بشكل صحيح.
           self.layers = torch.nn.ModuleDict()
           for layer_id in range(model_args.n_layers):
               self.layers[str(layer_id)] = TransformerBlock(...)

           self.output = nn.Linear(...)

       def forward(self، tokens: torch.Tensor):
           # يمكّن التعامل مع الطبقات "None" في وقت التشغيل من تقسيم الأنابيب بسهولة
           h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

           for layer in self.layers.values():
               h = layer(h، self.freqs_cis)

           h = self.norm(h) if self.norm else h
           output = self.output(h).float() if self.output else h
           return output

يمكن تكوين نموذج محدد بهذه الطريقة بسهولة لكل مرحلة عن طريق أولاً تهيئة النموذج بالكامل (باستخدام جهاز ميتا لتجنب أخطاء OOM)، وحذف الطبقات غير المرغوب فيها لتلك المرحلة، ثم إنشاء "PipelineStage" الذي يلتف حول النموذج. على سبيل المثال:

.. code-block:: python

   مع جهاز الشعلة ("الميتا"):
       التأكيد num_stages == 2، "هذا مثال بسيط على مرحلتين"

       # نقوم ببناء النموذج بالكامل، ثم نقوم بحذف الأجزاء التي لا نحتاجها لهذه المرحلة
       # في الممارسة العملية، يمكن القيام بذلك باستخدام دالة مساعدة تقسم الطبقات تلقائيًا عبر المراحل.
       النموذج = محول ()

       إذا كان stage_index == 0:
           # إعداد نموذج المرحلة الأولى
           حذف النموذج.الطبقات ["1"]
           النموذج.norm = None
           النموذج.output = None

       elif stage_index == 1:
           # إعداد نموذج المرحلة الثانية
           النموذج.tok_embeddings = None
           حذف النموذج.الطبقات ["0"]

       من الشعلة.distributed.pipelining import PipelineStage
       stage = PipelineStage(
           النموذج،
           stage_index،
           num_stages،
           الجهاز،
           input_args=example_input_microbatch،
       )


تحتاج "PipelineStage" إلى حجة مثال "input_args" التي تمثل إدخال وقت التشغيل للمرحلة، والتي ستكون دفعة صغيرة واحدة من بيانات الإدخال. يتم تمرير هذه الحجة عبر طريقة الإرسال الأمامي لوحدة المرحلة لتحديد أشكال الإدخال والإخراج المطلوبة للاتصال.

عند التركيب مع تقنيات الموازاة الأخرى للبيانات أو النماذج، قد تكون "output_args" مطلوبة أيضًا، إذا تأثر شكل/نوع بيانات جزء النموذج.


.. _option_2_tracer:

الخيار 2: تقسيم نموذج تلقائيًا
=====================

إذا كان لديك نموذج كامل ولا تريد قضاء الوقت في تعديله إلى سلسلة من "أجزاء النموذج"، فإن واجهة برمجة التطبيقات "بايبلاين" موجودة للمساعدة.
فيما يلي مثال مختصر:

.. code-block:: python

  class Model(torch.nn.Module):
      def __init__(self) -> None:
          super().__init__()
          self.emb = torch.nn.Embedding(10, 3)
          self.layers = torch.nn.ModuleList(
              Layer() for _ in range(2)
          )
          self.lm = LMHead()

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          x = self.emb(x)
          for layer in self.layers:
              x = layer(x)
          x = self.lm(x)
          return x


إذا قمنا بطباعة النموذج، يمكننا أن نرى تسلسلات هرمية متعددة، مما يجعل من الصعب التقسيم يدويًا::

  Model(
    (emb): Embedding(10, 3)
    (layers): ModuleList(
      (0-1): 2 x Layer(
        (lin): Linear(in_features=3, out_features=3, bias=True)
      )
    )
    (lm): LMHead(
      (proj): Linear(in_features=3, out_features=3, bias=True)
    )
  )

دعونا نرى كيف تعمل واجهة برمجة التطبيقات "بايبلاين":

.. code-block:: python

  from torch.distributed.pipelining import pipeline, SplitPoint

  # مثال على إدخال الميكروبتش
  x = torch.LongTensor([1, 2, 4, 5])

  pipe = pipeline(
      module=mod,
      mb_args=(x,),
      split_spec={
          "layers.1": SplitPoint.BEGINNING,
      }
  )

تقوم واجهة برمجة التطبيقات "بايبلاين" بتقسيم نموذجك بناءً على "split_spec"، حيث
يشير "SplitPoint.BEGINNING" إلى إضافة نقطة تقسيم
*قبل* تنفيذ وحدة فرعية معينة في دالة "forward"، وبالمثل، يشير "SplitPoint.END" إلى نقطة التقسيم *بعد* ذلك.

إذا قمنا بـ "print(pipe)"، يمكننا أن نرى::

  GraphModule(
    (submod_0): GraphModule(
      (emb): InterpreterModule()
      (layers): Module(
        (0): InterpreterModule(
          (lin): InterpreterModule()
        )
      )
    )
    (submod_1): GraphModule(
      (layers): Module(
        (1): InterpreterModule(
          (lin): InterpreterModule()
        )
      )
      (lm): InterpreterModule(
        (proj): InterpreterModule()
      )
    )
  )

  def forward(self, x):
      submod_0 = self.submod_0(x);  x = None
      submod_1 = self.submod_1(submod_0);  submod_0 = None
      return (submod_1,)


تمثل "أجزاء النموذج" الوحدات الفرعية (submod_0، submod_1)، ويتم إعادة بناء كل منها باستخدام عمليات النموذج الأصلي والأوزان والهياكل الهرمية. بالإضافة إلى ذلك، يتم إعادة بناء دالة "forward" على مستوى "الجذر" لالتقاط تدفق البيانات بين هذه الأقسام. وسيقوم وقت تشغيل "البايبلاين" لاحقًا بتشغيل تدفق البيانات هذا بطريقة موزعة.

يوفر كائن "بايبلاين" طريقة لاسترداد "أجزاء النموذج":

.. code-block:: python

  stage_mod : nn.Module = pipe.get_stage_module(stage_idx)

تكون "stage_mod" المرتجعة عبارة عن "nn.Module"، والتي يمكنك من خلالها إنشاء محدد لسرعة التعلم، أو حفظ نقاط التفتيش أو تحميلها، أو تطبيق عمليات موازية أخرى.

يسمح "بايبلاين" أيضًا بإنشاء وقت تشغيل مرحلة موزعة على جهاز معين
إعطاء "مجموعة عمليات":

.. code-block:: python

  stage = pipe.build_stage(stage_idx, device, group)

بدلاً من ذلك، إذا كنت ترغب في بناء وقت تشغيل المرحلة لاحقًا بعد إجراء بعض
التعديلات على "stage_mod"، فيمكنك استخدام إصدار وظيفي من
واجهة برمجة تطبيقات "build_stage". على سبيل المثال:

.. code-block:: python

  from torch.distributed.pipelining import build_stage
  from torch.nn.parallel import DistributedDataParallel

  dp_mod = DistributedDataParallel(stage_mod)
  info = pipe.info()
  stage = build_stage(dp_mod, stage_idx, info, device, group)

.. note::
  تستخدم واجهة "بايبلاين" الأمامية أداة تتبع (torch.export) لالتقاط نموذجك
  في رسم بياني واحد. إذا كان نموذجك غير قابل للرسم الكامل، فيمكنك استخدام
  واجهة برمجة التطبيقات اليدوية أدناه.


أمثلة Hugging Face
******************

في مستودع "PiPPy" <https://github.com/pytorch/PiPPy> حيث تم إنشاء هذه الحزمة
في الأصل، قمنا بالاحتفاظ بأمثلة بناءً على نماذج Hugging Face غير المعدلة.
راجع دليل "examples/huggingface
<https://github.com/pytorch/PiPPy/tree/main/examples/huggingface>`_.

تشمل الأمثلة ما يلي:

* `GPT2 <https://github.com/pytorch/PiPPy/tree/main/examples/huggingface/pippy_gpt2.py>`_
* `Llama <https://github.com/pytorch/PiPPy/tree/main/examples/llama>`_


نظرة فنية متعمقة
*************

كيف تقوم واجهة برمجة التطبيقات "بايبلاين" بتقسيم نموذج؟
=======================================

أولاً، تحول واجهة برمجة التطبيقات "بايبلاين" نموذجنا إلى رسم بياني موجه غير دوري (DAG)
من خلال تتبع النموذج. إنه يتعقب النموذج باستخدام "torch.export" - أداة التقاط الرسم البياني الكامل لـ PyTorch 2.

بعد ذلك، يقوم بتجميع **العمليات والبارامترات** اللازمة لكل مرحلة
في وحدة فرعية معاد بناؤها: "submod_0"، "submod_1"، ...

على عكس طرق الوصول إلى الوحدة الفرعية التقليدية مثل "Module.children()"، فإن واجهة برمجة التطبيقات "بايبلاين" لا تقطع بنية الوحدة النمطية للنموذج فحسب، بل أيضًا دالة **forward** للنموذج.

هذا أمر ضروري لأن بنية النموذج مثل "Module.children()" تلتقط المعلومات فقط أثناء "Module.__init__()"، ولا تلتقط أي
معلومات حول "Module.forward()". وبعبارة أخرى، تفتقر "Module.children()"
معلومات حول الجوانب الرئيسية لتقسيم النموذج:

* ترتيب تنفيذ الوحدات الفرعية في "forward"
* تدفقات التنشيط بين الوحدات الفرعية
* ما إذا كانت هناك أي عوامل تشغيل وظيفية بين الوحدات الفرعية (على سبيل المثال،
  لن يتم التقاط عمليات "relu" أو "add" بواسطة "Module.children()").

من ناحية أخرى، تضمن واجهة برمجة التطبيقات "بايبلاين" الحفاظ على سلوك "forward"
بشكل حقيقي. كما أنه يلتقط تدفق التنشيط بين الأقسام،
مساعدة وقت تشغيل الموزع على إجراء مكالمات الإرسال/الاستقبال الصحيحة دون تدخل بشري.

تتمثل إحدى ميزات واجهة برمجة التطبيقات "بايبلاين" في إمكانية وجود نقاط التقسيم على
مستويات عشوائية داخل التسلسل الهرمي للنموذج. في الأقسام المقسمة، سيتم إعادة بناء التسلسل الهرمي للنموذج الأصلي
المتعلق بذلك القسم دون أي تكلفة عليك.
ونتيجة لذلك، ستظل الأسماء المؤهلة بالكامل (FQNs) التي تشير إلى وحدة فرعية أو معلمة
صحيحة، ويمكن للخدمات التي تعتمد على FQNs (مثل FSDP أو TP أو
التخزين المؤقت) لا تزال تعمل مع الوحدات النمطية المقسمة الخاصة بك دون أي تغيير تقريبًا في التعليمات البرمجية.


تنفيذ الجدول الزمني الخاص بك
******************************

يمكنك تنفيذ جدول زمني للبايبلاين الخاص بك عن طريق توسيع إحدى الفئتين التاليتين:

* ``PipelineScheduleSingle``
* ``PipelineScheduleMulti``

يُستخدم "PipelineScheduleSingle" للجداول الزمنية التي تقوم بتعيين *مرحلة واحدة فقط* لكل رتبة.
يُستخدم "PipelineScheduleMulti" للجداول الزمنية التي تقوم بتعيين مراحل متعددة لكل رتبة.

على سبيل المثال، "ScheduleGPipe" و "Schedule1F1B" هما فئتان فرعيتان من "PipelineScheduleSingle".
بينما "ScheduleFlexibleInterleaved1F1B" و "ScheduleInterleaved1F1B" و "ScheduleLoopedBFS"
هي فئات فرعية من "PipelineScheduleMulti".


التسجيل
*******

يمكنك تشغيل تسجيل إضافي باستخدام متغير البيئة `TORCH_LOGS` من [`torch._logging`] (https://pytorch.org/docs/main/logging.html#module-torch._logging):

* `TORCH_LOGS=+pp` ستعرض رسائل `logging.DEBUG` وجميع المستويات أعلاها.
* `TORCH_LOGS=pp` ستعرض رسائل `logging.INFO` والمستويات أعلاها.
* `TORCH_LOGS=-pp` ستعرض رسائل `logging.WARNING` والمستويات أعلاها.


مرجع واجهة برمجة التطبيقات
*******************

.. automodule:: torch.distributed.pipelining

واجهات برمجة تطبيقات تقسيم النماذج
========================

مجموعة واجهات برمجة التطبيقات التالية تحول نموذجك إلى تمثيل بايبلاين.

.. currentmodule:: torch.distributed.pipelining

.. autoclass:: SplitPoint

.. autofunction:: pipeline

.. autoclass:: Pipe

.. autofunction:: pipe_split

مرافق الميكروبتش
===========

.. automodule:: torch.distributed.pipelining.microbatch

.. currentmodule:: torch.distributed.pipelining.microbatch

.. autoclass:: TensorChunkSpec

.. autofunction:: split_args_kwargs_into_chunks

.. autofunction:: merge_chunks

مراحل البايبلاين
==========

.. automodule:: torch.distributed.pipelining.stage

.. currentmodule:: torch.distributed.pipelining.stage

.. autoclass:: PipelineStage

.. autofunction:: build_stage

جداول زمنية للبايبلاين
==============

.. automodule:: torch.distributed.pipelining.schedules

.. currentmodule:: torch.distributed.pipelining.schedules

.. autoclass:: ScheduleGPipe

.. autoclass:: Schedule1F1B

.. autoclass:: ScheduleFlexibleInterleaved1F1B

.. autoclass:: ScheduleInterleaved1F1B

.. autoclass:: ScheduleLoopedBFS

.. autoclass:: ScheduleInterleavedZeroBubble

.. autoclass:: PipelineScheduleSingle
  :members:

.. autoclass:: PipelineScheduleMulti
  :members: