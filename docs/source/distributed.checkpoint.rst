.. role:: hidden
    :class: hidden-section

نقطة تفتيش موزعة - torch.distributed.checkpoint
==============================================

تدعم نقطة التفتيش الموزعة (DCP) تحميل وحفظ النماذج من مراتب متعددة بالتوازي.
وهي تتعامل مع إعادة التجزئة في وقت التحميل، مما يمكّن من الحفظ في بنية عنقودية وتحميلها في بنية أخرى.

تختلف DCP عن ``torch.save`` و ``torch.load`` بعدة طرق مهمة:

* ينتج ملفات متعددة لكل نقطة تفتيش، مع ملف واحد على الأقل لكل رتبة.
* تعمل في المكان، مما يعني أن النموذج يجب أن يخصص بياناته أولاً ويستخدم DCP ذلك التخزين بدلاً من ذلك.

نقاط الدخول لتحميل وحفظ نقطة تفتيش هي كما يلي:

.. automodule:: torch.distributed.checkpoint

.. currentmodule:: torch.distributed.checkpoint.state_dict_saver

.. autofunction:: save
.. autofunction:: async_save
.. autofunction:: save_state_dict

.. currentmodule:: torch.distributed.checkpoint.state_dict_loader

.. autofunction:: load
.. autofunction:: load_state_dict

الفصل التالي مفيد أيضًا للتخصيص الإضافي لآليات التجهيز المستخدمة لنقطة التفتيش غير المتزامنة (``torch.distributed.checkpoint.async_save``):

.. automodule:: torch.distributed.checkpoint.staging

.. autoclass:: torch.distributed.checkpoint.staging.AsyncStager
  :members:

.. autoclass:: torch.distributed.checkpoint.staging.BlockingAsyncStager
  :members:

بالإضافة إلى نقاط الدخول المذكورة أعلاه، توفر الكائنات "Stateful" الموصوفة أدناه تخصيصًا إضافيًا أثناء الحفظ/التحميل.

.. automodule:: torch.distributed.checkpoint.stateful

.. autoclass:: torch.distributed.checkpoint.stateful.Stateful
  :members:

يُظهر هذا `المثال <https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py>`_ كيفية استخدام نقطة تفتيش PyTorch الموزعة لحفظ نموذج FSDP.

تحدد الأنواع التالية واجهة الإدخال/الإخراج المستخدمة أثناء نقطة التفتيش:

.. autoclass:: torch.distributed.checkpoint.StorageReader
  :members:

.. autoclass:: torch.distributed.checkpoint.StorageWriter
  :members:

تحدد الأنواع التالية واجهة المخطط المستخدمة أثناء نقطة التفتيش:

.. autoclass:: torch.distributed.checkpoint.LoadPlanner
  :members:

.. autoclass:: torch.distributed.checkpoint.LoadPlan
  :members:

.. autoclass:: torch.distributed.checkpoint.ReadItem
  :members:

.. autoclass:: torch.distributed.checkpoint.SavePlanner
  :members:

.. autoclass:: torch.distributed.checkpoint.SavePlan
  :members:

.. autoclass:: torch.distributed.checkpoint.planner.WriteItem
  :members:

نقدم طبقة تخزين قائمة على نظام الملفات:

.. autoclass:: torch.distributed.checkpoint.FileSystemReader
  :members:

.. autoclass:: torch.distributed.checkpoint.FileSystemWriter
  :members:

نقدم التطبيقات الافتراضية لـ ``LoadPlanner`` و ``SavePlanner`` التي
يمكنها التعامل مع جميع البنيات الموزعة في PyTorch مثل FSDP و DDP و ShardedTensor و DistributedTensor.

.. autoclass:: torch.distributed.checkpoint.DefaultSavePlanner
  :members:

.. autoclass:: torch.distributed.checkpoint.DefaultLoadPlanner
  :members:

بسبب قرارات التصميم القديمة، قد تحتوي القواميس الحالة لـ ``FSDP`` و ``DDP`` على مفاتيح أو أسماء مؤهلة بشكل كامل مختلفة (مثل layer1.weight) حتى عندما يكون النموذج غير الموازي الأصلي متطابقًا. بالإضافة إلى ذلك، توفر ``FSDP`` أنواعًا مختلفة من قواميس حالة النموذج، مثل القواميس الحالة الكاملة والمجزأة. علاوة على ذلك، تستخدم قواميس حالة المحسن معرفات المعلمات بدلاً من الأسماء المؤهلة بالكامل لتحديد المعلمات، مما قد يتسبب في حدوث مشكلات عند استخدام التوازي (مثل التوازي الأنبوبي).

للتغلب على هذه التحديات، نقدم مجموعة من واجهات برمجة التطبيقات للمستخدمين لإدارة قواميس الحالة بسهولة. تعيد دالة ``get_model_state_dict`` قاموس حالة النموذج بمفاتيح متسقة مع تلك التي تعيدها قاموس حالة النموذج غير الموازي. وبالمثل، توفر ``get_optimizer_state_dict`` قاموس حالة المحسن بمفاتيح موحدة عبر جميع التوازيات المطبقة. لتحقيق هذا الاتساق، تحول ``get_optimizer_state_dict`` معرفات المعلمات إلى أسماء مؤهلة بالكامل متطابقة مع تلك الموجودة في قاموس حالة النموذج غير الموازي.

لاحظ أنه يمكن استخدام النتائج التي تعيدها هذه الواجهات مباشرة مع طريقتي ``torch.distributed.checkpoint.save()`` و ``torch.distributed.checkpoint.load()`` دون الحاجة إلى أي تحويلات إضافية.

يرجى ملاحظة أن هذه الميزة تجريبية، وقد تتغير تواقيع واجهات برمجة التطبيقات في المستقبل.

.. autofunction:: torch.distributed.checkpoint.state_dict.get_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.get_model_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.get_optimizer_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_model_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_optimizer_state_dict

.. autoclass:: torch.distributed.checkpoint.state_dict.StateDictOptions
   :members:

بالنسبة للمستخدمين المعتادين على استخدام ومشاركة النماذج بتنسيق ``torch.save``، يتم توفير الطرق التالية التي توفر المرافق غير المتصلة بالإنترنت للتحويل بين التنسيقات.

.. automodule:: torch.distributed.checkpoint.format_utils

.. currentmodule:: torch.distributed.checkpoint.format_utils

.. autofunction:: dcp_to_torch_save
.. autofunction:: torch_save_to_dcp

يمكن أيضًا استخدام الفئات التالية لتحميل وإعادة تجزئة النماذج عبر الإنترنت من تنسيق ``torch.save``.

.. autoclass:: torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader
   :members:

.. autoclass:: torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner
   :members:

تتوفر واجهات تجريبية التالية لتحسين الرصد في بيئات الإنتاج:

.. py:module:: torch.distributed.checkpoint.logger
.. py:module:: torch.distributed.checkpoint.logging_handlers