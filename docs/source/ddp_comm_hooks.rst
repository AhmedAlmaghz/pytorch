هوكات اتصال DDP
===============

هوكة اتصال DDP هي واجهة عامة للتحكم في كيفية تبادل التدرجات عبر العمال عن طريق تجاوز allreduce الفانيليا في
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.>`_.
يتم توفير بعض هوكات الاتصال المدمجة،
يمكن للمستخدمين بسهولة تطبيق أي من هذه الهوكات لتحسين الاتصال.
بالإضافة إلى ذلك، يمكن أيضًا أن يدعم واجهة هوك استراتيجيات اتصال محددة من قبل المستخدم لحالات الاستخدام المتقدمة.

كيفية استخدام هوك الاتصال؟
---------------------

لاستخدام هوك الاتصال، يحتاج المستخدم فقط إلى السماح لنموذج DDP بتسجيل
الهوك قبل حلقة التدريب كما هو موضح أدناه.

:func:`torch.nn.parallel.DistributedDataParallel.register_comm_hook`

ما الذي يعمل عليه هوك الاتصال؟
------------------------

يوفر هوك الاتصال طريقة مرنة ل allreduce التدرجات.
لذلك، فهو يعمل بشكل أساسي على التدرجات على كل نسخة قبل allreduce،
التي يتم تجميعها في دلاء لزيادة التداخل بين الاتصال والحساب.
وعلى وجه الخصوص، :class:`torch.distributed.GradBucket` يمثل دلو من تنسورات التدرج التي سيتم allreduce.

.. autoclass:: torch.distributed.GradBucket

.. autofunction:: torch.distributed.GradBucket.index
.. autofunction:: torch.distributed.GradBucket.buffer
.. autofunction:: torch.distributed.GradBucket.gradients
.. autofunction:: torch.distributed.GradBucket.is_last
.. autofunction:: torch.distributed.GradBucket.set_buffer
.. autofunction:: torch.distributed.GradBucket.parameters

هوكات الاتصال الافتراضية
------------------

هوكات الاتصال الافتراضية هي هوكات بسيطة **عديمة الحالة**، لذلك حالة الإدخال
في ``register_comm_hook`` هي إما مجموعة عمليات أو ``None``.
يكون إدخال ``bucket`` عبارة عن كائن :class:`torch.distributed.GradBucket`.

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.default_hooks
.. autofunction:: allreduce_hook
.. autofunction:: fp16_compress_hook
.. autofunction:: bf16_compress_hook

بالإضافة إلى ذلك، يتم توفير غلاف هوك الاتصال لدعم :meth:`~fp16_compress_hook` أو :meth:`~bf16_compress_hook` كغلاف،
والذي يمكن دمجه مع هوكات الاتصال الأخرى.

.. autofunction:: fp16_compress_wrapper
.. autofunction:: bf16_compress_wrapper

هوك PowerSGD
---------------------------

PowerSGD (`Vogels et al.، NeurIPS 2019 <https://arxiv.org/abs/1905.13727>`_)
هو خوارزمية ضغط التدرج، والتي يمكن أن توفر معدلات ضغط عالية جدًا وتسريع التدريب الموزع المحدود بالنطاق الترددي.
تحتاج هذه الخوارزمية إلى الحفاظ على كل من بعض فرط المعلمات وحالة الداخلية. لذلك، PowerSGD هوك الاتصال هو هوك **ذو حالة**،
ويحتاج المستخدم إلى توفير كائن الحالة المحدد أدناه.

حالة PowerSGD
^^^^^^^^^^^^^^^^

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook
.. autoclass:: PowerSGDState

هوكات PowerSGD
^^^^^^^^^^^^^^^^

.. warning ::
    عادةً ما يتطلب PowerSGD ذاكرة إضافية بنفس حجم تدرجات النموذج لتمكين التغذية الراجعة للأخطاء، والتي يمكن أن تعوض عن الاتصال المضغوط المتحيز وتحسين الدقة.

.. warning ::
    قد تتعارض هوكات PowerSGD مع `حزمة Apex automatic mixed precision <https://github.com/NVIDIA/apex>`_.
    يرجى استخدام PyTorch `native automatic mixed precision package <https://pytorch.org/docs/stable/amp.html>`_
    بدلا من ذلك.

.. autofunction:: powerSGD_hook
.. autofunction:: batched_powerSGD_hook

هوكات تصحيح الأخطاء
---------------

كما يوحي الاسم، يتم استخدام هوكات تصحيح الأخطاء **فقط** لأغراض تصحيح الأخطاء وتحسين الأداء.

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks

.. warning ::
    لا تخرج هوكات تصحيح الأخطاء بالضرورة النتائج الصحيحة.

.. autofunction:: noop_hook

تسجيل حالة هوكات الاتصال
------------------

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook

يمكن حفظ هوك الاتصال ذي الحالة كجزء من تسجيل نقاط نموذج الاتصال لتمكين إعادة تشغيل المدرب.
لجعل هوك قابلًا للتسلسل، يجب تعريف ``__setstate__`` و ``__getstate__``.

.. warning ::
    يجب أن يستبعد ``__getstate__`` السمات غير القابلة للتسلسل من قاموس تمت إعادته.

.. warning ::
    يجب أن يقوم ``__setstate__`` بتأسيس سمات غير متسلسلة بشكل صحيح، مستبعدة من حالة "مقدمة".

:class:`PowerSGDState` لديها ``__setstate__`` و ``__getstate__`` المنفذة ويمكن استخدامها كمرجع.

.. class:: PowerSGDState
    :noindex:

    .. automethod:: PowerSGDState.__getstate__
    .. automethod:: PowerSGDState.__setstate__

فيما يلي مثال بسيط وشامل لحفظ وإعادة تحميل حالة PowerSGD وهوك.

::

    import os
    import sys
    import tempfile
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    import torch.multiprocessing as mp

    from torch.nn.parallel import DistributedDataParallel
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(24,24)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(24,12)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

    def run_demo(demo_fn, world_size):
        mp.spawn(
            demo_fn,
            args=(world_size,),
            nprocs=world_size,
            join=True)

    def demo_serialization(rank, world_size):
        setup(rank, world_size)

        CHECKPOINT = tempfile.gettempdir() + "/checkpoint.pt"

        model = SimpleModel().to(rank)
        ddp_model = DistributedDataParallel(model, device_ids=[rank])

        powersgd_hook = powerSGD.powerSGD_hook
        powersgd_state = powerSGD.PowerSGDState(process_group=None)

        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        ddp_model.register_comm_hook(powersgd_state, powersgd_hook)

        state = {
            'state_dict': ddp_model.state_dict(),
            'comm_hook': powersgd_hook,
            'comm_hook_state': powersgd_state}

        if rank == 0:
            torch.save(state, CHECKPOINT)

        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(CHECKPOINT, map_location=map_location)

        new_ddp_model = DistributedDataParallel(SimpleModel().to(rank), device_ids=[rank])
        new_ddp_model.load_state_dict(checkpoint['state_dict'])
        powersgd_hook = checkpoint['comm_hook']
        powersgd_state = checkpoint['comm_hook_state']

        new_ddp_model.register_comm_hook(powersgd_state, powersgd_hook)

        if rank == 0:
            os.remove(CHECKPOINT)

        cleanup()

    if __name__ == "__main__":
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(demo_serialization, world_size)

الشكر والتقدير
----------

شكرًا جزيلاً لمؤلف ورقة PowerSGD **Thijs Vogels** على مراجعة الكود لـ
هوك اتصال PowerSGD، بالإضافة إلى
`تجارب المقارنة <https://observablehq.com/@tvogels/powersgd-benchmark>`_،
والتي تظهر أن أداء هوك اتصال PowerSGD يعادل
التنفيذ في `الأصل <https://arxiv.org/abs/1905.13727>`_.