.. role:: hidden
    :class: hidden-section

:hidden:`torch.nn`
*******************

يحتوي هذا النص على واجهة برمجة تطبيقات الشبكة العصبية في PyTorch.

.. currentmodule:: torch.nn

.. autosummary::
    :nosignatures:
    :toctree: generated/

    BatchNorm1d
    BatchNorm2d
    BatchNorm3d
    SyncBatchNorm
    LayerNorm
    InstanceNorm1d
    InstanceNorm2d
    InstanceNorm3d
    GroupNorm
    LocalResponseNorm
    CrossMapLRN2d
    Dropout
    Dropout2d
    Dropout3d
    AlphaDropout
    FeatureAlphaDropout
    DropoutLayer
    GaussianDropout
    Identity
    ELU
    SELU
    GELU
    SiLU
    Mish
    Hardswish
    Hardshrink
    LeakyReLU
    LogSigmoid
    PReLU
    ReLU
    ReLU6
    RReLU
    Softplus
    Softshrink
    Tanh
    Tanhshrink
    Threshold
    Softmin
    Softmax
    Softmax2d
    LogSoftmax
    AdaptiveLogSoftmaxWithLoss
    NLLLoss
    BCEWithLogitsLoss
    BCELoss
    BCELoss
    MarginRankingLoss
    HingeEmbeddingLoss
    MultiLabelMarginLoss
    MultiLabelSoftMarginLoss
    CosineEmbeddingLoss
    MultiMarginLoss
    TripletMarginLoss
    L1Loss
    MSELoss
    SmoothL1Loss
    KLDivLoss
    HubberLoss
    MultiLabelSoftMarginLoss
    CTCLoss
    PoissonNLLLoss
    L1Loss
    NLLLoss2d
    SmoothL1Loss
    CrossEntropyLoss
    CTCLoss
    NTXentLoss
    TransformerLoss
    CTCLoss
    CTMultiplier
    CosineSimilarity
    PairwiseDistance
    L1Loss
    MSELoss
    MarginRankingLoss
    MultiLabelMarginLoss
    MultiMarginLoss
    MultiLabelSoftMarginLoss
    TripletMarginLoss
    CosineEmbeddingLoss
    CosineSimilarity
    TripletMarginWithDistanceLoss
    MultiLabelSoftMarginLoss
    SmoothL1Loss
    SoftMarginLoss
    MultiLabelMarginLoss
    MultiLabelSoftMarginLoss
    FocalLoss
    ClassBalancedLoss
    LovaszSoftmax
    LovaszHinge
    FocalCosineLoss
    FocalLoss
    CircleLoss
    TripletLoss


===================================
.. automodule:: torch.nn
.. automodule:: torch.nn.modules

هذه هي اللبنات الأساسية للرسوم البيانية:

.. contents:: torch.nn
    :depth: 2
    :local:
    :backlinks: أعلى

.. currentmodule:: torch.nn

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~parameter.Buffer
    ~parameter.Parameter
    ~parameter.UninitializedParameter
    ~parameter.UninitializedBuffer

الحاويات
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Module
    Sequential
    ModuleList
    ModuleDict
    ParameterList
    ParameterDict

خطافات عالمية للوحدة

.. currentmodule:: torch.nn.modules.module
.. autosummary::
    :toctree: generated
    :nosignatures:

    register_module_forward_pre_hook
    register_module_forward_hook
    register_module_backward_hook
    register_module_full_backward_pre_hook
    register_module_full_backward_hook
    register_module_buffer_registration_hook
    register_module_module_registration_hook
    register_module_parameter_registration_hook

.. currentmodule:: torch

طبقات التجزئة
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Conv1d
    nn.Conv2d
    nn.Conv3d
    nn.ConvTranspose1d
    nn.ConvTranspose2d
    nn.ConvTranspose3d
    nn.LazyConv1d
    nn.LazyConv2d
    nn.LazyConv3d
    nn.LazyConvTranspose1d
    nn.LazyConvTranspose2d
    nn.LazyConvTranspose3d
    nn.Unfold
    nn.Fold

طبقات التجميع
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.MaxPool1d
    nn.MaxPool2d
    nn.MaxPool3d
    nn.MaxUnpool1d
    nn.MaxUnpool2d
    nn.MaxUnpool3d
    nn.AvgPool1d
    nn.AvgPool2d
    nn.AvgPool3d
    nn.FractionalMaxPool2d
    nn.FractionalMaxPool3d
    nn.LPPool1d
    nn.LPPool2d
    nn.LPPool3d
    nn.AdaptiveMaxPool1d
    nn.AdaptiveMaxPool2d
    nn.AdaptiveMaxPool3d
    nn.AdaptiveAvgPool1d
    nn.AdaptiveAvgPool2d
    nn.AdaptiveAvgPool3d

طبقات الحشو
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ReflectionPad1d
    nn.ReflectionPad2d
    nn.ReflectionPad3d
    nn.ReplicationPad1d
    nn.ReplicationPad2d
    nn.ReplicationPad3d
    nn.ZeroPad1d
    nn.ZeroPad2d
    nn.ZeroPad3d
    nn.ConstantPad1d
    nn.ConstantPad2d
    nn.ConstantPad3d
    nn.CircularPad1d
    nn.CircularPad2d
    nn.CircularPad3d

التنشيطات غير الخطية (المجموع المرجح، اللانخطية)
---------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ELU
    nn.Hardshrink
    nn.Hardsigmoid
    nn.Hardtanh
    nn.Hardswish
    nn.LeakyReLU
    nn.LogSigmoid
    nn.MultiheadAttention
    nn.PReLU
    nn.ReLU
    nn.ReLU6
    nn.RReLU
    nn.SELU
    nn.CELU
    nn.GELU
    nn.Sigmoid
    nn.SiLU
    nn.Mish
    nn.Softplus
    nn.Softshrink
    nn.Softsign
    nn.Tanh
    nn.Tanhshrink
    nn.Threshold
    nn.GLU

التنشيطات غير الخطية (أخرى)
------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Softmin
    nn.Softmax
    nn.Softmax2d
    nn.LogSoftmax
    nn.AdaptiveLogSoftmaxWithLoss

طبقات التوحيد القياسي
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.BatchNorm1d
    nn.BatchNorm2d
    nn.BatchNorm3d
    nn.LazyBatchNorm1d
    nn.LazyBatchNorm2d
    nn.LazyBatchNorm3d
    nn.GroupNorm
    nn.SyncBatchNorm
    nn.InstanceNorm1d
    nn.InstanceNorm2d
    nn.InstanceNorm3d
    nn.LazyInstanceNorm1d
    nn.LazyInstanceNorm2d
    nn.LazyInstanceNorm3d
    nn.LayerNorm
    nn.LocalResponseNorm
    nn.RMSNorm

الطبقات المتكررة
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RNNBase
    nn.RNN
    nn.LSTM
    nn.GRU
    nn.RNNCell
    nn.LSTMCell
    nn.GRUCell

طبقات المحول
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Transformer
    nn.TransformerEncoder
    nn.TransformerDecoder
    nn.TransformerEncoderLayer
    nn.TransformerDecoderLayer

الطبقات الخطية
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Identity
    nn.Linear
    nn.Bilinear
    nn.LazyLinear

طبقات التوقف
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Dropout
    nn.Dropout1d
    nn.Dropout2d
    nn.Dropout3d
    nn.AlphaDropout
    nn.FeatureAlphaDropout

الطبقات المتناثرة
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Embedding
    nn.EmbeddingBag

وظائف المسافة
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.CosineSimilarity
    nn.PairwiseDistance

وظائف الخسارة
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.L1Loss
    nn.MSELoss
    nn.CrossEntropyLoss
    nn.CTCLoss
    nn.NLLLoss
    nn.PoissonNLLLoss
    nn.GaussianNLLLoss
    nn.KLDivLoss
    nn.BCELoss
    nn.BCEWithLogitsLoss
    nn.MarginRankingLoss
    nn.HingeEmbeddingLoss
    nn.MultiLabelMarginLoss
    nn.HuberLoss
    nn.SmoothL1Loss
    nn.SoftMarginLoss
    nn.MultiLabelSoftMarginLoss
    nn.CosineEmbeddingLoss
    nn.MultiMarginLoss
    nn.TripletMarginLoss
    nn.TripletMarginWithDistanceLoss

طبقات الرؤية
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.PixelShuffle
    nn.PixelUnshuffle
    nn.Upsample
    nn.UpsamplingNearest2d
    nn.UpsamplingBilinear2d

طبقات الخلط
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ChannelShuffle

طبقات DataParallel (متعددة GPU، موزعة)
--------------------------------------------
.. automodule:: torch.nn.parallel
.. currentmodule:: torch

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.DataParallel
    nn.parallel.DistributedDataParallel

الأدوات المساعدة
---------
.. automodule:: torch.nn.utils

من وحدة "torch.nn.utils":

وظائف الأدوات المساعدة لتقليم تدرجات المعلمات.

.. currentmodule:: torch.nn.utils
.. autosummary::
    :toctree: generated
    :nosignatures:

    clip_grad_norm_
    clip_grad_norm
    clip_grad_value_

وظائف الأدوات المساعدة لتقطيع وتسطيح معلمات الوحدة إلى ناقل واحد ومنه.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parameters_to_vector
    vector_to_parameters

وظائف الأدوات المساعدة لدمج الوحدات مع وحدات BatchNorm.

.. autosummary::
    :toctree: generated
    :nosignatures:

    fuse_conv_bn_eval
    fuse_conv_bn_weights
    fuse_linear_bn_eval
    fuse_linear_bn_weights

وظائف الأدوات المساعدة لتحويل تنسيق ذاكرة معلمات الوحدة.

.. autosummary::
    :toctree: generated
    :nosignatures:

    convert_conv2d_weight_memory_format
    convert_conv3d_weight_memory_format

وظائف الأدوات المساعدة لتطبيق وإزالة التوحيد الوزني لمعلمات الوحدة.

.. autosummary::
    :toctree: generated
    :nosignatures:

    weight_norm
    remove_weight_norm
    spectral_norm
    remove_spectral_norm

وظائف الأدوات المساعدة لتهيئة معلمات الوحدة.

.. autosummary::
    :toctree: generated
    :nosignatures:

    skip_init

فئات ووظائف الأدوات المساعدة لتشذيب معلمات الوحدة.

.. autosummary::
    :toctree: generated
    :nosignatures:

    prune.BasePruningMethod
    prune.PruningContainer
    prune.Identity
    prune.RandomUnstructured
    prune.L1Unstructured
    prune.RandomStructured
    prune.LnStructured
    prune.CustomFromMask
    prune.identity
    prune.random_unstructured
    prune.l1_unstructured
    prune.random_structured
    prune.ln_structured
    prune.global_unstructured
    prune.custom_from_mask
    prune.remove
    prune.is_pruned

التمثيلات المنفذة باستخدام وظيفة التمثيل الجديدة
في: func: torch.nn.utils.parameterize.register_parametrization `.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parametrizations.orthogonal
    parametrizations.weight_norm
    parametrizations.spectral_norm

وظائف الأدوات المساعدة لتمثيل المتوترات على الوحدات الموجودة.
يرجى ملاحظة أن هذه الوظائف يمكن استخدامها لتمثيل معلمة أو عازل معين
نظرًا لوظيفة محددة تقوم بالتعيين من مساحة الإدخال إلى مساحة التمثيل. إنها ليست تمثيلات
من شأنها أن تحول كائنًا إلى معلمة. راجع
`تمثيل البرنامج التعليمي <https://pytorch.org/tutorials/intermediate/parametrizations.html>`_
لمزيد من المعلومات حول كيفية تنفيذ التمثيلات الخاصة بك.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parametrize.register_parametrization
    parametrize.remove_parametrizations
    parametrize.cached
    parametrize.is_parametrized

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    parametrize.ParametrizationList

وظائف الأدوات المساعدة لاستدعاء وحدة معينة بطريقة غير حالية.

.. autosummary::
    :toctree: generated
    :nosignatures:

    stateless.functional_call

وظائف الأدوات المساعدة في الوحدات الأخرى

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:

    nn.utils.rnn.PackedSequence
    nn.utils.rnn.pack_padded_sequence
    nn.utils.rnn.pad_packed_sequence
    nn.utils.rnn.pad_sequence
    nn.utils.rnn.pack_sequence
    nn.utils.rnn.unpack_sequence
    nn.utils.rnn.unpad_sequence

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Flatten
    nn.Unflatten

الوظائف الكمية
--------------------

يشير التكميم إلى تقنيات لأداء الحسابات وتخزين المتوترات عند عرض بت أقل من
دقة النقطة العائمة. تدعم PyTorch كل من التكميم الخطي غير المتماثل لكل تنس و
التكميم الخطي غير المتماثل لكل قناة. لمعرفة المزيد حول كيفية استخدام الوظائف الكمية في PyTorch، يرجى الرجوع إلى
:ref:`quantization-doc` الوثائق.

تهيئة الوحدات الكسولة
بالتأكيد! هذا هو النص المترجم إلى اللغة العربية بتنسيق ReStructuredText:

---------------------------

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.lazy.LazyModuleMixin

المترادفات
-----------

ما يلي هي مرادفات لنظيراتها في ``torch.nn``:

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.normalization.RMSNorm

.. تحتاج هذه الوحدة إلى توثيق. يتم إضافتها هنا في الوقت الحالي
.. لأغراض التتبع
.. py:module:: torch.nn.backends
.. py:module:: torch.nn.utils.stateless
.. py:module:: torch.nn.backends.thnn
.. py:module:: torch.nn.common_types
.. py:module:: torch.nn.cpp
.. py:module:: torch.nn.functional
.. py:module:: torch.nn.grad
.. py:module:: torch.nn.init
.. py:module:: torch.nn.modules.activation
.. py:module:: torch.nn.modules.adaptive
.. py:module:: torch.nn.modules.batchnorm
.. py:module:: torch.nn.modules.channelshuffle
.. py:module:: torch.nn.modules.container
.. py:module:: torch.nn.modules.conv
.. py:module:: torch.nn.modules.distance
.. py:module:: torch.nn.modules.dropout
.. py:module:: torch.nn.modules.flatten
.. py:module:: torch.nn.modules.fold
.. py:module:: torch.nn.modules.instancenorm
.. py:module:: torch.nn.modules.lazy
.. py:module:: torch.nn.modules.linear
.. py:module:: torch.nn.modules.loss
.. py:module:: torch.nn.modules.module
.. py:module:: torch.nn.modules.normalization
.. py:module:: torch.nn.modules.padding
.. py:module:: torch.nn.modules.pixelshuffle
.. py:module:: torch.nn.modules.pooling
.. py:module:: torch.nn.modules.rnn
.. py:module:: torch.nn.modules.sparse
.. py:module:: torch.nn.modules.transformer
.. py:module:: torch.nn.modules.upsampling
.. py:module:: torch.nn.modules.utils
.. py:module:: torch.nn.parallel.comm
.. py:module:: torch.nn.parallel.distributed
.. py:module:: torch.nn.parallel.parallel_apply
.. py:module:: torch.nn.parallel.replicate
.. py:module:: torch.nn.parallel.scatter_gather
.. py:module:: torch.nn.parameter
.. py:module:: torch.nn.utils.clip_grad
.. py:module:: torch.nn.utils.convert_parameters
.. py:module:: torch.nn.utils.fusion
.. py:module:: torch.nn.utils.init
.. py:module:: torch.nn.utils.memory_format
.. py:module:: torch.nn.utils.parametrizations
.. py:module:: torch.nn.utils.parametrize
.. py:module:: torch.nn.utils.prune
.. py:module:: torch.nn.utils.rnn