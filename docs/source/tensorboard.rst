torch.utils.tensorboard
===========================

.. automodule:: torch.utils.tensorboard

قبل الاسترسال أكثر، يمكن إيجاد المزيد من التفاصيل حول TensorBoard على الرابط التالي:
https://www.tensorflow.org/tensorboard/

بمجرد تثبيت TensorBoard، تسمح هذه الأدوات المساعدة بتسجيل نماذج ومعايير PyTorch في دليل للتصور ضمن واجهة مستخدم TensorBoard.
يتم دعم المخططات البيانية والصور والرسوم البيانية والتعبئات والرسوم البيانية والتصورات المضمنة لنماذج PyTorch وtensors، بالإضافة إلى شبكات Caffe2 وblobs.

تعد فئة SummaryWriter المدخل الرئيسي الخاص بك لتسجيل البيانات لاستهلاكها وتصورها بواسطة TensorBoard. على سبيل المثال:

.. code:: python

    import torch
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms

    # سيقوم الكاتب بالإخراج إلى دليل ./runs/ بشكل افتراضي
    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # قم بتعديل نموذج ResNet لأخذ تدرج الرمادي بدلاً من RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()

بعد ذلك، يمكن تصور هذا باستخدام TensorBoard، والذي يجب أن يكون قابلاً للتثبيت والتشغيل باستخدام::

    pip install tensorboard
    tensorboard --logdir=runs

يمكن تسجيل الكثير من المعلومات لتجربة واحدة. لتجنب الفوضى في واجهة المستخدم والحصول على نتائج أفضل للتجمع، يمكننا تجميع المخططات بتسميتها بشكل هرمي. على سبيل المثال، سيتم تجميع "Loss/train" و "Loss/test" معًا، في حين سيتم تجميع "Accuracy/train" و "Accuracy/test" بشكل منفصل في واجهة TensorBoard.

.. code:: python

    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

    writer = SummaryWriter()

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writerMultiplier.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


النتيجة المتوقعة:

.. image:: _static/img/tensorboard/hier_tags_ar.png
    :scale: 75 %

|
|

.. currentmodule:: torch.utils.tensorboard.writer

.. autoclass:: SummaryWriter

   .. automethod:: __init__
   .. automethod:: add_scalar
   .. automethod:: add_scalars
   .. automethod:: add_histogram
   .. automethod:: add_image
   .. automethod:: add_images
   .. automethod:: add_figure
   .. automethod:: add_video
   .. automethod:: add_audio
   .. automethod:: add_text
   .. automethod:: add_graph
   .. automethod:: add_embedding
   .. automethod:: add_pr_curve
   .. automethod:: add_custom_scalars
   .. automethod:: add_mesh
   .. automethod:: add_hparams
   .. automethod:: flush
   .. automethod:: close