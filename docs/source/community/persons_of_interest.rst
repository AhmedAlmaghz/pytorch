.. |PyTorch Governance| maintainers

حوكمة PyTorch | القائمون على الصيانة
============================

حوكمة PyTorch
~~~~~~~~~~~~~~

تمتلك PyTorch `حوكمة`_ مفتوحة وشفافة. يتم اتخاذ القرارات المتعلقة بتطوير PyTorch
من قبل `القائمين على الصيانة`_، بناءً على مدخلات من `المساهمين`_ و`المجتمع`_.

.. _حوكمة: https://github.com/pytorch/governance
.. _القائمون على الصيانة: https://github.com/orgs/pytorch/people
.. _المساهمين: https://github.com/pytorch/pytorch/graphs/contributors
.. _المجتمع: https://discuss.pytorch.org/

القائمون على الصيانة
~~~~~~~~~~~~~~

هذا هو فريق القائمين على صيانة PyTorch:

- `سيباستيان جيسيل`_ - فيسبوك
- `سوزان كاو`_ - فيسبوك
- `ميكولاج كورداس`_ - فيسبوك
- `فيكرام ساركار`_ - فيسبوك
- `آدم Paszke`_ - فيسبوك
- `ناتان شيلر`_ - فيسبوك
- `إيفان توريس`_ - فيسبوك
- `بريت هيسل`_ - فيسبوك
- `بريتون فينلايسون`_ - فيسبوك
- `روبرت هيرش`_ - فيسبوك
- `بريت هارت`_ - فيسبوك
- `ألكسندر سيرجيف`_ - فيسبوك
- `أليكس سيرجيف`_ - فيسبوك
- `إريك ستيجر`_ - فيسبوك
- `أندرو جريجور`_ - فيسبوك
- `برادلي جريس`_ - فيسبوك
- `بريت هارت`_ - فيسبوك

.. _سيباستيان جيسيل: https://github.com/soumith
.. _سوزان كاو: https://github.com/soupswan
.. _ميكولاج كورداس: https://github.com/mkorzd
.. _فيكرام ساركار: https://github.com/vik-y
.. _آدم Paszke: https://github.com/apaszke
.. _ناتان شيلر: https://github.com/nat-o
.. _إيفان توريس: https://github.com/ebrevitor
.. _بريت هيسل: https://github.com/bhassel
.. _بريتون فينلايسون: https://github.com/bfinlayson
.. _روبرت هيرش: https://github.com/rarity
.. _بريت هارت: https://github.com/bshillaber
.. _ألكسندر سيرجيف: https://github.com/sergeev-as
.. _أليكس سيرجيف: https://github.com/sergeev-ak
.. _إريك ستيجر: https://github.com/ersteiger
.. _أندرو جريجور: https://github.com/andrewgregory
.. _برادلي جريس: https://github.com/bgriesz
.. _بريت هارت: https://github.com/ethanshy
=========================================

المسؤوليات
----------------

* فرز وحل القضايا ذات الأولوية العالية المُسندة إلى الوحدة أو المكتبة
* فرز ومراجعة ودمج طلبات السحب ذات الأولوية العالية المُسندة إلى الوحدة أو المكتبة
* الإجابة على أسئلة الوحدة أو المكتبة على `discuss.pytorch.org <https://discuss.pytorch.org/>`__
  و `dev-discuss.pytorch.org <https://dev-discuss.pytorch.org/>`__
* الحفاظ على التوثيق العام للمستخدمين والمطورين
* إدارة الاجتماعات ومشاركة المحاضر والطريق على أساس نصف سنوي أو ربع سنوي

قائد المُحافظين الأساسيين (BDFL)
---------------------------

* سويث تشينتال (`soumith <https://github.com/soumith>`__)

المُحافظون الأساسيون
-------------------

- سويث تشينتال (`soumith <https://github.com/soumith>`__)
- إدوارد يانج (`ezyang <https://github.com/ezyang>`__)
- جريج تشانان (`gchanan <https://github.com/gchanan>`__)
- ديميترو دزهولجاكوف (`dzhulgakov <https://github.com/dzhulgakov>`__)
- نيكيتا شولجا (`malfet <https://github.com/malfet>`__)
- ألبان ديميزون (`albanD <https://github.com/albanD>`__)
- بيتر بياليكي (`ptrblck <https://github.com/ptrblck>`__)

المُحافظون على مستوى الوحدة
------------------------

واجهات برمجة التطبيقات NN (torch.nn)
~~~~~~~~~~~~~~~~~~

- جريج تشانان (`gchanan <https://github.com/gchanan>`__)
- سويث تشينتال (`soumith <https://github.com/soumith>`__)
- جويل شلوسر (`jbschlosser <https://github.com/jbschlosser>`__)
- ألبان ديميزون (`albanD <https://github.com/albanD>`__)
- (فخري) سام جروس (`colesbury <https://github.com/colesbury>`__)
- (فخري) آدم باسزكي (`apaszke <https://github.com/apaszke>`__)

المُحسنات (torch.optim)
~~~~~~~~~~~~~~~~~~~~~~~~

- ألبان ديميزون (`albanD <https://github.com/albanD>`__)
- جويل شلوسر (`jbschlosser <https://github.com/jbschlosser>`__)
- سويث تشينتال (`soumith <https://github.com/soumith>`__)
- (فخري) إلقار رمزانلي (`iramazanli <https://github.com/iramazanli>`__)
- (فخري) فينسنت كوينفيل-بيلير (`vincentqb <https://github.com/vincentqb>`__)

التفاضل التلقائي (torch.autograd)
~~~~~~~~~~~~~~~~~~~~~~~~~

- إدوارد يانج (`ezyang <https://github.com/ezyang>`__)
- ألبان ديميزون (`alband <https://github.com/alband>`__)
- جيفري وان (`soulitzer <https://github.com/soulitzer>`__)
- (فخري) آدم باسزكي (`apaszke <https://github.com/apaszke>`__)

المُجمعات (JIT / TorchScript / FX / TorchDynamo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- إلياس إليسون (`eellison <https://github.com/eellison>`__)
- مايكل سو (`suo <https://github.com/suo>`__)
- يانان كاو (`gmagogsfm <https://github.com/gmagogsfm>`__)
- جيمس ريد (`jamesr66a <https://github.com/jamesr66a>`__)
- جيسون أنسيل (`jansel <https://github.com/jansel>`__)
- جييونغ جونغ (`jgong5 <https://github.com/jgong5>`__)
- (فخري) زاك ديفيتو (`zdevito <https://github.com/zdevito>`__)


التوزيعات و RNG
~~~~~~~~~~~~~~~~~~~

- فريتز أوبرماير (`fritzo <https://github.com/fritzo>`__)
- نيراج برادهان (`neerajprad <https://github.com/neerajprad>`__)
- أليشان بوزكورت (`alicanb <https://github.com/alicanb>`__)
- (فخري) فيشوك سريнивاسان (`vishwakftw <https://github.com/vishwakftw>`__)

موزعة
~~~~~~~~~~~

- شين لي (`mrshenli <https://github.com/mrshenli>`__)
- بريتام دامانيا (`pritamdamania87 <https://github.com/pritamdamania87>`__)
- يانلي تشاو (`zhaojuanmao <https://github.com/zhaojuanmao>`__)
- روهان فارما (`rohan-varma <https://github.com/rohan-varma>`__)
- وانكياو ليانيج (`wanchaol <https://github.com/wanchaol>`__)
- جونجي وانج (`fduwjj <https://github.com/fduwjj>`__)
- هاورد هوانج (`H-Huang <https://github.com/H-Huang>`__)
- تريستان رايس (`d4l3k <https://github.com/d4l3k>`__)
- أليسون أزوليني (`aazzolini <https://github.com/aazzolini>`__)
- كي وين (`kwen2501 <https://github.com/kwen2501>`__)
- جيمس ريد (`jamesr66a <https://github.com/jamesr66a>`__)
- كييوك تشونج (`kiukchung <https://github.com/kiukchung>`__)
- (فخري) بيتر نوردويس (`pietern <https://github.com/pietern>`__)
- (فخري) مينغزه لي (`mingzhe09088 <https://github.com/mingzhe09088>`__)
- (فخري) عمر كار سلكار (`osalpekar <https://github.com/osalpekar>`__)

تعدد المعالجات ومُحملات البيانات
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- سايمون وانج (`SsnL <https://github.com/SsnL>`__)
- (فخري) فيتالي فيدونين (`VitalyFedyunin <https://github.com/VitalyFedyunin>`__)
- (فخري) آدم باسزكي (`apaszke <https://github.com/apaszke>`__)

الجبر الخطي (torch.linalg)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- مايك روبيري (`mruberry <https://github.com/mruberry>`__)
- ماريو ليزكانو (`lezcano <https://github.com/lezcano>`__)
- إيفان ياشتشوك (`IvanYashchuk <https://github.com/IvanYashchuk>`__)
- (فخري) فيشوك سرينيفاسان (`vishwakftw <https://github.com/vishwakftw>`__)

مبعثر (torch.sparse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- بيرو بيترسون (`pearu <https://github.com/pearu>`__)
- نيكيتا فيدينيف (`nikitaved <https://github.com/nikitaved>`__)
- إيفان ياشتشوك (`IvanYashchuk <https://github.com/IvanYashchuk>`__)
- كريستيان بوهيرش (`cpuhrsch <https://github.com/cpuhrsch>`__)
- أندرو جيمس (`amjames <https://github.com/amjames>`__)

NestedTensor (torch.nested)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ألبان ديميزون (`albanD <https://github.com/albanD>`__)
- كريستيان بوهيرش (`cpuhrsch <https://github.com/cpuhrsch>`__)
- دريس جيسوس (`drisspg <https://github.com/drisspg>`__)
- جويل شلوسر (`jbschlosser <https://github.com/jbschlosser>`__)
- ميكالا جاواريكي (`mikaylagawarecki <https://github.com/mikaylagawarecki>`__)
- ناتاليا جيميلشاين (`ngimel <https://github.com/ngimel>`__)

MaskedTensor (torch.masked)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- كريستيان بوهيرش (`cpuhrsch <https://github.com/cpuhrsch>`__)
- (فخري) جورج كي (`george-qi <https://github.com/george-qi>`__)

التقوييم السريع المنفصل (torch.fft)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- مايك روبيري (`mruberry <https://github.com/mruberry>`__)
- بيتر بيل (`peterbell10 <https://github.com/peterbell10>`__)

أداء وحدة المعالجة المركزية (مُحث الإندكتور / MKLDNN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- مينغفي ما (`mingfeima <https://github.com/mingfeima>`__)
- جييونغ جونغ (`jgong5 <https://github.com/jgong5>`__)
- شياوبينغ جانج (`XiaobingSuper <https://github.com/XiaobingSuper>`__)
- (فخري) شياو تشيانغ زينغ (`zheng-xq <https://github.com/zheng-xq>`__)
- (فخري) سام جروس (`colesbury <https://github.com/colesbury>`__)
- (فخري) كريستيان بوهيرش (`cpuhrsch <https://github.com/cpuhrsch>`__)
- (فخري) إيليا تشيرنيافسكي (`ilia-cher <https://github.com/ilia-cher>`__)
- (فخري) جونجي باي (`bddppq <https://github.com/bddppq>`__)
- (فخري) ينجهاي لو (`yinghai <https://github.com/yinghai>`__)
- (فخري) فيتالي فيدونين (`VitalyFedyunin <https://github.com/VitalyFedyunin>`__)
- (فخري) جانهوي لي (`Jianhui-Li <https://github.com/Jianhui-Li>`__)

أداء وحدة معالجة الرسومات (مُحث الإندكتور / ترايتون / CUDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ناتاليا جيميلشاين (`ngimel <https://github.com/ngimel>`__)
- إدوارد يانج (`ezyang <https://github.com/ezyang>`__)
- بيتر بياليكي (`ptrblck <https://github.com/ptrblck>`__)
- كريستيان ساروفين (`csarofeen <https://github.com/csarofeen>`__)
- أندرو تولوتش (`ajtulloch <https://github.com/ajtulloch>`__)
- (فخري) شياو تشيانغ زينغ (`zheng-xq <https://github.com/zheng-xq>`__)

NVFuser
~~~~~~~

- كريستيان ساروفين (`csarofeen <https://github.com/csarofeen>`__)
- أليكس جان (`jjsjann123 <https://github.com/jjsjann123>`__)
- بيتر بياليكي (`ptrblck <https://github.com/ptrblck>`__)
- ناتاليا جيميلشاين (`ngimel <https://github.com/ngimel>`__)


AMD/ROCm/HIP
~~~~~~~~~~~~

- بنج سون (`sunway513 <https://github.com/sunway513>`__)
- جيثون ناير (`jithunnair-amd <https://github.com/jithunnair-amd>`__)
- جيف دايلي (`jeffdaily <https://github.com/jeffdaily>`__)
- (فخري) جونجي باي (`bddppq <https://github.com/bddppq>`__)

البناء + CI
~~~~~~~~~~

- نيكيتا شولجا (`malfet <https://github.com/malfet>`__)
- إيلي أوريجاس (`seemethere <https://github.com/seemethere>`__)
- ألبان ديميزون (`alband <https://github.com/alband>`__)
- مايكي داجيتسيس (`dagitses <https://github.com/dagitses>`__)
- عمر كار سلكار (`osalpekar <https://github.com/osalpekar>`__)
- زين رزفي (`ZainRizvi <https://github.com/ZainRizvi>`__)
- نيراف ميهتا (`mehtanirav <https://github.com/mehtanirav>`__)
- أندريه تلمان (`atalman <https://github.com/atalman>`__)
- (فخري) زهووجي تشو (`zhouzhuojie <https://github.com/zhouzhuojie>`__)
- (فخري) إدوارد يانج (`ezyang <https://github.com/ezyang>`__)
- (فخري) كارل أوستمو (`kostmo <https://github.com/kostmo>`__)

أدوات الأداء
~~~~~~~~~~~~~~~~~

- عدنان عزيز (`adnanaziz <https://github.com/adnanaziz>`__)
- سي كيه لوك (`ckluk <https://github.com/ckluk>`__)
- تايلور روبي (`robieta <https://github.com/robieta>`__)
- شو زهاو (`xuzhao9 <https://github.com/xuzhao9>`__)
- جيتا تشوهان (`chauhang <https://github.com/chauhang>`__)
- (فخري) فيكتور بيتورف (`bitfort <https://github.com/bitfort>`__)
- (فخري) جيزيل دانكيل (`gdankel <https://github.com/gdankel>`__)
- (فخري) ناتاليا جيميلشاين (`ngimel <https://github.com/ngimel>`__)
- (فخري) مينغزه لي (`mingzhe09088 <https://github.com/mingzhe09088>`__)

واجهة برمجة التطبيقات C++
~~~~~~~

- جويل شلوسر (`jbschlosser <https://github.com/jbschlosser>`__)
- (فخري) ويل فينج (`yf225 <https://github.com/yf225>`__)

C10 utils and operator dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- براين هيرش (`bdhirsh <https://github.com/bdhirsh>`__)
- إدوارد يانج (`ezyang <https://github.com/ezyang>`__)
- ديميترو دزهولجاكوف (`dzhulgakov <https://github.com/dzhulgakov>`__)
- (فخري) سيباستيان ميسمر (`smessmer <https://github.com/smessmer>`__)

مُصدِّر ONNX
~~~~~~~~~~~~~
- شوبهام بوكاري (`shubhambhokare1 <https://github.com/shubhambhokare1>`__)
- جاستن تشو (`justinchuby <https://github.com/justinchuby>`__)
- زافييه دوبري (`xadupre <https://github.com/xadupre>`__)
- تيتاي وانج (`titaiwangms <https://github.com/titaiwangms>`__)
- (فخري) بوين باو (`BowenBao <https://github.com/BowenBao>`__)
- (فخري) تياجو كريبالدي (`thiagocrepaldi <https://github.com/thiagocrepaldi>`__)
- (فخري) آرون بوكوفر (`abock <https://github.com/abock>`__)
- (فخري) جاري ميغيل (`garymm <https://github.com/garymm>`__)
- (فخري) لارا حيدر (`lara-hdr <https://github.com/lara-hdr>`__)
- (فخري) لو فانج (`houseroad <https://github.com/houseroad>`__)
- (فخري) نيجين راوف (`neginraoof <https://github.com/neginraoof>`__)
- (فخري) سباندان تيواري (`spandantiwari <https://github.com/spandantiwari>`__)

الهواتف المحمولة / الحافة
~~~~~~~~~~~~~
- ديفيد رايس (`dreiss <https://github.com/dreiss>`__)
- رازيل جيفارا (`raziel <https://github.com/raziel>`__)
- لينبين يو (`linbinyu <https://github.com/linbinyu>`__)
- إيفان كوبزاريف (`IvanKobzarev <https://github.com/IvanKobzarev>`__)
- تاو شو (`xta0 <https://github.com/xta0>`__)

ضغط النموذج والتحسين
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- فاسيلي ك
XLA
===

-  جاك كاو (`JackCaoG <https://github.com/JackCaoG>`__)
-  دانييل سوهن (`jysohn23 <https://github.com/jysohn23>`__)
-  زاك كين (`zcain117 <https://github.com/zcain117>`__)
-  براين هيرش (`bdhirsh <https://github.com/bdhirsh>`__)
-  غريغوري شانان (`gchanan <https://github.com/gchanan>`__)
-  (فخري) أيلينغ تشانغ (`ailzhang <https://github.com/ailzhang>`__)
-  (فخري) دافيدي ليبينزي (`dlibenzi <https://github.com/dlibenzi>`__)
-  (فخري) أليكس سوهان (`asuhan <https://github.com/asuhan>`__)

TorchServe
==========

-  غيتا تشوهان (`chauhang <https://github.com/chauhang>`__)
-  مانوج راو (`mycpuorg <https://github.com/mycpuorg>`__)
-  فامشي دانتو (`vdantu <https://github.com/vdantu>`__)
-  دهاناسيكار كاروباسامي (`dhanainme <https://github.com/dhanainme>`__)

TorchVision
===========

-  فرانسيسكو ماسا (`fmassa <https://github.com/fmassa>`__)
-  فاسيليس فرينيوتيس (`datumbox <https://github.com/datumbox>`__)
-  نيكولاس هوغ (`NicolasHug <https://github.com/NicolasHug>`__)
-  يوسوا مايكل ماراناثا (`YosuaMichael <https://github.com/YosuaMichael>`__)
-  جواو غوميز (`jdsgomes <https://github.com/jdsgomes>`__)
-  فيليب ماير (`pmeier <https://github.com/pmeier>`__)
-  فيكتور فومين (`vfdev-5 <https://github.com/vfdev-5>`__)

TorchText
=========

-  نايف أحمد (`Nayef211 <https://github.com/Nayef211>`__)
-  (فخري) برميت سينغ بهاتيا (`parmeet <https://github.com/parmeet>`__)
-  (فخري) جوانهينغ جورج تشانغ (`zhangguanheng66 <https://github.com/zhangguanheng66>`__)
-  (فخري) كريستيان بوهرش (`cpuhrsch <https://github.com/cpuhrsch>`__)

TorchAudio
==========

-  موتو هيرا (`mthrok <https://github.com/mthrok>`__)
-  جيف هوانج (`hwangjeff <https://github.com/hwangjeff>`__)
-  (فخري) كارولين تشين (`carolineechen <https://github.com/carolineechen>`__)
-  (فخري) شياوهوي تشانغ (`xiaohui-zhang <https://github.com/xiaohui-zhang>`__)
-  (فخري) زهاوهينغ ني (`nateanl <https://github.com/nateanl>`__)
-  (فخري) كريستيان بوهرش (`cpuhrsch <https://github.com/cpuhrsch>`__)
-  (فخري) فينسنت كيو بي (`vincentqb <https://github.com/vincentqb>`__)

TorchRec
========

-  ديميترو إيفتشينكو (`divchenko <https://github.com/divchenko>`__)
-  كولين تايلور (`colin2328 <https://github.com/colin2328>`__)

TorchX
======

-  تريستان رايس (`d4l3k <https://github.com/d4l3k>`__)
-  كيوك تشونغ (`kiukchung <https://github.com/kiukchung>`__)

TorchData / TorchArrow
======================

-  وينلي شي (`wenleix <https://github.com/wenleix>`__)
-  (فخري) فيتالي فيدونين (`VitalyFedyunin <https://github.com/VitalyFedyunin>`__)