# **M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models**的复现工作

公共数据集 [CQ500](https://paperswithcode.com/dataset/cq500)和[PhysioNet](https://physionet.org/content/ct-ich/1.3.1/)数据集

## CQ500数据分析和问题构建

这个数据集包含了多个指标：
intracerebral hemorrhage、intraparenchymal hemorrhage、intraventricular hemorrhage、subdural hemorrhage、epidural hemorrhage、subarachnoid hemorrhage、Bleed Location-Left、Bleed Location-Right、chronic bleed?、fracture、calvarial fracture、mass effect、midline shift

可以划分为：

1. 出血类型：

> + **脑内出血（Intracerebral Hemorrhage）**: 指在脑组织内发生的出血。 
>
> + **脑室出血（Intraventricular Hemorrhage）**: 指出血发生在脑室内。
>
> + **硬膜下出血（Subdural Hemorrhage）**: 指发生在硬膜下腔的出血。
>
> + **硬膜外出血（Epidural Hemorrhage）**: 指发生在硬膜外腔的出血。
>
> + **蛛网膜下腔出血（Subarachnoid Hemorrhage）**: 指发生在蛛网膜下腔的出血。

2. 出血位置：

> - **左侧出血（Bleed Location-Left）**: 出血发生在大脑左侧。
> - **右侧出血（Bleed Location-Right）**: 出血发生在大脑右侧。

3. 其他相关指标：

> - **慢性出血（Chronic Bleed）**: 指是否存在慢性出血的情况。
> - **骨折（Fracture）**: 可能与外伤有关，需考虑是否影响脑部情况。
> - **颅骨骨折（Calvarial Fracture）**: 特指颅骨的骨折情况。
> - **肿块效应（Mass Effect）**: 由于肿块或积液引起的对周围组织的压迫。
> - **中线偏移（Midline Shift）**: 脑部结构因肿胀或积液而导致的中线位置变化。

设计 1个开放式问题：

+ **患者是否出现脑出血？如果出现，具体是哪种类型（如脑内出血、脑室出血）以及发生在哪个位置（左侧、右侧或双侧等）？**

其余的指标设计成5个Closed-ended的问题：

+ 患者是否存在慢性出血？
+ 患者是否患有骨折？
+ 患者是否有颅骨骨折？
+ 患者是否存在肿块效应？
+ 患者是否存在中线偏移？

# PhysioNet数据分析与问题构建

数据标签文件（Patient demographics.csv）中标注了Intraventricular Hemorrhage、intraparenchymal hemorrhage、subarachnoid hemorrhage、epidural hemorrhage、subdural hemorrhage和fracture

可以划分为：

1. 出血类型：

> Intraventricular Hemorrhage:
>
> intraparenchymal hemorrhage:
>
> subarachnoid hemorrhage:
>
> epidural hemorrhage:
>
> subdural hemorrhage

2. 骨折：

Fracture

故而可以设计一个open-ended和一个closed-ended问题

+ 患者是否存在脑出血？如果存在，具体是哪种类型的脑出血？（**open-ended**）

+ 患者是否患有骨折？（**closed-ended**）