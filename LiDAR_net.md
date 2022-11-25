<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [点云物体检测](#点云物体检测)
  - [萌芽期（～2017）](#萌芽期~2017)
  - [起步期（2017）](#起步期2017)
    - [VoxelNet](#voxelnet)
    - [PointNet](#pointnet)
    - [PointNet++](#pointnet-1)
  - [发展期（2018～2020）](#发展期2018~2020)
    - [Point-RCNN](#point-rcnn)
    - [3DSSD](#3dssd)

<!-- /code_chunk_output -->

# 点云物体检测

## 萌芽期（～2017）

将 3D 点云转换为 2D 图像格式，就可以套用图像的物体检测网络。

__VeloFCN__

将 3D 点云映射到 2D 平面，输入 R-CNN 网络。

__MV3D__

将 3D 点云分别映射到正视图和俯视图，并融合 2D 图像，输入 R-CNN 网络。
（映射到俯视图损失点云的高度信息，而高度信息相对不太重要。）

## 起步期（2017）

出现了两个里程碑式的工作：VoxelNet 和 PointNet++。之后点云物体检测领域几乎所有的方法都离不开这两个工作。

### VoxelNet

<img src="net_img/voxelnet_2.png" width=100%>

__Feature Learning Network__

1. Grouping
将空间分割为体素地图，每个体素内点云密度不同。
2. Random Samping
<font color=OrangeRed>由于：原始点云数量过多，不适合直接投入网络；每个体素内点云密度不同。所以：要随机采样。</font>
在每个体素内随机采样 T 个点（不足的就重复）。
3. Stacked Voxel Feature Encoding
<img src="net_img/voxelnet_3.png" width=45%>
每个采样点用 7 维特征表示，包括该点的 X,Y,Z 坐标，反射强度 R，以及相对体素质心（体素内所有点位置的均值）的位置差 ΔX,ΔY,ΔZ。<font color=OrangeRed>个人认为：ΔX,ΔY,ΔZ 用来编码体素内（局部）的 surface 信息。</font>
Fully Connected Neural Net 分别将每个采样点的特征从 7 维扩展到 m 维。
每个采样点的特征再与最大池化特征进行拼接，得到新的特征向量。
<font color=OrangeRed>此过程可以重复多次，以增强特征向量对本体素 surface 的描述能力。</font>
最终融合体素内所有采样点的特征，得到一个固定长度的特征向量。
4. Sparse 4D Tensor
最终输出 4D Tensor。

__Convolutional Middle Layer__

采用 3D 卷积对 Z 维度进行压缩（令 stride=2）。假设 4D Tensor 的维度为 HxWxDxC，经过若干次 3D 卷积后，被压缩为 HxWx2xC'，再化为 3D Tensor HxWx2C'，类似于鸟瞰图。这样就可以使用图像物体检测网络（比如 Region Proposal Network）来生成物体检测框。

__Region Proposal Network__

<img src="net_img/voxelnet_4.png" width=100%>

1. Regression map
ground truth box 可以用一个 7 维向量表示 $(x_c^g,y_c^g,z_c^g,l^g,w^g,h^g,\theta^g)$。
文中预定义了两个 anchor，Regression map 中的 14 维向量，就是在学习从预定义 anchor 到 ground truth box 的变换 $(\Delta x,\Delta y,\Delta z,\Delta l,\Delta w,\Delta h,\Delta \theta)$。
2. Probability score map
两个 anchor 的置信度。
3. Loss Function
<img src="net_img/voxelnet_eq2.png" width=45%>

__不足__

Convolutional middle layer 的 3D 巻积速度非常慢。

### PointNet

网络的输入直接是点云数据。

__点云数据的特征__

1. 网络模型需要：对输入的点云数据的不同排列保持不变性。
2. 邻域点云包含有意义的局部信息。
3. 网络模型需要：对输入的点云数据的旋转平移变换保持不变性。

__网络结构__

<img src="net_img/point_2.png" width=100%>

1. Symmetry Function for Unordered Input
红框部分，使用 max pool 这个对称函数来提取点云数据的特征，以此保证提取出来的 global feature 不随点云排序的改变而改变。
2. Local and Global Information Aggregation
绿框部分，将局部特征和全局特征进行串联。
<font color=OrangeRed>注意：局部特征没有用到邻域信息，这个问题在 PointNet++ 中做了完善。</font>
3. Joint Alignment Network
蓝框部分，在进行特征提取之前，使用 T-Net 网络对点云数据/特征进行对齐，以此保证旋转平移不变性。
输入的点云经过 T-Net 会获得一个 3x3 的仿射变换矩阵，用这个矩阵与原始点云相乘得到新的 nx3 点云；同样的方法可以应用到特征空间，但是在特征空间中仿射变换矩阵太大了，所以通过在损失函数中增加正则化相，以使该矩阵尽量为正交矩阵。

__定理__

<img src="net_img/point_the1.png" width=45%>
<img src="net_img/point_the2.png" width=45%>

1. Universal approximation
定理 1 说明红框部分的网络结构能够拟合任意的连续集合函数。
2. Bottleneck dimension and stability
定理 2(a) 说明对于任何输入数据集 S，都存在一个最小集 Cs 和一个最大集 Ns，使得对 Cs 和 Ns 之间的任意集合 T，其输出等于 S。换句话说，模型对输入数据在有噪声和有数据损坏的情况下都是鲁棒的。
定理 2(b) 说明 Cs (critical point set) 的尺寸上界由 K (bottleneck dimension) 给出。

### PointNet++

PointNet 提取的全局特征能够很好地完成分类任务，但其局部特征提取能力较差，很难对复杂场景进行分析。PointNet++ 提出了多尺度特征提取结构，能够有效提取局部特征。 

<img src="net_img/pointpp_2.png" width=100%>

__Hierarchical Point Set Feature Learning__

1. Sampling layer
使用 farthest point sampling (FPS) 算法，对点云进行采样。
2. Grouping layer
输入：大小为 (N,d+C) （d 为坐标，C 为特征）的点云数据；大小为 (N',d) 的局部区域中心数据。
输出：大小为 (N',K,d+C) （K 为局部区域中心的邻域点个数）的点云数据。
3. PointNet layer
使用 pointnet 提取局部区域特征。
输出：大小为 (N‘,d+C’) 的点云数据。

__Robust Feature Learning under Non-Uniform Sampling Density__

<font color=OrangeRed>点云数据往往是密度不均的，如何用一个网络同时应对稠密点云与稀疏点云？</font>
解决方法：在 pointnet 处搭配 MRG。

<img src="net_img/pointpp_3.png" width=30%>

1. Multi-scale grouping
不同尺度的 grouping 接 pointnet。计算量巨大。
2. Multi-resolution grouping
每一层的局部特征都由两个向量串联构成：对上一层点云数据作 set abstraction 而来的特征向量，对原始点云数据作 pointnet 而来的特征向量。

__Point Feature Propagation for Set Segmentation__

对于 Segmentation 任务，还需要将下采样后的特征进行上采样。插值、跳层连接、unit pointnet。

## 发展期（2018～2020）

__SECOND__

对 VoxelNet 的改进。采用稀疏卷积策略，避免了空白区域的无效计算，提升运行速度。

__PointPillar__

对 VoxelNet 的改进。将 3D 点云量化到 2D 平面，把落到每个 2D 网格内的点直接叠放在一起，形象的称其为 Pillar。

### Point-RCNN

对 PointNet++ 的改进，两阶段检测器。首先，用 PointNet++ 提取点云特征，进行前景分割；每个前景点会输出一个 3D Bounding Box；再对 3D Bounding Box 内的点云作进一步的分析。

<img src="net_img/pointrcnn_2.png" width=100%>

__Bottom-up 3D proposal generation via point cloud segmentation__

1. Learning point cloud representations
使用 PointNet++ 和 multi-scale grouping 来提取点云特征。
2. Foreground point segmentation
前景点分割。这步操作与 3D Box Generation 同步进行。
3. Bin-based 3D Box Generation
VoxelNet 中设置 anchor box 的方法计算量巨大。本文直接由逐个 foreground point 估计 3D box。

__Point cloud region pooling__

综合 3D box 内点的：原始数据、特征向量、foreground mask，精调 3D box。

__Canonical 3D bounding box refinement__

1. Canonical transformation
将 3D box 内的点云转换到其自身坐标系，以便下一阶段的局部特征学习。
2. Feature learning for box proposal refinement
3. Losses for box proposal refinement



### 3DSSD






PV-CNN 
Fast Point RCNN
PV-RCNN
SA-SSD
