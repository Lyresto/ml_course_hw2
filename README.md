- [exp](./exp)： 模型权重文件和训练参数保存路径
    - *.pth：权重文件
    - model_config.json：模型超参数
    - training_config.json：训练超参数
- [result](./result/test)：结果文件
- [runs](./runs)：tensorboard运行文件，可查看每次运行的dice系数变化
- [config.py](./config.py)：训练和模型参数
- [dataset.py](./dataset.py)：数据集处理与加载
- [trainer.py](./trainer.py)：模型训练
- [unet.py](./unet.py)：模型定义
- [util.py](./util.py)：工具