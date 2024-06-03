# graph

- `graphnet.py`：神经网络的构造、训练、测试代码
- `graphdata.py`：数据预处理的相关代码
- `torchamg.py`：基于`pytorch`实现的迭代算法
- `main.py`：程序入口
- `TestPyAMG/`：用于测试`pyamg`的目录
- `TestSparseTensor/`：用于测试`sparse tensor`的目录

以上代码是其它项目的代码，还需要针对本项目进行修改。运行以上代码需要安装`pytorch geometric`



## 基于`pyamg`的测试程序

在`verify_by_pyamg.py`中使用`pyamg`中的两网格算法测试矩阵`p`的求解效率。为了提高直接法的运行效率，需要安装`umfpack`求解器，通过以下命令进行安装

```bash
conda install -c conda-forge suitesparse
conda install scikit-umfpack
```



可以修改`verify_by_pyamg.py`中磨光算法和粗网格上的求解算法，部分可选参数在注释行中。详细设置可参考 [pyamg.multilevel](https://pyamg.readthedocs.io/en/latest/generated/pyamg.multilevel.html#pyamg.multilevel.MultilevelSolver)，磨光算法和求解算法的选择没有完善的文档，只能在程序源码中寻找。
