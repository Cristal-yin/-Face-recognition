# Matplotlib使用

### 为什么选择matplotlib

+ 是一个很强大的python画图工具
+ 太多的数据不知道如何呈现

### 安装

```
pip install matplotlib
```

### 基本用法

```
import matplotlib.pyplot as plt
import numpy as np
# np.linsapce 定义x :范围是(-1, 1) 个数是50个
x = np.linspace(-1, 1, 50)
y = 2 * x + 1


# plt.figure定义一个图像窗口， 使用plt.plot画(x, y)曲线，使用plt.show显示图像
plt.figure()
plt.plot(x, y)
plt.show()
```

### figure图像

