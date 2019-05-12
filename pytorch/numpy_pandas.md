## 数据处理

+ Why numpy & pandas?

  + 底层是c语言，运行效率快

  - 矩阵运算

+ 安装

  + pip install numpy
  + pip install pandas

+ Numpy属性

  

  - ndim维度
  - shape行数和列数
  - size元素个数

  - ```
    import numpy as np
    array = np.array([1,2,3],[2,3,4])
    print(array)
    
    
    ```

  - Numpy创建array

    - array 创建数组

    - Dtype 指定数据类型

    - zeros穿件数据全为0

    - ones创建数据全为1

    - empty创建数据接近0

    - Arrange 按制定范围创建数据

    - linspace创建线段

    - ```
      a = np.array([2, 23, 4])
      print(a)
      a = np.array([2, 23, 4],dtype=np.int)
      print(a.dtype)
      
      a = np.array([2, 23, 4],dtype=np.int32)
      a = np.array([2,23,4],dtype=np.float)
       
      a = np.arange(10, 20, 2)
      # 10-19步长为2
      a = np.arange(12).reshape((3, 4))
      
      ```

+ Numpy的基本运算

  ```
  import numpy as np
  a = np.array([10, 20, 30, 40])
  b = np.arange(4)
  c = a - b
  c = a + b
  c = a * b
  c = a ** b
  c = 10 * np.sin(a)
  #矩阵相乘
  c_dot = np.dot(a, b)
  
  ```

  - **axis**

    axis = 0 以列作为查找单元

    axis = 1以行作为查找单元

  - argmin()、argmax()

    两个函数分别对应着求矩阵中最小元素和最大元素的索引。

  - mean() 求均值 average



### pandas

+ Numpy是列表形式的 、没有数值标签，Pandas是字典形式。Pandas是基于Numpy构建的。

+ Series

  `Series`的字符串表现形式为：索引在左边，值在右边。由于我们没有为数据指定索引。于是会自动创建一个0到N-1（N为长度）的整数型索引。

+ DataFrame

  

















































































