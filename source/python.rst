python
=============


standard library
---------------------

当前文件路径
................


.. code-block:: python

    import os
    source_dir = os.path.split(os.path.realpath('__file__'))[0]


    # os.path.split(os.path.realpath(__file__))得到的是一个tuple
    # 第一个位置是当前文件所在路径，第二个位置是当前文件名
    ('C:\\Users\\luoxb\\PycharmProjects\\spark', 'main.py')

    source_dir = os.path.abspath('.')
    'C:\\Users\\luoxb\\PycharmProjects\\spark'


当前文件夹下面是否有指定文件
..................................


::

    file_name = "community.csv"
    os.path.isfile("./" + file_name)



pandas
-------------------


我这里是一个个 链接_.

.. _链接: https://cn.bing.com/?mkt=zh-CN&mkt=zh-CN


numpy
------------------