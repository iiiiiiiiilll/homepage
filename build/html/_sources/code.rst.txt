Python
======

::

   注意文件名和要导入的模块名不能一样

.. code:: python

   (1<2)>1.5 False
   1<2>1.5 True

当前文件路径
~~~~~~~~~~~~

.. code:: python

   import os
   source_dir = os.path.split(os.path.realpath('__file__'))[0]


   # os.path.split(os.path.realpath(__file__))得到的是一个tuple
   # 第一个位置是当前文件所在路径，第二个位置是当前文件名
   ('C:\\Users\\luoxb\\PycharmProjects\\spark', 'main.py')

   source_dir = os.path.abspath('.')
   'C:\\Users\\luoxb\\PycharmProjects\\spark'

当前文件夹下面是否有指定文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   file_name = "community.csv"
   os.path.isfile("./" + file_name)

pip下载指定清华源
~~~~~~~~~~~~~~~~~

::

   pip install lxml -i https://pypi.tuna.tsinghua.edu.cn/simple

字典求最大值键值对
~~~~~~~~~~~~~~~~~~

.. code:: python

   max(m.items(),key=lambda x:x[1])

字典最大值的key
~~~~~~~~~~~~~~~

.. code:: python

   max_key = max(dic, key = dic.get)

datetime / timedelta
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from datetime import datetime
   from datetime import timedelta
   time= datetime(2021, 10, 1)
   delta = time  - timedelta(days=140)
   time,delta

   (datetime.datetime(2021, 10, 1, 0, 0), datetime.datetime(2021, 5, 14, 0, 0))

   a = np.datetime64('2021-10-01')
   delta = np.timedelta64(140,'D')
   a-delta

   numpy.datetime64('2021-05-14')

itertools
~~~~~~~~~

.. code:: python

   >>> aa = ['a', 'b', 'c']
   >>> list(itertools.permutations(aa, 2))
   [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]


   >>> list(itertools.combinations(aa, 2))
   [('a', 'b'), ('a', 'c'), ('b', 'c')]


   import itertools
   first_letter = lambda x: x[0]
   names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']

   for letter, names in itertools.groupby(names, first_letter):
       print(letter, list(names)) # names is a generator

   A ['Alan', 'Adam']
   W ['Wes', 'Will']
   A ['Albert']
   S ['Steven']

笛卡尔积
~~~~~~~~

.. code:: python

   import itertools

   b = [1,0] 
   for i in itertools.product(b,b,b):
       print(i)

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211130101647105.png
   :alt: image-20211130101647105

   image-20211130101647105

集合运算
~~~~~~~~

.. code:: python

   交  
   a & b 
   a.intersection(b)

   并
   a | b
   a.union(b)


   差
   a - b
   a.difference(b)

isinstance
~~~~~~~~~~

.. code:: python

   a = 5
   isinstance(a,int)

   True

   isinstance(a,(int,float))
   True

查看对象所有方法与属性
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   help(x)
   # 返回所有注释

   dir(x)
   # 只返回名字

is / ==
~~~~~~~

.. code:: python

   is 判断是否指向同一个对象
   == 

   >>> a = [1,2,3]
   >>> b = a

   # a,b指向同一个对象

   >>> a is b
   True


   # 对a做了修改，b也会变
   >>> a.append(4)
   >>> a
   [1, 2, 3, 4]
   >>> b
   [1, 2, 3, 4]


   >>> c = list(a)   # 会创建新的对象
   >>> c is a 
   False
   >>> c == a 
   True

   # 作为list,每个位置上的元素相等，但是不是同一个对象


   # A very common use of is and is not is to check if a variable is None, since there is
   # only one instance of None:

   >>> a is None
   False

format
~~~~~~

.. code:: python

   template = '{0:.2f} {1:s} are worth US${2:d}'
   template.format(4.5560, 'Argentine Pesos', 1)

   '4.56 Argentine Pesos are worth US$1'

   seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
   for a, b, c in seq:
       print('a={0}, b={1}, c={2}'.format(a, b, c))
       

   a=1, b=2, c=3
   a=4, b=5, c=6
   a=7, b=8, c=9


   n = "123"
   n.zfill(5)
   '00123'

list 按另一个list排序
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   a = [np.random.randint(1,100) for i in range(5)]
   b = [np.random.randint(1,100) for i in range(5)]
   a,b

   ([59, 56, 25, 80, 6], [66, 11, 2, 24, 12])

   [x for x, _ in sorted(zip(a,b), key=lambda pair: pair[1])]
   [25, 56, 6, 80, 59]

速度解析
~~~~~~~~

.. code:: python

   # Note that list concatenation by addition is a comparatively expensive operation since
   # a new list must be created and the objects copied over. Using extend to append ele‐
   # ments to an existing list, especially if you are building up a large list, is usually pref‐
   # erable. Thus,

   everything = []
   for chunk in list_of_lists:
       everything.extend(chunk)
       
   # is faster than the concatenative alternative:

   everything = []
   for chunk in list_of_lists:
       everything = everything + chunk

try/raise
~~~~~~~~~

如果没有try,
raise有错误程序就会暂停，如果想让程序有错误仍能正常运行需要，需要用try
except 捕获异常

.. code:: python

   try:
       a = input("输入一个数：")
       #判断用户输入的是否为数字
       if(not a.isdigit()):
           raise ValueError("a 必须是数字")
   except ValueError as e:
       print("引发异常：",repr(e))
       
       

pandas
------

test_data
~~~~~~~~~

.. code:: python

   pd.DataFrame(np.random.randint(1,10,size=(8,3)))

打印显示列数
~~~~~~~~~~~~

.. code:: python

   pd.set_option('display.max_columns', None)

CSV读写
~~~~~~~

.. code:: python

   # 将dataframe转为csv保存到当前文件夹，自命名为file_name
   data.to_csv(source_dir + "\\" + file_name,
                   sep=',', index=False, header=True)

.. code:: python

   # 读取csv
   file_name = "游戏路径.csv"
   data = pd.read_csv(filepath_or_buffer=file_name, sep=',')


   # 指定读取列
   d = pd.read_csv('2.csv', usecols=['a', 'b'])

   # 只读前n行
   d = pd.read_csv('2.csv', usecols=['a', 'b'], nrows=10)

Excel 读写
~~~~~~~~~~

.. code:: python

    pd.read_excel('data.xlsx', sheet_name=3, engine='openpyxl',header=[0,1,2])
       
       
       
    pandas.read_excel(io, sheet_name=0, header=0, names=None, index_col=None, usecols=None, squeeze=False, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, parse_dates=False, date_parser=None, thousands=None, comment=None, skipfooter=0, convert_float=None, mangle_dupe_cols=True, storage_options=None)


   df.to_excel('file_name.xlsx')



   DataFrame.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None, storage_options=None)[source]

小数精度
~~~~~~~~

.. code:: python

   pd.set_option('precision', 9)

列转字典
~~~~~~~~

.. code:: python

   df[["a", "b"]].set_index("a").to_dict()["b"]


   a = pd.DataFrame(np.random.randint(1,100,size=[8,2]))
   a

       0   1
   0   84  70
   1   97  15
   2   76  78
   3   11  12
   4   62  86
   5   91  33
   6   40  38
   7   55  81

   {i:j for row,i,j in a.itertuples()}

   {84: 70, 97: 15, 76: 78, 11: 12, 62: 86, 91: 33, 40: 38, 55: 81}

值统计
~~~~~~

.. code:: python

   # 对一列元素计数，返回值及出现的次数
   data.label.value_counts()

   #统计各列nan的个数
   data.isna().sum()

.. code:: python

   data.label.describe()
   data.describe()
   data[[0,1,2]].describe()

布尔值筛选
~~~~~~~~~~

.. code:: python

   # 筛选满足条件的行 
   y[y['label']==1]

   # label 非nan的行
   data[ ~ pd.isna(data['label'])]

   # label 为nan的行
   data[pd.isna(data['label'])]  

   # 布尔值筛选
   # 不同的条件必须用()包裹起来,并或非分别使用&,|,~而非and,or,not
   # 不然会报错
   stripes_or_bars=flags[(flags['stripes']>=1) | (flags['bars']>=1)]

   # Series
   a = pd.Series([3, 1, 4, 5, 0, 2, 6, 7])
   b = pd.Series([1, 3, 1, 3, 3, 3, 5, 6])
   a[b==1]

列是否在list中
~~~~~~~~~~~~~~

::

   df1['a'].isin(list1)

分位数
~~~~~~

.. code:: python

   # 查看分位数
   data[['num']].quantile(q=[i/10.0 for i in range(1,10)]+[0.95,0.99,0.999,1])
   data[['num']].quantile(q=0.5)

按list取行
~~~~~~~~~~

.. code:: python

   df = pd.DataFrame(np.random.randint(1,10,size=[10,3]))
   row_list = [0,3,5]

   df[df.index.isin(row_list)]

相等比较
~~~~~~~~

.. code:: python

   from pandas.testing import assert_frame_equal

   a = pd.DataFrame(np.random.randint(1,100,size=(8,3)))
   b = pd.DataFrame(np.random.randint(1,100,size=(8,3)))
   assert_frame_equal(a,b)

   # 如果一样，不会返回任何东西，不一样会有差异分析

   #作为dataframe整体是否相等的比较，返回一个bool
   a.equals(b)

   # 逐个元素比较，返回形状一样的dataframe,每个位置是bool值
   a==b
   a.eq(b)

append/ concat
~~~~~~~~~~~~~~

.. code:: python

   行列和并堆叠

   pd.concat([a,b],axis=1,ignore_index=False)

   axis=1添加列，0添加行，ignore_index为True时，去掉原有的index或column,从0开始

   a = pd.DataFrame(np.random.randint(1,10,size=[2,3]))
   b = pd.DataFrame(np.random.randint(1,10,size=[2,3]))


       0   1   2
   0   5   4   4
   1   8   7   7

       0   1   2
   0   3   9   7
   1   9   6   9


   pd.concat([a,b],axis=0,ignore_index=True)
       0   1   2
   0   5   4   4
   1   8   7   7
   2   3   9   7
   3   9   6   9

   pd.concat([a,b],axis=1,ignore_index=True)

       0   1   2   3   4   5
   0   5   4   4   3   9   7
   1   8   7   7   9   6   9



   a.append(b,ignore_index=True)

       0   1   2
   0   5   4   4
   1   8   7   7
   2   3   9   7
   3   9   6   9

apply
~~~~~

.. code:: python

   # dataframe 对几列使用apply方法
   ans['1_percent'] = ans.apply(lambda x: sum_of_list(x['original_route'], x['temp'], 1), axis=1)

   # 注意apply默认axis=0,此时的x是一列的数据，axis=1时是一行


   # 根据dataframe a 某两列值的情况添加一列新的标签
   a = pd.DataFrame(np.random.randint(1,100,size=(8,3)))
   a[4]=a.apply(lambda x: x[0]>20,axis=1)

cut/ qcut
~~~~~~~~~

.. code:: python

   def cut(
       x,
       bins,
       right: bool = True,
       labels=None,
       retbins: bool = False,
       precision: int = 3,
       include_lowest: bool = False,
       duplicates: str = "raise",
       ordered: bool = True,
   ):

   # duplicates : {default 'raise', 'drop'}, optional
   # If bin edges are not unique, raise ValueError or drop non-uniques.

   bins = [0, 200, 1000, 5000, 10000]
   # 按照给定区间 使用 pd.cut 将数据进行离散化
   # 默认左开右闭 (0,200] < (200,1000] < (1000,5000] < (5000,10000]
   # 可以改为左闭右开
   # 不在bins内的返回nan

   # cut也可以直接将bins指定为int,则将值分为


   # cut是保证每个区间的长度一样
   # qcut是保证每个区间的元素个数一样


   a = pd.DataFrame(np.random.normal(size=[10000,1]))
   bins = [i/2 for i in range(-10,11)]
   a[1] = pd.cut(a[0],bins=bins)
   b = a.groupby(1,as_index=False)[0].count()
   b.plot.bar(x= 1,y=0,color='lightblue', alpha=0.7)


   d = pd.DataFrame([x**2 for x in range(11)])

       0
   0   0
   1   1
   2   4
   3   9
   4   16
   5   25
   6   36
   7   49
   8   64
   9   81
   10  100

   d_cut = d.copy()
   d_cut['cut_group'] =pd.cut(d_cut[0], 4)
   d_cut

       0   cut_group
   0   0   (-0.1, 25.0]
   1   1   (-0.1, 25.0]
   2   4   (-0.1, 25.0]
   3   9   (-0.1, 25.0]
   4   16  (-0.1, 25.0]
   5   25  (-0.1, 25.0]
   6   36  (25.0, 50.0]
   7   49  (25.0, 50.0]
   8   64  (50.0, 75.0]
   9   81  (75.0, 100.0]
   10  100 (75.0, 100.0]
            

   def qcut(
       x,
       q,
       labels=None,
       retbins: bool = False,
       precision: int = 3,
       duplicates: str = "raise",
   ):
            
            
            
   d_qcut = d.copy()
   d_qcut['qcut_group'] = pd.qcut(d_qcut[0], 4)
   d_qcut
           
       0   qcut_group
   0   0   (-0.001, 6.5]
   1   1   (-0.001, 6.5]
   2   4   (-0.001, 6.5]
   3   9   (6.5, 25.0]
   4   16  (6.5, 25.0]
   5   25  (6.5, 25.0]
   6   36  (25.0, 56.5]
   7   49  (25.0, 56.5]
   8   64  (56.5, 100.0]
   9   81  (56.5, 100.0]
   10  100 (56.5, 100.0]

   a  = pd.DataFrame()
   lists = [1] * 5 + [2]* 10 + [3]* 15
   a[0]=lists       
        
   pd.cut(a[0],3)不会报错
   pd.qcut(a[0],3) 会
            
   pd.qcut(a[0],3,duplicates='drop')
            
   频率区间法：按照等频率或指定频率离散化
   df['amount3'] = pd.qcut(df['amount'], 4, labels=['bad', 'medium', 'good', 'awesome'])

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211112143042092.png
   :alt: image-20211112143042092

   image-20211112143042092

drop_duplicates
~~~~~~~~~~~~~~~

.. code:: python

   DataFrame.drop_duplicates(subset=None, keep='first', inplace=False)



   subset : column label or sequence of labels, optional
   用来指定特定的列，默认所有列

   keep : {‘first’, ‘last’, False}, default ‘first’
   删除重复项并保留第一次出现的项

   inplace : boolean, default False
   是直接在原来数据上修改还是保留一个副本

explode
~~~~~~~

.. code:: python

   >>> df=pd.DataFrame({'等级':['青铜','王者','皇冠'],
                        '充值':[[10,10,10,9,9],[9,9,8,8,8,7,6],[7,6,6,6,5,5,4]]
                       })
   >>> df


       等级  充值
   0   青铜  [10, 10, 10, 9, 9]
   1   王者  [9, 9, 8, 8, 8, 7, 6]
   2   皇冠  [7, 6, 6, 6, 5, 5, 4]


   >>> x = df.explode('充值')
   >>> x

       等级  充值
   0   青铜  10
   0   青铜  10
   0   青铜  10
   0   青铜  9
   0   青铜  9
   1   王者  9
   1   王者  9
   1   王者  8
   1   王者  8
   1   王者  8
   1   王者  7
   1   王者  6
   2   皇冠  7
   2   皇冠  6
   2   皇冠  6
   2   皇冠  6
   2   皇冠  5
   2   皇冠  5
   2   皇冠  4

   # 注意index


   >>> x['C'] = [1]* x.shape[0]
   >>> x


       等级  充值  C
   0   青铜  10  1
   0   青铜  10  1
   0   青铜  10  1
   0   青铜  9   1
   0   青铜  9   1
   1   王者  9   1
   1   王者  9   1
   1   王者  8   1
   1   王者  8   1
   1   王者  8   1
   1   王者  7   1
   1   王者  6   1
   2   皇冠  7   1
   2   皇冠  6   1
   2   皇冠  6   1
   2   皇冠  6   1
   2   皇冠  5   1
   2   皇冠  5   1
   2   皇冠  4   1

fillna()
~~~~~~~~

.. code:: python

   DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)

groupby
~~~~~~~

.. code:: python

   # 分组求和 
   # as_index = False是为了防止index变为multi-index
   data.groupby(["uid","desc_mpid"],as_index = False).sum("cnt")
   data.groupby(["uid","desc_mpid"],as_index = False).apply(sum)  
   # 效果一样，但第二条速度慢多了，没有默认函数的情况下再用apply




   # 自定义函数，注意新建的columns的名字
   data.groupby(["uid","desc_mpid"],as_index = False).apply(lambda x: sum(x['cnt']))


   users.groupby("occupation").age.agg(["min","max","mean"])



   company=["A","B","C"]

   data=pd.DataFrame({
       "company":[company[x] for x in np.random.randint(0,len(company),10)],
       "salary":np.random.randint(5,50,10),
       "age":np.random.randint(15,50,10)
   }
   )

   data.groupby('company').agg({'salary':'median','age':'mean'})

            salary    age
   company
   A          21.5  27.50
   B          10.0  29.00
   C          30.0  27.25


   将组内结算结果赋给每组的成员
   一：

   avg_salary_dict = data.groupby('company')['salary'].mean().to_dict()
   data['avg_salary'] = data['company'].map(avg_salary_dict)

   二：

   data.groupby('company')['salary'].transform('mean')

       company  salary  age  avg_salary
   0       C      43   35       29.25
   1       C      17   25       29.25
   2       C       8   30       29.25
   3       A      20   22       21.50
   4       B      10   17       13.00
   5       B      21   40       13.00
   6       A      23   33       21.50
   7       C      49   19       29.25
   8       B       8   30       13.00

   # 对于groupby后的apply，以分组后的子DataFrame作为参数传入指定函数的，基本操作单位是DataFrame

   def get_oldest_staff(x):
       df = x.sort_values(by = 'age',ascending=True)
       return df.iloc[-1,:]

   data.groupby('company',as_index=False).apply(get_oldest_staff)

   # 虽然说apply拥有更大的灵活性，但apply的运行效率会比agg和transform更慢。
   # 所以，groupby之后能用agg和transform解决的问题还是优先使用这两个方法，
   # 实在解决不了了才考虑使用apply进行操作



   # 要实现多列统计不同的函数，只需要新建一个空的dataframe,然后每次添加计算结果作为新的一列即可


   a = pd.DataFrame()

   a[0] = ans.groupby('pt_dt').next_day.agg('count')
   a[1] = ans.groupby('pt_dt').next_day.apply(label_count)
   a[2] = a[1] / a[0]
   a

                   0   1   2
   pt_dt           
   2021-11-01  300266  5542    0.018457
   2021-11-02  297289  5238    0.017619
   2021-11-03  297161  5337    0.017960
   2021-11-04  294732  5065    0.017185
   2021-11-05  298289  6111    0.020487
   2021-11-06  296181  5865    0.019802
   2021-11-07  295426  5949    0.020137
   2021-11-08  296464  5385    0.018164
   2021-11-09  293347  4971    0.016946
   2021-11-10  292051  4981    0.017055
   2021-11-11  290456  4759    0.016385
   2021-11-12  291972  4758    0.016296
   2021-11-13  288225  4944    0.017153
   2021-11-14  288429  5082    0.017620
   2021-11-15  298986  5564    0.018610

group_concat
~~~~~~~~~~~~

.. code:: python

   import pandas as pd

   df = pd.DataFrame({
     "name":["小明","小明","小明","小红","小张","小张"],
     "score":[10,20,20,20,200,500]
   })

   df

       name    score
   0   小明  10
   1   小明  20
   2   小明  20
   3   小红  20
   4   小张  200
   5   小张  500

   df.groupby('name').agg({'score':list})

       score
   name    
   小张  [200, 500]
   小明  [10, 20, 20]
   小红  [20]


   df.groupby('name').agg({'score':list}).reset_index()

       name    score
   0   小张  [200, 500]
   1   小明  [10, 20, 20]
   2   小红  [20]


   df.astype(str).groupby('name').apply(lambda x:";".join(x.score))

   name
   小张     200;500
   小明    10;20;20
   小红          20
   dtype: object
       
       
   (df.astype(str)
    .groupby('name')
    .apply(lambda x:";".join(x.score))
    .to_frame('score')
    .reset_index()
   )

       name    score
   0   小张  200;500
   1   小明  10;20;20
   2   小红  20


   df.groupby('name').agg({'score':'unique'})

       score
   name    
   小张  [200, 500]
   小明  [10, 20]
   小红  [20]

   df.groupby('name').agg({'score':'unique'}).reset_index()

       name    score
   0   小张  [200, 500]
   1   小明  [10, 20]
   2   小红  [20]


   df.groupby('name').agg({'score':list})['score'].apply(lambda x: sorted(x,reverse=True)).reset_index()

       name    score
   0   小张  [500, 200]
   1   小明  [20, 20, 10]
   2   小红  [20]

insert
~~~~~~

.. code:: python

   a = pd.DataFrame(np.random.randint(1,10,[5,3]))
   a

       0   1   2
   0   4   1   7
   1   7   8   4
   2   6   1   7
   3   4   9   1
   4   5   9   7

   a.insert(0,'pt_dt','2021-12-01')

       pt_dt   0   1   2
   0   2021-12-01  4   1   7
   1   2021-12-01  7   8   4
   2   2021-12-01  6   1   7
   3   2021-12-01  4   9   1
   4   2021-12-01  5   9   7


   # 直接在原dataframe上操作，没有return

iterrows
~~~~~~~~

.. code:: python

   df = pd.DataFrame({
       'node1': [1, 2, 3],
       'node2': [2, 3, 4]
   })

   for index, row in df.iterrows():
       print(index,row["node1"],row["node2"])
       
   >>> type(row) 
   <class 'pandas.core.series.Series'>

   既可以用 row['node1'],也可以用 row[0]

itertuples
~~~~~~~~~~

.. code:: python


   a = pd.DataFrame(np.random.randint(3,10,size=(8,3)))
   for i in a.itertuples():
       print(i)
       print(type(i))
       print(i[1])
       break
       
   for i,j,k in a.itertuples():
       print(i,j,k)

join
~~~~

.. code:: python

   t = np.random.choice(range(6),size = 6, replace= False)
   mu = np.random.choice(range(6),size = 6, replace= False)
   t,mu
   (array([4, 5, 3, 0, 2, 1]), array([0, 1, 4, 3, 2, 5]))

   a= pd.DataFrame({'0':t,
                   '1':[i*2+1 for i in t]})
       0   1
   0   4   9
   1   5   11
   2   3   7
   3   0   1
   4   2   5
   5   1   3

   b = pd.DataFrame({'7':mu,
                   '5':[i*i+4 for i in mu]})

       7   5
   0   0   4
   1   1   5
   2   4   20
   3   3   13
   4   2   8
   5   5   29

   a.join(b.set_index('7'), on='0', how='left')
       0   1   5
   0   4   9   20
   1   5   11  29
   2   3   7   13
   3   0   1   4
   4   2   5   8
   5   1   3   5

map
~~~

.. code:: python

   Series 根据字典替换值

   # python 自己也有map
   # map()是一个 Python 内建函数，它允许你不需要使用循环就可以编写简洁的代码。


   map(function, iterable, ...)
   function - 针对每一个迭代调用的函数
   iterable - 支持迭代的一个或者多个对象。在 Python 中大部分内建对象，例如 lists, dictionaries, 和 tuples 都是可迭代的。

   当提供多个可迭代对象时，返回对象的数量大小和最短的迭代对象的数量一致

   a = [1, 4, 6]
   b = [2, 3, 5]
    
   result = map(lambda x, y: x*y, a, b)
    
   print(list(result))

nan判断、统计
~~~~~~~~~~~~~

.. code:: python

   # nan判断，可以放入值，series，dataframe
   pd.isna(data['label'][0])
   pd.isna(data['label'])
   pd.isna(data[['label']])

   vertex_df.isna().sum()

   uid              0
   is_loss          0
   city         38227
   member      667230
   battle      664506
   family      622684
   fan_club    614239
   dtype: int64

pivot
~~~~~

.. code:: python

   data = x.groupby(by=['等级','充值'], as_index=False).count()

       等级  充值  C
   0   王者  6   1
   1   王者  7   1
   2   王者  8   3
   3   王者  9   2
   4   皇冠  4   1
   5   皇冠  5   2
   6   皇冠  6   3
   7   皇冠  7   1
   8   青铜  9   2
   9   青铜  10  3


   data = pd.pivot(data, index='等级', columns='充值')
   data.columns = [i[1] for i in data.columns]
   data.index.name = None
   data

       4   5   6   7   8   9   10
   王者  NaN NaN 1.0 1.0 3.0 2.0 NaN
   皇冠  1.0 2.0 3.0 1.0 NaN NaN NaN
   青铜  NaN NaN NaN NaN NaN 2.0 3.0

pivot_table
~~~~~~~~~~~

.. code:: python

   pivot 只是reshape, 而pivot_table可以做各种统计计算

plot
~~~~

.. code:: python

       1   balance
   0   (-0.001, 50.0]  8974
   1   (50.0, 200.0]   6283
   2   (200.0, 1000.0] 9590
   3   (1000.0, 5000.0]    8224
   4   (5000.0, 10000.0]   2252
   5   (10000.0, 30000.0]  2841
   6   (30000.0, 100000.0] 1641
   7   (100000.0, 300000.0]    860
   8   (300000.0, 500000.0]    196
   9   (500000.0, 519797860.0] 657

   c.plot.bar( color='k', alpha=0.7)

   c.plot.barh( x= 1,color='k', alpha=0.7)


   a = pd.DataFrame(np.random.randint(1,100,size=[6,3]))
   a.plot.barh(x= 0,y=[1,2],color=['lightblue','Gray'], alpha=0.7)

rank
~~~~

.. code:: python

   def rank(
       self: FrameOrSeries,
       axis=0,
       method: str = "average",
       numeric_only: Optional[bool_t] = None,
       na_option: str = "keep",
       ascending: bool_t = True,
       pct: bool_t = False,
       ) -> FrameOrSeries:

reindex
~~~~~~~

.. code:: python

reset_index
~~~~~~~~~~~

.. code:: python

   # 扔掉原有索引，重新建立从0到1的index, 若drop为False,则会将原index变为一列
   df.reset_index(drop=True,inplace=False)

replace
~~~~~~~

.. code:: python

   # 值替换
   data.replace([70102, 70115, 70113, 70144], [1, 2, 3, 4])

   # 根据字典替换值
   data.replace(dicts)

   # 非nan替换为0，nan替换为1
   data['label'] = data.label.map(lambda x: 0 if pd.isna(x) else 1)

.. code:: python

   # 注意 list 和 pandas.core.series.Series 的区别
   # 当list长度为1时，可以del list[0],得到[], 而series 会报错
   loc = self.axes[-1].get_loc(key)
   raise KeyError(key) from err
   KeyError: 0

Series值替换
~~~~~~~~~~~~

.. code:: python

   new_df = df.replace({"col1": column_dict})


   df['col1'] = df['col1'].map(column_dict)

   数据量大的情况下，map比 replace 快了不是一点点

   replace是查找值，若值在dict里，则替换为对应的value,不在则保留
   map是对每一个位置的元素做同样的操作

shuffle
~~~~~~~

.. code:: python

   df.sample(frac=1).reset_index(drop=True)

stack/unstack
~~~~~~~~~~~~~

.. code:: python

sort
~~~~

.. code:: python

   # dataframe 按某一列排序
   ret.sort_values(by="num", inplace=True, ascending=False)

   # 多列排序
   test.sort_values(by=[0,1].ascending=(False,True))

to_numpy()
~~~~~~~~~~

.. code:: python

   >>> pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
   array([[1, 3],
          [2, 4]])

unique/ nunique
~~~~~~~~~~~~~~~

.. code:: python

   >>> a = pd.DataFrame(np.random.randint(1,10,size=(8,3)))
   >>> a

       0   1   2
   0   5   2   9
   1   1   6   2
   2   8   2   8
   3   9   3   8
   4   8   5   7
   5   4   1   6
   6   8   5   2
   7   9   8   6

   >>> a[0].unique()
   array([5, 1, 8, 9, 4])

   >>> a[0].nunique()
   5

Numpy
-----

cumsum/cumprod
~~~~~~~~~~~~~~

.. code:: python

   a = np.array([i for i in range(1,10)])
   a

   array([1, 2, 3, 4, 5, 6, 7, 8, 9])

   a.cumsum()
   array([ 1,  3,  6, 10, 15, 21, 28, 36, 45], dtype=int32)


   a.cumprod()
   array([1, 2, 6, 24, 120, 720, 5040, 40320,362880], dtype=int32)

np.maximum
~~~~~~~~~~

.. code:: python

   a = [1,3,5,7,0]
   b = [5,2,7,4,9]

   >>> np.maximum(a,b)
   array([5, 3, 7, 7, 9])


   >>> np.minimum(a,b)
   array([1, 2, 5, 4, 0])

choice/ sample
~~~~~~~~~~~~~~

.. code:: python

   import random
   lists = [i for i in range(10)]
   random.sample(lists,2)


   numpy.random.choice(a, size=None, replace=True, p=None)
   #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
   #replace:是否放回
   #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

Basic Indexing and Slicing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   #  An important first distinction from Python’s built-in lists is that array slices 
   #  are views on the original array.
   # This means that the data is not copied, and any modifications to the view will be
   # reflected in the source array.

   arr = np.arange(10)
   arr

   array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

   arr_slice = arr[5:8]
   arr_slice

   array([5, 6, 7])

   arr_slice[1] = 12345
   array([0, 1, 2, 3, 4, 5, 12345, 7, 8, 9])


   If you want a copy of a slice of an ndarray instead of a view, you
   will need to explicitly copy the array—for example,
   arr[5:8].copy()

np.arange 生成日期
~~~~~~~~~~~~~~~~~~

.. code:: python

   a  = np.arange('2019-07-06', '2020-10-12', dtype='datetime64[M]')
   a


   datetime64 [Y]  [M] [W] [D] 分别代表 year,month,week day,


   h   hour    
   m   minute  
   s   second
   ms  millisecond 
   us  microsecond 
   ns  nanosecond  
   ps  picosecond  
   fs  femtosecond
   as  attosecond  

   for i in a:
       np.datetime_as_string(i)
       # str(i)
       
       
   a = np.datetime64('2021-10-01')
   delta = np.timedelta64(140,'D')
   a-delta

   numpy.datetime64('2021-05-14')

np.where
~~~~~~~~

.. code:: python

   xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
   yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
   cond = np.array([True, False, True, True, False])

   result = np.where(cond, xarr, yarr)

   array([ 1.1, 2.2, 1.3, 1.4, 2.5])

   arr = np.random.randn(4, 4)
   np.where(arr > 0, 2, -2)
   np.where(arr > 0, 2, arr)

NetwrokX
--------

::

   任意可hash的对象都可以作为node,
   可以重复添加节点，已有的不会被重新创建，
   但是属性如果不一样了会被覆盖
   添加edge时不需要node已经在graph中，不在的则会被创建
   G.add_edge(1,2,year=0)

.. code:: python

   NetworkX与igraph 都允许 一个节点到自己有边
   不同之处在于，NetworkX中，一条边若已经存在，重复添加没有效果
   而在igraph中允许两个节点间有多条边


   Self loops are allowed but multiple (parallel) edges are not.

.. code:: python

   louvain社区发现算法需要下载包

   pip install python-louvain

测试数据
~~~~~~~~

.. code:: python

   test = nx.Graph()
   lists = [i for i in range(50000)]
   test.add_nodes_from(lists)
   edge_list = [random.sample(lists, 2) for i in range(100000)]
   test.add_edges_from(edge_list)

可视化
~~~~~~

.. code:: python

   import matplotlib.pyplot as plt
   import networkx as nx

   nx.draw(test, with_labels=True)
   plt.show()

dir(nx.Graph( ))
~~~~~~~~~~~~~~~~

.. code:: python

   ['__class__',
    '__contains__',
    '__delattr__',
    '__dict__',
    '__dir__',
    '__doc__',
    '__eq__',
    '__format__',
    '__ge__',
    '__getattribute__',
    '__getitem__',
    '__gt__',
    '__hash__',
    '__init__',
    '__init_subclass__',
    '__iter__',
    '__le__',
    '__len__',
    '__lt__',
    '__module__',
    '__ne__',
    '__new__',
    '__reduce__',
    '__reduce_ex__',
    '__repr__',
    '__setattr__',
    '__sizeof__',
    '__str__',
    '__subclasshook__',
    '__weakref__',
    '_adj',
    '_node',
    'add_edge',
    'add_edges_from',
    'add_node',
    'add_nodes_from',
    'add_weighted_edges_from',
    'adj',
    'adjacency',
    'adjlist_inner_dict_factory',
    'adjlist_outer_dict_factory',
    'clear',
    'clear_edges',
    'copy',
    'degree',
    'edge_attr_dict_factory',
    'edge_subgraph',
    'edges',
    'get_edge_data',
    'graph',
    'graph_attr_dict_factory',
    'has_edge',
    'has_node',
    'is_directed',
    'is_multigraph',
    'name',
    'nbunch_iter',
    'neighbors',
    'node_attr_dict_factory',
    'node_dict_factory',
    'nodes',
    'number_of_edges',
    'number_of_nodes',
    'order',
    'remove_edge',
    'remove_edges_from',
    'remove_node',
    'remove_nodes_from',
    'size',
    'subgraph',
    'to_directed',
    'to_directed_class',
    'to_undirected',
    'to_undirected_class',
    'update']

   # convert_matrix.py下，可以直接从dataframe导入edgelist
   [
       "from_numpy_matrix",
       "to_numpy_matrix",
       "from_pandas_adjacency",
       "to_pandas_adjacency",
       "from_pandas_edgelist",
       "to_pandas_edgelist",
       "to_numpy_recarray",
       "from_scipy_sparse_matrix",
       "to_scipy_sparse_matrix",
       "from_numpy_array",
       "to_numpy_array",
   ]

create
~~~~~~

.. code:: python

   df = pd.DataFrame({
       'node1': [1, 2, 3],
       'node2': [2, 3, 4]
   })
   G = nx.from_pandas_edgelist(df, "node1", "node2")


   g = nx.from_pandas_edgelist(edge_df, "uid0", "uid1",edge_attr=True)
   7.98s



   for i in edge_df.index:
       g.add_edge(edge_df['uid0'][i],edge_df['uid1'][i],cnt=edge_df['cnt'][i])
   25.9s

   # networkx中没有index，现在edge list的顺序不是dataframe的顺序
   # 会自己做优化，有排序
   # 在计算common friends的时候，前期比较慢，后面特别快，可能是按邻居节点多少做了先后排序

   h.add_edge()
   h.add_edges_from()
   h.add_node()
   h.add_nodes_from()

attribute
~~~~~~~~~

.. code:: python

   nx.set_node_attributes()
   nx.get_node_attributes()
   nx.set_edge_attributes()
   nx.get_edge_attributes()


   >>> G = nx.path_graph(3)
   >>> bb = nx.edge_betweenness_centrality(G, normalized=False)
   >>> nx.set_edge_attributes(G, bb, "betweenness")
   >>> G.edges[1, 2]["betweenness"]
   2.0


   If you provide a list as the second argument, updates to the list will be reflected in the edge attribute for each edge:
       
   >>> labels = []
   >>> nx.set_edge_attributes(G, labels, "labels")
   >>> labels.append("foo")
   >>> G.edges[0, 1]["labels"]
   ['foo']
   >>> G.edges[1, 2]["labels"]
   ['foo']

igraph
------

.. code:: python

   g = Graph([(0,1), (0,2), (2,10)])
   summary(g)

   IGRAPH U--- 11 3 --
   # 这里是index

.. code:: python

   注意name中的字段并不具有唯一性，会被反复添加

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211028203819776.png
   :alt: image-20211028203819776

   image-20211028203819776

.. _create-1:

create
~~~~~~

.. code:: python

   g = Graph()
   g.add_vertex('apple')
   g.add_vertex('google')
   g.add_vertex('apple')
   g.vs['label'] =['apple','google','apple']
   list(g.vs)


   g.add_edges([('apple','apple'),('apple','google')],{'label' :[3,4]})


   layout = g.layout("kk")
   plot(g, layout=layout,bbox=(200, 200))


   # 通过edge端点的name得到该edge在es中的index
   g.get_eid('apple','google')

.. code:: python

   # 添加单个节点
   g = Graph()
   g.add_vertex(23,year = 203,gender = 0)
   list(g.vs)

   [igraph.Vertex(<igraph.Graph object at 0x000001B6B1265048>, 0, {'year': 203, 'gender': 0, 'name': 23})]

.. code:: python

   # 添加多个节点

   g = Graph()
   uid_list = [3,5,6,9]
   gender = [0,0,1,0]
   g.add_vertices(len(uid_list),attributes={'name':uid_list,'gender':gender})
   list(g.vs)

   [igraph.Vertex(<igraph.Graph object at 0x000001B6B1265318>, 0, {'name': 3, 'gender': 0}),
    igraph.Vertex(<igraph.Graph object at 0x000001B6B1265318>, 1, {'name': 5, 'gender': 0}),
    igraph.Vertex(<igraph.Graph object at 0x000001B6B1265318>, 2, {'name': 6, 'gender': 1}),
    igraph.Vertex(<igraph.Graph object at 0x000001B6B1265318>, 3, {'name': 9, 'gender': 0})]

.. code:: python

   # 添加单个edge

   g= Graph()
   g.add_vertices(2,{'name':['apple','google']})
   g.add_edge('apple','google',year=34,num=45)


   # 添加多个edge 
   # 通过这种方式添加edge时，端点必须已经在vs里，不像NetworkX
   # 但是在创建graph时可以只通过edge来创建

   g= Graph()
   name = ['a','b','c','d','e']
   num = [3,5,7,9,0]
   g.add_vertices(5,{'name':['a','b','c','d','e'],'label':name})
   # g.add_edges([(1,2),(3,0)],{'label' :[3,4]})
   g.add_edges([('a','c'),('d','e')],{'label' :[3,4]})


   # 从dataframe创建
   g = Graph().DataFrame(df, directed=False)



   t = Graph.DataFrame(df,directed=False,use_vids=True,vertices=z)

   # df 的前两列是vertex, use_vids=True时，是vertex在图中的index，从0到t.vcount()
   # vertices 是vertex的df,第一列是vertex的uid,不能有重复值，从上到下一次代表0到vcount()
   # 列在图中的attribute是name
   # 列是vertex的其他属性，attribute就是列名

.. _plot-1:

plot
~~~~

.. code:: python

   # 画图

   layout = g.layout("kk")
   plot(g, layout=layout,bbox=(280, 280))

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211028203436878.png
   :alt: image-20211028203436878

   image-20211028203436878

subgraph
~~~~~~~~

.. code:: python

   h = Graph([(0,1),(1,2),(1,3),(2,4),(3,4),(3,5),(4,5),(5,6),(6,7),(7,8)])
   h.vs['label'] = [i for i in range(9)]
   h.vs['name'] =['a','b','c','d','e','f','g','h','i']
   layout = h.layout("kk")
   plot(h, layout=layout,bbox=(200, 200))


   h.get_edge_dataframe()



   def show(graph):
       layout = graph.layout("kk")
       return plot(graph, layout=layout,bbox=(200, 200))


   # 注意同样的节点在子图和原图中的index是不一样的，因为index始终是从0到n

   # 按边创建子图
   def func(edge):
       return edge.index<6

   def func1(vertex):
       return vertex.index % 2==1

   h_sub = h.subgraph_edges(h.es.select(func))
   h_sub = h.subgraph_edges(h.es.select([e for e in h.es if e.index<6]))

   u = h.subgraph_edges([e for e in h.es if e.index < 6])
   show(u)


   # 按节点创建子图
   h_sub1 = h.subgraph(h.vs.select(func1))
   h.subgraph([v for v in h.vs if v.index % 2==1 or v.degree() >1])

BFS
~~~

(Breadth First Search)

.. code:: python

   # 测试数据
   h = Graph([(0,1),(1,2),(1,3),(2,4),(3,4),(3,5),(4,5),(5,6),(6,7),(7,8)])
   h.vs['label'] = [i for i in range(9)]
   h.vs['name'] =['a','b','c','d','e','f','g','h','i']
   layout = h.layout("kk")
   plot(h, layout=layout,bbox=(200, 200))


   h.bfs(3)
   # 1. vertex的id遍历的顺序
   # 2. 1中哪些index是新的一层
   # 3. g.vs中按id,每个的(直系)parent是谁

   # 2到3步内的点的index
   h.neighborhood(i,mindist=2,order=3)

alter attribute
~~~~~~~~~~~~~~~

.. code:: python

   h = Graph([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),(6,4),(6,5),(6,7),(6,8)])
   h.vs['label'] = [i for i in range(h.vcount())]
   h.vs[6]['label'] = 'a'
   h.vs[0]['label']='b'
   layout = h.layout("kk")
   plot(h, layout=layout,bbox=(200, 200))


   # no use, wrong
   h.vs['label'][6]='b' 

Notebook
--------

插件安装
~~~~~~~~

::



   pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install

按照 jupyter notebook 的 extension 配置后，反而tab 提示模块和命令
功能不正常

查了一下有说 jedi
版本的，我这边环境依赖问题没办法直接改版本，发现有个魔术命令可以直接关闭调整jedi的功能
运行之后，环境正常了

代码补全
~~~~~~~~

.. code:: python

   %config Completer.use_jedi = False

字符串操作
----------

.. code:: python

   # 按行读取txt文件，避免文件过大，一次性读入占太多内存

   filename = "input.txt"
   with open(filename) as files:
       for line in files:
           print(line)

   # 按指定字符将字符串分隔，返回list
   line.split(" ")


   # 将第一个位置的（子）字符替换为第二个位置
   line.replace("\n", "") 

   # 判断是否存在指定子字符串直接用in

   "avg" in line

ML
==

Keras 损失函数：categorical_crossentropy,
sparse_categorical_crossentropy

的区别,见\ `链接 <https://blog.csdn.net/CxsGhost/article/details/106095615?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1.essearch_pc_relevant&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1.essearch_pc_relevant>`__

.. code:: python

   如果 labels 是 one-hot 编码，用 categorical_crossentropy
   one-hot 编码
   　　[[0, 1, 0],
   　　 [1, 0, 0],
   　　 [0, 0, 1]]
   每条每一行是一个label的编码，1所在的位置代表label

   如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy
   数字编码：[2, 0, 1, 5, 19]
   每个数字就是label

Linux
=====

::

   # 创建文件夹
   mkdir name

   # 移动文件
   move name [目标文件夹]

   # 删除文件
   rm [文件名]

   # 删除目录(会询问是否删除子目录)
   rm -r /test

   # 删除目录下所有文件
   rm -rf /test

::

   #打开文件，若没有则新建
   vim hello_world.sql

   # 首先按Esc退出编辑模式，保存文件并退出vim
   :wq   
   # (write,quit)

Crontab定时任务
---------------

::

   crontab -l
   查看当前用户下的定时任务

   crontab -e
   新增或编辑定时任务

   0 9 * * * /home/luoxb kinit -kt luoxb.keytab luoxb  
   每天早上9点运行命令

SQL
===

my uid
~~~~~~

::

   929731696

服务器认证
~~~~~~~~~~

::

   kinit -kt luoxb.keytab luoxb

Notebook连接
~~~~~~~~~~~~

::

   nohup jupyter notebook > jupter.log 2>&1 &

hive/impala 差别
~~~~~~~~~~~~~~~~

::

   impala和hive两者有微妙的不同。

   union all的时候，上下两部分都只有一个uid字段，但是上面字段名叫pid，下面叫uid，impala支持，但是hive会报错

why insert data is forbidden in impala
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   1. impala 一次写入太多数据，会占用整个集群的资源，有可能引起整个集群的崩溃

   2. impala 写入的数据也会自动同步到hive,但是不知道是什么时候，有可能写入了，但还没同步，会导致在hive里无法查到

   3. 数据部分丢失

explain
~~~~~~~

.. code:: sql

   explain
   select * from ....

create table
~~~~~~~~~~~~

.. code:: sql

   create table if not exists luoxb.test(
    round int,
    a int,
    b int,
    coin int
    ) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n' STORED AS textfile;


   CSV文件的分割符 选  ',' 
   Txt 文件的分隔符 选  '\t' 

   # 将txt文件读入表中
   load data inpath '/user/luoxb/ne.txt' overwrite into table luoxb.test


   # 删除表
   drop table if exists luoxb.test


   # 将hive中创建的表同步到impala中，（只能）在impala中运行
   INVALIDATE METADATA luoxb.test



   load data inpath '/user/yexh/0626newuser.csv' overwrite into table zhangxiang.chunqiu_newuser partition(pt_dt='2019-06-26')

INVALIDATE METADATA
~~~~~~~~~~~~~~~~~~~

.. code:: sql

   -- 清空表中数据，保留列，及其数据类型

   TRUNCATE table_name

插入分区，删除分区，查看分区
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   CREATE TABLE IF NOT EXISTS luoxb.jjmall_promote_activity56_user(UID bigint,rnk bigint)
   PARTITIONED BY(pt_dt string,classid int);


   INSERT OVERWRITE TABLE luoxb.jjmall_promote_activity56_user  
   partition(pt_dt='2021-11-24',classid = 1)
   SELECT *....;


   # 只会删除指定分区
   alter table luoxb.jjmall_promote_activity56_user  
   drop partition(pt_dt='2021-11-24',classid = 1);

   alter table luoxb.jjmall_promote_activity56_user  
   drop partition(pt_dt='2021-11-24',classid in (1,2,3));

   alter table tablename  drop partition(etl_dt>='2018-01-01')


   show partitions table_name

查询结果建表
~~~~~~~~~~~~

.. code:: sql

   create table luoxb.lucky_0501_0820 as select * from ( )

   -- 注意不能在with temp as ()之后

   # 插入已有表（字段完全一样，只是增加行）
   insert into tab1 select * from tab2 

::

   insert overwrite / insert into


   两者都可以向 hive 表中插入数据，
   但 insert into 操作是以追加的方式向 hive 表尾部追加数据，
   而 insert overwrite 操作则是直接重写数据，即先删除 hive 表的数据，再执行写入操作。
   注意，如果 hive 表是分区表的话，insert overwrite 操作只会重写当前分区的数据，不会重写其他分区数据。

`add column <https://blog.csdn.net/Yvettre/article/details/80239531?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

按Ctrl同时点击打开链接

[`Hive列添加/更新/删除/虚拟列) <https://www.cnblogs.com/luxj/p/13352464.html>`__

.. code:: sql

   # 将查询结果作为新的一列添加到表中




   ---添加列
   ALTER TABLE 表1 ADD [列1] INT ,[列2] INT

   --- 在tb_student表额外添加一列 L3VPN，并且该L3VPN列含有默认值 apink，添加的位置在指定的列sex的后面
   ALTER TABLE tb_student ADD l3vpn VARCHAR(10) DEFAULT 'apink' AFTER sex


   ---使用alter命令在原表格test上创建新的字段new_data，数据类型为int，默认为NULL
   ---使用update、inner join、select共同实现更新值


   alter table test add columns(new_data int);

   insert overwrite table tablename
   select a.col1,a.col2,b.col2
   from tablename a join mastertable b on a.col3=b.col1


   ---可添加多个列，用逗号分隔开,可以添加注释
   alter table test add columns(new_data int COMMENT '商品销售信息表', pt_dt datetime);




   --删除列
   ALTER TABLE XXX DROP COLUMN Grade




   INSERT INTO 表1([列1],[列2])
   SELECT [列1],[列2] FROM 表2

.. code:: sql

   # 修改列的位置
   ALTER TABLE 表名 MODIFY 字段1 数据类型 FIRST|AFTER 字段2

timestamp, to_date
~~~~~~~~~~~~~~~~~~

.. code:: sql

   # 注意hive 和 impala 里时间戳所在时区
   # impala 里需要加8小时

   from_utc_timestamp(from_unixtime(desc_ret,"yyyy-MM-dd HH:mm:ss"),'PRC') as last_time

   from_unixtime(desc_ret+ 8* 3600)

.. code:: sql

   -- 指定日期之前60日

   WHERE pt_dt BETWEEN to_date(date_sub("2021-08-01",60)) AND "2021-08-01"


   -- 第一个日期-第二个日期

   -- datediff(TIMESTAMP enddate, TIMESTAMP startdate)
   -- Returns the number of days between two TIMESTAMP values.

   datediff("2021-08-10","2021-08-01")

   注意，使用
   datediff("2021-08-22",pt_dt)<= 30
   时，包含的天数是2021-07-23之后的所有天数，计算会包括负数

   0 < datediff("2021-08-22","2021-08-22") <= 3 得出来是True
   从左往右计算，False <= 3, False 记为 0

本月最后一天
~~~~~~~~~~~~

.. code:: sql

   to_date(months_add((date_sub(now(), day(now()))),1))

   -- 上月最后一天
   to_date((date_sub(now(), day(now())))) 

   -- 本月第一天
   to_date((date_sub(now(), day(now())-1)))

.. code:: sql

   -- 想在sql中验证一个结算结果的值可以使用如下方法，
   -- 选了，但是完全没选
   select date_sub("2021-08-25",3)="2021-08-22" as label from dwd.dwd_game_rec_2ddz
   where pt_dt = "2021-08-23"
   limit 10

last_value
~~~~~~~~~~

.. code:: sql

   with temp as (
   SELECT row_number() over(ORDER BY mp_id) uid from dim.dim_product_info_d
   limit 20
   )
   ,temp1 as (
   select uid
   ,uid * uid as day_gap_regist
   from temp
   )

   ,temp2 as (
   select 
   uid
   , if(uid in (1,4,7,9,10,11,15),null,day_gap_regist) day_gap_regist
   from temp1
   )

   SELECT *
   ,last_value(day_gap_regist ignore nulls)  over (order by uid)  --impala
   -- ,last_value(day_gap_regist,True)  over (order by uid)  --hive
   from temp2

.. code:: sql

   -- 要限制uid在指定集合中除了用

   where uid in (select * from temp)

   -- 还可以使用 

   inner join on

.. code:: sql

   -- 打标签的方式，
   -- 先把连接的键和label的值和名字拿出来，
   -- 下面是取统计对应的值，全量的
   -- 最后使用inner join 筛选

   left join(
       SELECT distinct t1.uid,
                 1 AS if_ip_ab
   )

   -- 没有的自动是null

   -- 不需要用
   if(bool,1,0)
   case when

连续时间窗口
~~~~~~~~~~~~

.. code:: sql

   join 表的时候，     
   on 后面跟的不只有 =     
   只要是bool值都可以    



   with temp as (    
   SELECT a.uid    
   ,a.pt_dt    
   ,b.uid is null as is_loss    
   from(    
   SELECT uid,pt_dt     
   from dw.dw_bal_user_match_stat_d    
   where pt_dt BETWEEN '2021-10-09' and date_add('2021-10-09',10)    
   GROUP BY uid,pt_dt    
   )a    
       
   LEFT JOIN (    
   SELECT uid,pt_dt    
   from dw.dw_bal_user_match_stat_d    
   where pt_dt BETWEEN '2021-10-09' and date_add('2021-10-09',10+16)    
   GROUP BY uid,pt_dt    
       
   )b    
       
   on a.uid = b.uid    
       
   and b.pt_dt BETWEEN date_add(a.pt_dt,1) and date_add(a.pt_dt,15)    

   GROUP BY a.uid,a.pt_dt,b.uid    
   )    

       
   select pt_dt, count(distinct uid),sum(is_loss),sum(is_loss) / count(distinct uid)    
   from temp     
   group by pt_dt    
   order by pt_dt    

.. code:: sql

   with temp as ()

::

   # 去重
   distinct 和 group by 效率

::

   # 用impala 拉数据会少前101条数据，应该用hive

greatest/ least
~~~~~~~~~~~~~~~

.. code:: sql


    SELECT id, chinese, math, english,
        greatest (chinese, math, english) max,
       least(chinese, math, english) min
     FROM tb

结果保留两位小数 / 小数转百分数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   cast(number as decimal(10,2))

   round(12.342,2)   12.34
   round(12,2)       12

   cast(10 as decimal(10,2))   10.00


   concat(cast(cast((count(distinct uid)/ dau) *100 as decimal(10,3)) as string),'%') ratio

.. code:: sql

   min 和 max 是取一列中的最大最小，
   least(), greatest()

类型转换, 模糊查询
~~~~~~~~~~~~~~~~~~

.. code:: sql

   where cast(mp_id as string) like "501%"

substring
~~~~~~~~~

.. code:: sql

   substring(s，n，len)          n-开始位置，len-截取长度


   SUBSTRING('computer',3)       mputer
   SUBSTRING('computer',3,4)     mput
   SUBSTRING('computer',-3)      ter
   SUBSTRING('computer',-5,3)    put

lpad/rpad
~~~~~~~~~

::

   字符串补齐到指定长度，超过的截取


   lpad("23",3,"0"),          023
   lpad("123456",3,"0"),      123
   rpad("12",3,"0"),          120
   rpad("12345",3,"0")        123

.. code:: sql

   有分隔符字符串计数

   name_list = "hello,world,this,is"

   LENGTH(name_list)-LENGTH(REPLACE(name_list,',',''))+1 

.. _group_concat-1:

group_concat
~~~~~~~~~~~~

.. code:: sql

   SELECT id,GROUP_CONCAT(score SEPARATOR ';') FROM testgroup GROUP BY id

    
   SELECT id,GROUP_CONCAT(DISTINCT score) FROM testgroup GROUP BY id

   -- 目前impala 只支持distinct，不支持 order by
   SELECT id,GROUP_CONCAT(score ORDER BY score DESC) FROM testgroup GROUP BY id



   SELECT mac,group_concat(cast(uid as string),",")
   from  dwm.dwm_usr_online_mac_d
   WHERE pt_dt = "2021-10-01"
   GROUP BY mac

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210908100301244.png
   :alt: image-20210908100301244

   image-20210908100301244

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210908100349719.png
   :alt: image-20210908100349719

   image-20210908100349719

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210908100442710.png
   :alt: image-20210908100442710

   image-20210908100442710

row_number( )
~~~~~~~~~~~~~

.. code:: sql

   # 分组取前十

   ROW_NUMBER() OVER (
       [PARTITION BY partition_expression, ... ]
       ORDER BY sort_expression [ASC | DESC], ...
   )

   PARTITION BY子句将结果集划分为分区。 ROW_NUMBER()函数分别应用于每个分区，并重新初始化每个分区的行号。PARTITION BY子句是可选的。如果未指定，ROW_NUMBER()函数会将整个结果集视为单个分区。

   ORDER BY子句定义结果集的每个分区中的行的逻辑顺序。 ORDER BY子句是必需的，因为ROW_NUMBER()函数对顺序敏感


   示例

   SELECT 
      ROW_NUMBER() OVER ( ORDER BY first_name) row_num,
      first_name, 
      last_name, 
      city
   FROM 
      sales.customers

常用排名函数
~~~~~~~~~~~~

.. code:: sql


   rank() over()
   1,1,1,4,4,6

   dense_rank() over()
   1,1,1,2,2,2,3,4

   row_number() over()
   1,2,3,4,5

   percent_rank() over()
   百分位排名，表示小于这个值的概率。最小值对应0，最大值对应1，null 会在最后

.. code:: sql

   -- 生成数据测试 percent_rank()对null,以及最大最小值的处理

   select percent_rank() over(order by name) as rank_num from (
   select case 
   when user_score < 3 then 1
   when user_score > 4 and user_score <9 then 4
   when user_score > 16 then 19
   else user_score end as name from (
   select distinct user_score from risk_features.jjplat_rc_user_score_features
   where pt_dt = "2021-08-22"
   order by user_score
   limit 20
   )x
   )x;


   with temp as (
   select uid,recharge_30d from (
   select distinct uid from dw.dw_bal_user_match_stat_d
   where pt_dt="2021-09-18"
   )a
   left join(
   select pid
   ,sum(if(datediff("2021-09-18",pt_dt)<= 30,rmb_amount,0)) recharge_30d
   from security_dwd.dwd_app_acc_payin
   where pt_dt="2021-09-18"
   group by pid
   )b

   on a.uid = b.pid
   )

   ,temp1 as (
   select recharge_30d
   from(
   select recharge_30d,row_number() over( partition by recharge_30d order by recharge_30d) as row_num from temp
   where 
   -- recharge_30d is null or 
   (recharge_30d <= 1500 and recharge_30d >=1000)
   )x
   where row_num <=1
   )


   select recharge_30d
   ,percent_rank() over(order by recharge_30d)
   from temp1
   order by recharge_30d

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211018175419336.png
   :alt: image-20211018175419336

   image-20211018175419336

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211018175319922.png
   :alt: image-20211018175319922

   image-20211018175319922

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211018175216813.png
   :alt: image-20211018175216813

   image-20211018175216813

.. code:: sql

   sum(if(features_giveup !=0,1,0))

   -- 可以直接写为 

   sum(features_giveup !=0)

   -- 对bool值求和时，TRUE=1，FALSE=0
   -- 不能用count，count只会统计有多少非Null的数据，true和false都会被计数

.. _速度解析-1:

速度解析
~~~~~~~~

.. code:: sql

   目前看来 left join / inner join 和 where 差别不大
   但是在哪一步做限制很重要

   比如30天日活和当天日活的交，如果在对局表dw.dw_bal_user_match_stat_d里直接对每一条数据判断是否是当日日活

   这样判断的次数为 对局表30日的数据条数*当日日活数

   如果对数据做完计算，按group by uid,之后再做限制，则判断次数为30日日活数乘以当日日活数
   405359490  *  6436683 降到
    22639872  *  6436683
    
    
    
   此外，行转列时

   sum(if(game_id= 1002,match_time,0)) group by uid,
   if(game_id=1002,sum(match_time),0)  group by uid,game_id

   -- 时间从56/57s直接降到8s
   -- 还需要group by uid, 对列求和




   with temp as (
   select uid,row_number() over (PARTITION BY uid order by mp_id ) mp_id,match_time match_time_day
   from dw.dw_bal_user_match_stat_d
   WHERE pt_dt = "2021-11-01"
   and uid in (538571529,956203992,821539261)
   )

   -- select uid
   -- ,if(mp_id = 1,sum(match_time_day),0)
   -- ,if(mp_id = 2,sum(match_time_day),0)
   -- ,if(mp_id = 3,sum(match_time_day),0)
   -- from temp
   -- group by uid,mp_id
   -- order by uid;


   SELECT uid
   ,sum(if(mp_id =1,match_time_day,0))  mp_1_time
   ,sum(if(mp_id =2,match_time_day,0))  mp_2_time
   ,sum(if(mp_id =3,match_time_day,0))  mp_3_time
   from temp
   group by uid;

::

   建立 erdou_day 时，用到两个中间表
   一是uid 0,1互换，二是整理两个用户对局信息

   最后得出单个用户信息统计
   事实证明，将这两个中间表用with 表示，不如嵌套速度快，with在将查询结果写入表中
   的时候根本跑不出来

   不写入表，只查询的时候嵌套也比with表快，嵌套用了9s左右，with 12s

select 1+1
~~~~~~~~~~

::

   想要验证结果时，可以直接select 1+1
   不用加任何的from table

::

   avg() 对 0 和 null的处理

   null不会贡献分母，而零会

infinity/nan
~~~~~~~~~~~~

::

   0/0 nan
   1/0 infinity

分群重叠人数查看
~~~~~~~~~~~~~~~~

.. code:: sql

   # 对不同群分别标记 1,10,100,1000,10000
   # uid,sum(label) group by uid

   WITH temp as (
   SELECT uid,sum(label) cnt
   from(
       SELECT uid
       ,power(10,operationtype) label
       FROM dwd.dwd_op_msg_receipt_d_v3
       WHERE pt_dt  = '2021-12-07'
       and msgid = 6001540
       GROUP BY uid,operationtype

   )x
   GROUP BY uid
   )

   select cnt,count(DISTINCT uid) FROM temp
   GROUP BY cnt
   ORDER BY cnt

Algorithm
=========

博弈树，alpha-beta剪枝
----------------------

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210929104351801.png
   :alt: image-20210929104351801

   image-20210929104351801

Neo4j
=====

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210811172117899.png
   :alt: image-20210811172117899

   image-20210811172117899

使用 Louvain 算法对二斗用户进行社区发现时，关系建立为 春天的次数或coin,
undirected 比 natural/reverse

在同一个社区能包含更多的label，
但是包含的非label用户也更多，可以通过游戏单一性，vip等级作进一步筛除

::

   注意在使用notebook,py2neo这种需要localhost的服务时，不能打开梯子
   否则直接报错

删除neo4j里，project中的graph
database的时候，会将import中的csv一并删除，be careful

::

   apoc.import.file.enabled=true
   apoc.import.file.use_neo4j_config=true

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211103164914397.png
   :alt: image-20211103164914397

   image-20211103164914397

::

   All data from the CSV file is read as a string, so you need to use toInteger(), toFloat(), split() or similar functions to convert values.

It is generally \*\ **good practice to create some uniqueness
constraints\*** on the nodes to ensure that there are no duplicates. One
advantage of doing so is that this will create an index for the given
label and property. In addition to speeding up query searches for that
data, it will ensure that MERGE statements on nodes (such as the one
used in our load statement) are significantly faster.

::

   CREATE CONSTRAINT ON (a:Player) ASSERT a.uid IS UNIQUE

   DROP CONSTRAINT ON (a:Player) ASSERT a.uid IS UNIQUE

::

   # 查看当前所有indexes & constraints

   :schema

Excel
=====

VLOOKUP
~~~~~~~

.. code:: excel

   =VLOOKUP(A2,$Q$2:$R$13411,2,FALSE)

   四个参数分别为

   查找值，
   查找区域， 请记住，查阅值应该始终位于所在区域的第一列
   返回查找区域中第几列的值，
   近似匹配(TRUE)与精确匹配(FALSE)


   绝对引用与相对引用

   加$$符号的是绝对引用，可以单边绝对引用，表示固定行或列

   remark: 在自动填充时，可以先将第一个参数所在的列复制，(注意不要清除内容)，
           然后将公式写在第一个位置

.. code:: excel

   COUNT(A:A)

单元格换行 Alt+Enter
~~~~~~~~~~~~~~~~~~~~

用户名单推送
============

account
-------

::

   luoxb/luoxb123321

.. _insert-1:

insert
------

.. code:: sql

   INSERT OVERWRITE TABLE operations.op_topic_activity 
   PARTITION(topicid=56, batchid=93, pt_dt = '2021-12-10', classid = 2)
   SELECT cast(UID AS int) UID,
          '' AS extend
   FROM  luoxb.jjmall_promote_activity56_user
   WHERE pt_dt = "2021-12-09"
   AND classid = 2
   AND rnk <= 50000;


   #
   topicid=56 and batchid=91 and pt_dt = '2021-11-26'
   #


   # 商城历次推送人数 各用户群人数

   SELECT pt_dt
   ,sum(classid =1)
   ,sum(classid =2)
   ,sum(classid =3)
   from luoxb.jjmall_promote_activity56_user
   group by pt_dt
   order by pt_dt

caution
-------

.. code:: sql

   [!!!] 工单在 “执行人” 审核完毕后才能在DMP系统中推送名单，否则即使推送了，名单也会无法导入到业务平台中。



   在查询用户群时，今天只能用昨天的pt_dt,

   但写入推送表里的时候，分区必须写今天，才能保证事件分析系统里能看见





   一次推送最好不要超过300万，超过的时候，先写入前300万数据，收到3封邮件之后，
   用后面的数据重写推送表，用同样的分区，再次推送


   推送前和马朝蕾确认一下吧，或者群里@马朝蕾 ，不要提前推送。尤其不要周五或周六晚上推送，
   万一推送出问题，晚上不方便查，
   不在时间内推送，推送失败，数据被丢掉

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211220102847046.png
   :alt: image-20211220102847046

   image-20211220102847046

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211126110743161.png
   :alt: image-20211126110743161

   image-20211126110743161

# Table_Info

SNS
---

sns.sns_dw_member_snapshot_d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   最后一天会包含之前所有的关系


   event_time 毫秒时间戳

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211105182015901.png
   :alt: image-20211105182015901

   image-20211105182015901

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211105182139160.png
   :alt: image-20211105182139160

   image-20211105182139160

sns.dwd_sns_transmsg_d_view
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql



   日志类型：**operation：sns_transmsg**
   字段名称|类型|说明
   :-|:-|:-
   project|string| 数据分析系统使用：项目名称
   event_name|string| 数据分析系统使用：事件名称
   event_time|int|数据分析系统使用：事件时间，单位：毫秒
   uid|int|数据分析系统使用：jj 用户ID[消息发送者]
   sns_type|int|关系类型：参见关系类型(sns_type)含义说明
   sns_pid|int|关系ID：参见关系类型(sns_type)含义说明（当sns_type为1时，为用户ID）
   msg_type|int| 消息类型：1 文案 2 语音 3 图片 4 普通表情 5 特殊表情



   select * from sns.dwd_sns_transmsg_d_view 
   where sns_type = '1' 
   and pt_dt >= '2021-11-01'
   and (uid = '929731696' or sns_pid = '929731696');


   # 数据类型全为string


       name            type               comment
   1   billguid        string  
   2   pt_dt           string  
   3   project         string  
   4   event_name      string  
   5   event_time      string  
   6   uid             string  
   7   sns_type        string  
   8   sns_pid         string  
   9   msg_type        string

推送接收表
----------

dwd.dwd_op_msg_receipt_d_v3
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

       name    type    comment
   1   msgid   int 消息ID
   2   msgtype int 消息类型 1多播，2广播，3生发
   3   uid int 用户ID
   4   operationtype   int 回执类型 1接收，2阅读，3操作
   5   receipttime string  回执返回时间，以达到MSGDetail的时间为准
   6   pt_dt   string


   # 1 接收 2 阅读 3 点击 4 发送


   SELECT *
   FROM dwd.dwd_op_msg_receipt_d_v3
   WHERE pt_dt  >= '2021-12-02'
   and msgid = 6001540
   and uid = 370143050
   order by receipttime

   # 注意去重

       msgid   msgtype uid operationtype   receipttime pt_dt
   1   6001540 1   370143050   4   2021-12-07 19:45:44 2021-12-07
   2   6001540 1   370143050   4   2021-12-07 19:45:44 2021-12-07
   3   6001540 1   370143050   1   2021-12-07 19:45:45 2021-12-07
   4   6001540 1   370143050   1   2021-12-07 19:45:52 2021-12-07
   5   6001540 1   370143050   2   2021-12-07 20:08:50 2021-12-07
   6   6001540 1   370143050   0   2021-12-07 20:08:53 2021-12-07
   7   6001540 1   370143050   2   2021-12-07 21:16:34 2021-12-07
   8   6001540 1   370143050   3   2021-12-07 21:16:36 2021-12-07
   9   6001540 1   370143050   2   2021-12-07 21:16:38 2021-12-07


   7.2.6.已发人数
   MSG消息系统将消息发往接收终端的人数，已发人数≥接收人数。发送率：已发人数/导入数量。
   7.2.7.接收人数
   已经接收到该消息的用户数，已发人数≥接收人数≥已读人数。请注意：接收到消息不一定阅读，不能阅读可能与客户端展示机制或消息展示模式有关。接收率：接收人数/已发人数。
   7.2.8.已读人数
   接收到消息的用户中，客户端将消息展示给多少个用户或多少个用户读取到该消息，多次阅读只计数一次。已发人数≥接收人数≥已读人数≤阅读次数。阅读率：阅读人数/接收人数。

   7.2.9.阅读次数
   客户端将消息展示给用户，用户点击该消息的次数。Snackbar消息、普通弹窗、专题弹窗、动效弹窗等，由于用户只能点击一次，所以已读人数≈已读次数；活动专区挂牌或信箱消息等，用户可能点击多次，所以已读人数≤阅读次数。
   7.2.10.动作点击人数
   已经接收到该消息的用户数，已发人数≥接收人数≥已读人数。请注意：接收到消息不一定阅读，不能阅读可能与客户端展示机制或消息展示模式有关。
   7.2.11.动作点击次数
       动作点击人数做了去重uid的处理，而动作点击次数未做去重uid的处理。

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211123182521488.png
   :alt: image-20211123182521488

   image-20211123182521488

.. code:: sql

   表里会有重复的uid


   SELECT pt_dt
   ,sum(if(operationtype=4,num,0))
   ,sum(if(operationtype=1,num,0))
   ,sum(if(operationtype=2,num,0))
   ,sum(if(operationtype=0,num,0))
   ,sum(if(operationtype=3,num,0))
   from(
   SELECT pt_dt
   ,operationtype
   ,count(distinct uid) num
   FROM dwd.dwd_op_msg_receipt_d_v3
   WHERE pt_dt  > = '2021-12-04'
   and msgid = 6001540
   GROUP BY pt_dt,operationtype
   )x
   GROUP BY pt_dt
   order BY pt_dt;


   SELECT pt_dt
   ,sum(operationtype=4)
   ,sum(operationtype=1) 
   ,sum(operationtype=2)
   ,sum(operationtype=0)
   ,sum(operationtype=3)

   FROM dwd.dwd_op_msg_receipt_d_v3
   WHERE pt_dt  > = '2021-12-04'
   and msgid = 6001540
   GROUP BY pt_dt
   order by pt_dt;

单局表
------

dwd.dwd_game_rec_2ddz
~~~~~~~~~~~~~~~~~~~~~

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210916170923071.png
   :alt: image-20210916170923071

   image-20210916170923071

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210916170854216.png
   :alt: image-20210916170854216

   image-20210916170854216

.. code:: sql

   -- 二斗录像表

   features_giveup 记录的是投降的人的uid, 没有人投降则为0
   features_abort   0代表未流局，1代表流局

.. code:: sql

   注意对录像去重，desc_oid

   当desc_oid 为null时，其余字段也全为null


   由于比赛设置，会出现desc_oid一样，后面的字段不一样的情况
   有的比赛是6轮2副牌，6轮6副牌，一轮是和同一个人打2/6副，就会解析为一个desc_oid

   select distinct desc_mpid, desc_mn from (
   select desc_oid,desc_mpid,desc_mn,count(*) as num from (
   SELECT distinct desc_oid,features_seat1,features_seat0,desc_mpid,desc_mn,desc_rbt,desc_ret-desc_rbt
   ,features_seat_result1,features_seat_result0,features_updatemulti
   ,features_abort
   ,features_giveup
   ,features_spring

   from dwd.dwd_game_rec_2ddz

   where pt_dt = "2021-08-22"
   )x
   group by desc_oid,desc_mpid,desc_mn

   having num >1
   order by num desc
   )x
   order by desc_mpid

   --- 去找金融表里对应的数据，6副牌对应两条数据，分别是积分和金币奖励
   select * from dwd.dwd_bal_user_deal_d
   where pt_dt = "2021-08-22"
   and pid = 340149980
   and deal_subtype_id = 500403

.. code:: sql

   同一个desc_mpid, desc_mn还有可能不一样
   名字还是最后在dim表里去配


   spring_win_uid / spring_win_cnt as unity_spring_win
   group by uid,desc_mpid,desc_mn

   就会出现infinity的情况，spring_win_uid始终大于0，而spring_win_cnt=0

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210917203612753.png
   :alt: image-20210917203612753

   image-20210917203612753

::

   二斗中，地主打一手牌认输会是春天
   农民不能直接认输，至少打一手牌才能认输，这时不是春天

dwd.dwd_usr_match_round_result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   dwd.dwd_usr_match_round_result

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210830101541672.png
   :alt: image-20210830101541672

   image-20210830101541672

event.dwd_usr_match_round_result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210909180530649.png
   :alt: image-20210909180530649

   image-20210909180530649

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210909180600291.png
   :alt: image-20210909180600291

   image-20210909180600291

dwd.dwd_bal_user_deal_d
~~~~~~~~~~~~~~~~~~~~~~~

::

   -- 具体字段意义见数仓说明文档

   记录每次金币变动，时间精确到秒

   包括报名缴费扣除、回兑、连胜奖励、付费表情

   问题是连续打的时候，没有回兑不知道中间发生了什么

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210913101559702.png
   :alt: image-20210913101559702

   image-20210913101559702

dwd.dwd_bal_user_deal_vert_d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

日表
----

dw.dw_bal_deal_balance_d
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   记录用户在各个场次的收支情况
   amount 代表输赢的金币数，cn中大的代表总次数，小的代表赢的次数


   收支类型 operate
   -- 1收入，2支出
   -- 如需求净收入，需用收入数量减去支出数量

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210824173033895.png
   :alt: image-20210824173033895

   image-20210824173033895

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210824173141199.png
   :alt: image-20210824173141199

   image-20210824173141199

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210824174231432.png
   :alt: image-20210824174231432

   image-20210824174231432

dw.dw_bal_user_match_stat_d
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   末尾的_d代表这是日表，只有一天在场次信息的汇总，没有单局信息

   coin_income, coin_pay, match_count, champion_cnt 
   对应于 amount,cn 两列中的四个数
   （对于打一局就结算的比赛是这样，可以打多局的仍需确认）
   （一局结算的也有count部分数据对应不上，需注意）

   通过看自己的数据，

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210824174331000.png
   :alt: image-20210824174331000

   image-20210824174331000

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210824173642408.png
   :alt: image-20210824173642408

   image-20210824173642408

dw.dw_bal_account_balance_d
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   用户账户信息表：记录当日有交易的用户的账户信息

   业务覆盖范围：棋牌类游戏

   currency_id 货币类型 =2 时代表JJ金币
   -- 关联 dim.dim_currency_type 的 id字段
   注意一个用户一天中可能有多条记录

   acc_type账户类型 与 acc_num账户号
   一个pid下可能有多个账户,所以即使是指定了currency_id=2，仍然会有多条记录

   SELECT * from dw.dw_bal_account_balance_d
   where pid = 949734032 and pt_dt = "2021-10-01"


   查询uid下金币最多账户的余额
   select pid
   ,max(balance) as balance
   from dw.dw_bal_account_balance_d
   where pt_dt = "2021-10-01" and currency_id =2
   group by pid

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211020140214391.png
   :alt: image-20211020140214391

   image-20211020140214391

dw.dw_usr_retain_d
~~~~~~~~~~~~~~~~~~

.. code:: sql

   留存

   stat_type = datediff(pt_dt,retain_dt)

   pt_dt 是今天，retain_dt 代表用户那天也有玩游戏

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20211112154040663.png
   :alt: image-20211112154040663

   image-20211112154040663

dwd.dwd_usr_workorder_d
~~~~~~~~~~~~~~~~~~~~~~~

::

   投诉工单

dm.dm_losscallback_width_d
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   # 统计每天 距上次online距今 [15,600] 天的用户，
   # 注意是online,不是游戏

   # loss_days = datediff(pt_dt,last_online_date)

       name    type    comment
   1   uid int 
   2   last_online_date    string  
   3   loss_days   int 
   4   total_recharge  int 
   5   score   int 
   6   qiuka_balance   int 
   7   gold_balance    int 
   8   most_match_exchange string  
   9   qiuka_expend_most   string  
   10  activity_match_if   int 
   11  match_level_pre int 
   12  loss_game_pre   string  
   13  exchange_coin_amount    double  
   14  pt_dt   string

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211202170159553.png
   :alt: image-20211202170159553

   image-20211202170159553

商城
----

::

   商城兑奖查询日期在operations.op_topic_activity表里日期第二天开始，连续三天

   比如12-03推送，查询效果是在12-04到12-06

dwd.dwd_jjmall_order_info_new
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   1   orserialid  int 序号 自增
   2   orderid string  订单ID        #
   3   uid int 用户ID        # 
   4   membermark  int 会员标识（0：非会员，1：会员）
   5   viplevel    int 会员等级 

   6   storeid int 店铺ID  #

   7   storename   string  店铺名称
   8   terminaltype    int 终端类型
   9   ostype  int 操作系统类型
   10  devtype int 设备类型
   11  deviceos    string  操作系统和版本号
   12  devicehardware  string  设备厂商和型号
   13  zoneid  int 下单入口信息
   14  addressprovince string  收货人省
   15  addresscity string  收货人市
   16  addresscounty   string  收货人县
   17  addresstown string  收货人镇

   18  commodityform   int 商品形态（2：实物，3：数字产品）  #

   19  commoditysubform    int 商品子形态（0：无子形态，1：话费）
   20  relserialid string  业务流水号
   21  orderfreight    int 订单运费金额（单位：分）
   22  expectdate  string  期望送达日期


   23  orderstate  int 订单状态        #
   24  orderresult int 订单结果，如果订单失败，表示失败原因      #


   25  ordertime   string  下单时间
   26  finishpaymenttime   string  支付完成时间
   27  examineflag int 是否审核（0：不审核，1：审核）
   28  procedureid int 步骤标识（0：审核未完成，1：审核完成，2：待向仓储确认下单，3：向仓储确认下单中，4：向仓储确认下单完成，5：待向采购推单，6：向采购推单中，7：采购推单完成）
   29  spitnum int 拆单数量
   30  ordercommcount  int 订单商品数量
   31  userinvisiable  int 用户不可见标识，0：可见；1：不可见
   32  updatereason    string  订单状态修改原有
   33  useractivity    int 用来标记用户参与的活动
   34  status  int 数据状态（0：有效，1：无效）
   35  ctime   string  创建时间     #
   36  mtime   string  修改时间
   37  note    string  订单备注信息
   38  pt_dt   string

.. code:: sql

   with temp as (
   select a.orderid
   ,a_label
   ,b_label
   from(
       select distinct orderid
       , 1 as a_label
       from dwd.dwd_jjmall_order_info_new
       where pt_dt = '2021-12-01'
   )a

   full join(
       select distinct orderid
       , 2 as b_label
       from dwd.dwd_jjmall_sub_order_info_new
       where pt_dt = '2021-12-01'
   )b

   on a.orderid = b.orderid

   )

   select count(distinct orderid),sum(a_label is null),sum(b_label is null) from temp



       count(distinct orderid) sum(a_label is null)    sum(b_label is null)
   1   62173510    1185    5393873

   by Kimmie : 是否拆分子订单是根据供货商，存在没有子订单的情形

dwd.dwd_jjmall_sub_order_info_new
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

count(suborderid) = count(orderid)

可能是共同作为主键

.. code:: sql

   1   suborderid  string  子订单ID
   2   orderid string  订单ID OrderInfo.OrderID
   3   uid int 用户ID
   4   membermark  int 会员标识（0：非会员，1：会员）
   5   viplevel    int 会员等级
   6   storeid int 店铺ID     ##
   7   storename   string  店铺名称
   8   terminaltype    int 终端类型
   9   ostype  int 操作系统类型

   17  commodityform   int 商品形态（2：实物，3：数字产品）      ##
   18  commoditysubform    int 商品子形态（0：无子形态，1：话费）

   22  orderstate  int 订单状态
   23  orderresult int 订单结果，如果订单失败，表示失败原因

   35  pt_dt   string  


   storeid:店铺ID
   0: 大厅店铺
   1：消消店铺
   2：qa店铺
   3：曙光店铺
   4：春秋店铺

.. code:: sql

   SELECT orderstate,count(DISTINCT orderid)
   FROM dwd.dwd_jjmall_sub_order_info_new
   WHERE pt_dt = '2021-12-03'
   GROUP BY orderstate
   ORDER BY orderstate;

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211204114843821.png
   :alt: image-20211204114843821

   image-20211204114843821

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211204120039538.png
   :alt: image-20211204120039538

   image-20211204120039538

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211204160519987.png
   :alt: image-20211204160519987

   image-20211204160519987

orderid 与 suborderid

按子订单个数对订单分组 去重计数

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211204115901772.png
   :alt: image-20211204115901772

   image-20211204115901772

dwd.dwd_jjmall_order_comm_info_new
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql


       name    type    comment
   1   ordercomminfoid int 订单商品明细ID
   2   orderid string  订单ID OrderInfo.OrderID
   3   suborderid  string  子订单ID SubOrderInfo.SubOrderID
   4   commodityid string  商品SKU
   5   commodityname   string  商品名称
   6   commodityssku   string  供应商SKU
   7   commodityspec   string  商品规格
   8   commodityunit   string  商品销售单位
   9   commoditytaxtype    string  商品税别
   10  commoditytaxrate    string  商品税率
   11  commodityu8code string  商品U8编码
   12  commodityform   int 商品形态（2：实物，3：数字产品:）


   13  commsubform int 商品子形态（0：无子形态，1：话费）     ## 

   14  commoditycount  int 商品数量

   15  facevalue   int 面额（单位：分）      # 商品单价


   16  commoditytype   int 商品类型（1：商品，2：赠品、3 基底商品）
   17  commodityfreight    int 商品承担运费金额（单位：分，注：赠品不承担运费）
   18  commodityweight int 商品重量
   19  supplierid  int 供应商ID
   20  supplierorderno string  供货商采购单号
   21  waybill string  物流单号
   22  returnstate int 标识是否退货（0：没退，1：已退）
   23  status  int 数据状态（0：有效，1：无效）
   24  ctime   string  创建时间（固定）
   25  mtime   string  修改时间（固定）
   26  note    string  备注信息
   27  pt_dt    string

.. code:: sql

   一个orderid 下面 suborderid 里所有商品的commodityform 一定是一样的

   SELECT orderid,count(DISTINCT commodityform) cnt from 
   dwd.dwd_jjmall_order_comm_info_new
   where pt_dt = '2021-12-01'
   GROUP BY orderid
   ORDER BY cnt desc


   一个suborderid 可能对应多行数据，commodityname 可以有多种

   SELECT suborderid, count(DISTINCT commodityname) cnt
   from dwd.dwd_jjmall_order_comm_info_new
   where pt_dt = '2021-12-01'
   GROUP BY suborderid
   ORDER BY cnt desc

       suborderid  cnt
   1   NULL    2937
   2   102110152055275887289501    27
   3   102111201050156121503801    25
   4   102109201010475691140801    25
   5   102110281438585971273201    21
   6   102104011901304532405401    20
   7   102108272339305515561101    20
   8   102006221945371795285602    20
   9   102111170454516100277401    20
   10  102104261331104777019501    20



   with temp as (
   select 
   b1.orderid
   ,b1.suborderid
   ,directiontype
   ,b2.ctime
   ,b1.commodityform
   ,commodityname
   ,facevalue
   ,oriid
   ,currencyname
   ,singleprice
   ,b2.commoditycount
   ,originalamount
   ,reduceamount
   ,dealamount
   from (
       SELECT * 
       FROM dwd.dwd_jjmall_order_comm_info_new
       where pt_dt = '2021-12-01'
   )b1


   left join (
       SELECT * 
       FROM dwd.dwd_jjmall_order_comm_price_info_new
       where pt_dt = '2021-12-01'
   )b2

   on b1.orderid = b2.orderid
   and b1.commodityid =b2.commodityid

   )

   select * from temp
   where orderid = '102104011901304532405400'
   order by suborderid



       singleprice commoditycount  originalamount  reduceamount    dealamount
       2300        2                   4600    600     4000
       1000        2                   2000    200     1800
       3900        2                    7800   1000    6800
       1800        2                   3600    200     3400
       5400        1                   5400    600     4800
       2400        1                   2400    200     2200
       
       singleprice * commoditycount = originalamount
       
       originalamount - reduceamount = dealamount
       
       
       suborderid                      ctime           commodityname   facevalue   oriid   currencyname    
   102104011901304532405401    2021-04-01 19:01:30.413 雀巢罐装中老年奶粉850g   9290    889 秋卡
   102104011901304532405401    2021-04-01 19:01:30.413 雀巢罐装中老年奶粉850g   9290    100 人民币
   102104011901304532405401    2021-04-01 19:01:30.413 金龙鱼物理压榨葵花籽油6.18L    9250    889 秋卡
   102104011901304532405401    2021-04-01 19:01:30.413 金龙鱼物理压榨葵花籽油6.18L    9250    100 人民币
   102104011901304532405401    2021-04-01 19:01:30.43  柴火大院稻花香米东北大米5kg 5600    889 秋卡
   102104011901304532405401    2021-04-01 19:01:30.43  柴火大院稻花香米东北大米5kg 5600    100 人民币
   102104011901304532405402    2021-04-01 19:01:30.43  JJ双层高硼硅玻璃茶杯升级款  3600    889 秋卡
   102104011901304532405402    2021-04-01 19:01:30.43  JJ双层高硼硅玻璃茶杯升级款  3600    100 人民币
   102104011901304532405402    2021-04-01 19:01:30.43  JJ泡茶师高硼硅分离玻璃杯升级款    4250    889 秋卡
   102104011901304532405402    2021-04-01 19:01:30.43  JJ泡茶师高硼硅分离玻璃杯升级款    4250    100 人民币
   102104011901304532405403    2021-04-01 19:01:30.413 JJ无芯卷纸3层*100g*12卷   1790    889 秋卡
   102104011901304532405403    2021-04-01 19:01:30.413 JJ无芯卷纸3层*100g*12卷   1790    100 人民币
   102104011901304532405403    2021-04-01 19:01:30.413 如水鱼皮花生500g  2790    889 秋卡
   102104011901304532405403    2021-04-01 19:01:30.413 如水鱼皮花生500g  2790    100 人民币




   一个orderid 是否生成suborderid是根据供货商的不同
   一般orderid 末两位是00， suborderid末两位从01开始
   102003161028570975617400    102003161028570975617401

   一个suborderid 也会有多行记录，对应不种货币

   秋卡+人民币的购买方式，依商品而定，没有固定的 秋卡/人民币  转换汇率
   同样数量的秋卡，在不同的商品中折算的人民币数量不一样

dwd.dwd_jjmall_order_comm_price_info_new
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   一个suborderid 也会有多条记录，对应不同货币支付金额


   有1/10的suborderid 为 null


       name                    type                comment
   1   ordercommpriceid         int                订单商品货币价格ID
   2   orderid                 string              订单ID或子订单ID OrderInfo.OrderID
   3   ordercomminfoid          int                订单商品明细ID OrderCommDetail.OrderCommInfoID
   4   suborderid              string               子订单ID SubOrderInfo.SubOrderID
   5   commodityid             string              商品SKU


   6   directiontype           int                 方向，-1：支付，1：赠送


   7   dtid                    int                 数据D
   8   atid    int 数据A


   9   oriid   int 数据O
   10  currencyname    string  数据名称


   11  dataid  string  业务方支付数据ID
   12  singleprice int 单价（原价）
   13  commoditycount  int 商品数量
   14  originalamount  int 总价（原价）
   15  reduceamount    int 优惠减掉的价格
   16  dealamount  int 交易价
   17  status  int 数据状态（0：有效，1：无效）
   18  ctime   string  创建时间（固定）
   19  mtime   string  修改时间（固定）
   20  note    string  备注信息
   21  pt_dt   string

.. code:: sql

dwd.dwd_jjmall_order_payment_info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

日表，不是全量

::

   1   orderpaymentinfoid  int 订单支付明细ID
   2   orderid string  订单ID
   3   directiontype   int 方向，-1：支付，1：赠送
   4   paymenttimes    int 支付次数
   5   dtid    int 数据D
   6   atid    int 数据A
   7   oriid   int 数据O
   8   currencyname    string  数据名称
   9   dataid  string  业务方支付货币ID
   10  belonging   int 数据归属，1：平台；2：业务
   11  originalamount  int 原价
   12  reduceamount    int 优惠减掉的价格
   13  dealamount  int 交易价
   14  headpaymentamount   int 首付应付数量
   15  tailpaymentamount   int 尾款应付数量
   16  payedamount int 已支付数量
   17  paymentchannel  int 支付渠道
   18  paymentmethod   string  支付方式
   19  paymentmode int 支付模式
   20  bankid  int 银行类型
   21  ecaid   int 订单执行的ECA方案ID
   22  ecagroupid  int 订单执行的ECA方案分组ID
   23  status  int 数据状态（0：有效，1：无效）
   24  ctime   string  创建时间（固定）
   25  mtime   string  修改时间（固定）
   26  note    string  订单商品备注信息
   27  pt_dt   string

售后表
~~~~~~

.. code:: sql

   SELECT * from dwd.dwd_jjmall_aftersale_commodity_dao_new;
   SELECT * from dwd.dwd_jjmall_aftersale_commodity_info_new;
   SELECT * from dwd.dwd_jjmall_aftersale_order_dao_new;
   SELECT * from dwd.dwd_jjmall_aftersale_order_new;

信息维度表
----------

dim.dim_product_info_d
~~~~~~~~~~~~~~~~~~~~~~

::

   dim.dim_product_info_d

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210827104737636.png
   :alt: image-20210827104737636

   image-20210827104737636

dim.dim_ware_info_v2 ( 物品维表)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   SELECT * from (
       SELECT DISTINCT oriid,currencyname
       from dwd.dwd_jjmall_order_comm_price_info_new
       where pt_dt = '2021-11-17'
   ) t1
   left join dim.dim_ware_info_v2 t2 
   on t1.oriid = t2.type_id
   ORDER BY oriid

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211124110740925.png
   :alt: image-20211124110740925

   image-20211124110740925

::


       oriid   currencyname
   1   2       金币
   2   88      福卡
   3   100     人民币
   4   505     手机话费直充50元
   5   506     手机话费直充30元
   6   510     手机话费充值百元兑换券
   7   889     秋卡
   8   889     秋卡抵扣
   9   1741    10元手机充值卡
   10  3118    100元京东兑换券
   11  3142    小海豚蓝(专用）
   12  3142    小海豚蓝
   13  3186    1元手机充值卡
   14  3187    5元手机充值卡
   15  3307    2元手机充值卡

dim.dim_currency_type
~~~~~~~~~~~~~~~~~~~~~

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211124112958322.png
   :alt: image-20211124112958322

   image-20211124112958322

dim.dim_company_area_code
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   SELECT prov_id,prov_name from dim.dim_company_area_code
   GROUP BY prov_id,prov_name
   ORDER BY prov_id


   SELECT * from dim.dim_company_area_code
   where country_id = 511325



       prov_id  prov_name  city_id  city_name  country_id  country_name
   1   510000   四川省     511300   南充市      511325       西充县

dim.dim_company_prov_code
~~~~~~~~~~~~~~~~~~~~~~~~~

::

   与上面的区别是只有31个省、市、自治区，不含港澳台

用户登录注册表
--------------

dwd.dwd_usr_register_d
~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   select uid,pt_dt from dwd.dwd_usr_register_d
   group by uid,pt_dt

   一个uid可能有不只一条记录

dwd.dwd_usr_online_v2_d
~~~~~~~~~~~~~~~~~~~~~~~

::

   dwd.dwd_usr_online_v2_d


   注意一个用户一天可能通过多个平台登录

   clientplatid 平台id


   -- 可以用于区分各个平台的登录，
   TK_PTID_ENUM_RESERVE = 0,    //保留
   TK_PTID_ENUM_MOBILE  = 1,    //手机大厅
   TK_PTID_ENUM_FLASH   = 2,    //flash（停用）
   TK_PTID_ENUM_LOBBY   = 3,    //PC大厅
   TK_PTID_ENUM_WEB  = 4,    //JJ官网
   TK_PTID_ENUM_WEBGAME = 5,     //webgame，网页游戏，对应的appid为70万+
   TK_PTID_ENUM_OPENID = 6,     //OPENID
   TK_PTID_ENUM_SNSID = 7,     //海豚
   TK_PTID_ENUM_MLOAD = 8,     // MLoad （没用）
   TK_PTID_ENUM_JJSDK = 9,     // JJSDK，联运或自研游戏
   TK_PTID_ENUM_JJBOX = 10,    // JJBOX
   TK_PTID_ENUM_WXGAME = 11,    // 微信游戏平台
   TK_PTID_ENUM_H5GAME = 12,    // 美团游戏平台
   TK_PTID_ENUM_PROMOTE = 13,    // 推广产品平台
   TK_PTID_ENUM_DYGAME = 14,     // 抖音游戏平台
   TK_PTID_ENUM_LONGHU = 15,     // 龙湖小程序平台
   TK_PTID_ENUM_DIDIGAME = 16,     // 滴滴游戏

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210830163800573.png
   :alt: image-20210830163800573

   image-20210830163800573

dw.dw_usr_game_last_i
~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   -- 拉链表，保存全量用户的状态，将用户状态切分为 “开始日期 ---> 结束日期” 和 “开始 ---> 2099-12-31”

   -- 注意这里的pt_dt 只到月 

   -- 目的是为了对数据做分区，查询的时候指定分区，避免全表扫，加快查询速度

   -- 全量表，每天都会更新用户当前时间在各个mp_id最后玩的情况

   -- 同月内没有变化，pt_dt 不更新不会增加新的行，跨月还没有变化才会增加行


   -- 2021-11-30 之前玩过1094的用户


   select distinct  uid
   from dw.dw_usr_game_last_i
   where pt_dt = '2021-11'
   and s_date <= '2021-11-30'
   and e_date >= '2021-11-30'
   and game_id=1094

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210901145819192.png
   :alt: image-20210901145819192

   image-20210901145819192

.. code:: sql


   select b.mp_name ,a.* from(
   select mp_id
   ,match_begin_date
   ,match_time
   ,s_date
   ,pt_dt
   ,e_date
   from dw.dw_usr_game_last_i
   where pt_dt = '2021-11'
   and s_date <= '2021-11-05'
   and e_date >= '2021-11-05'
   and uid =929731696
   )a

   left join(
   select mp_id,mp_name 
   from dim.dim_product_info_d
   )b on a.mp_id = b.mp_id
   order by 
   s_date

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211207160212486.png
   :alt: image-20211207160212486

   image-20211207160212486

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\1.jpg
   :alt: 1

   1

::

   match_begin_time   |   mp_id   |   s_date  |    e_date   |    pt_dt

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210902142545915.png
   :alt: image-20210902142545915

   image-20210902142545915

.. figure:: C:\Users\luoxb\AppData\Roaming\Typora\typora-user-images\image-20210902142847656.png
   :alt: image-20210902142847656

   image-20210902142847656

dwd.dwd_usr_login_d ( nickname)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   SELECT * from dwd.dwd_usr_login_d 
   where uid = 929731696
   and pt_dt >= '2021-07-01'
   ORDER BY pt_dt desc

.. figure:: C:\Users\luoxb\Downloads\学习资料\笔记markdown\code.assets\image-20211201154821467.png
   :alt: image-20211201154821467

   image-20211201154821467

.. code:: sql

   name    type    comment
       name    type    comment
   1   dwaais2govv string  登录时间           
   2   uid int 用户ID             
   3   sznickname  string  昵称               
   4   dwcoin  int 元宝数量           
   5   dwbonus int 奖券数量           
   6   dwgold  int 金币数量           
   7   dwcert  int 代金券             
   8   dwscore int 总积分             
   9   dwmasterscore   int 总大师分           
   10  dwsafegold  int 保险箱货币         
   11  dwluckcard  int 幸运卡数量         
   12  dwpfsct int 用户平台分类       
   13  ullmac  string  
   14  dwlocalip   bigint  内网IP             
   15  dwinternetip    bigint  外网IP             
   16  dwterminertype  int 终端类型           
   17  dwuitype    int UI类型             
   18  dwostype    int 操作系统类型       
   19  sztcompany  string  终端所属公司       
   20  sztunittype string  终端型号           
   21  sztcode1    string  设备号             
   22  sztcode2    string  设备号             
   23  sztcode3    string  设备号             
   24  sztcode4    string  设备号             
   25  dwplatid    int 平台ID             
   26  dwsiteid    int 媒体ID(Agent)      
   27  dwacctype   int 帐号类型           
   28  dwlogintime string  登录时间           
   29  dwareaid    int 用户区域           
   30  dwproductid int 产品ID或者叫游戏ID 
   31  dwparam3    string  字段含义未知       
   32  dwcmtsid    string  组委会方案ID       
   33  logon_time  string  dwaais2govv转换日期
   34  outnet_prov string  外网省份           
   35  outnet_city string  外网地市           
   36  outnet_isp  string  外网运营商         
   37  nickname_type   int 昵称类型           
   38  internetip  string  转dwinternetip     
   39  pt_dt   string  

标签表
------

tags.tag_log_local_tostype_b [最常用终端]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sql

   SELECT DISTINCT UID
   FROM tags.tag_log_local_tostype_b
   WHERE TYPE = 's'
   AND pt_dt = '9999-12-31'
   AND log_local_tostype = 2


   系统标识    字典含义
   0   保留
   1   Windows
   2   Android
   3   IOS
   4   WindowsPhone
   5   Android TV
   6   安卓模拟器
   7   苹果模拟器
   -9999   异常

::

   # password

   LDAP
   q she/me LL
