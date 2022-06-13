# Python与MySQL数据库的交互

- [1. 导入PyMySQL库](#1-导入pymysql库)
- [2. 使用Python连接MySQL数据库](#2-使用python连接mysql数据库)
- [3. Cursor游标对象的一些常用方法](#3-cursor游标对象的一些常用方法)
  - [3.1. 创建表](#31-创建表)
  - [3.2. 添加数据](#32-添加数据)
  - [3.3. 查询表格](#33-查询表格)
- [4. SQL注入](#4-sql注入)
- [5. 排序](#5-排序)
- [6. 删除记录](#6-删除记录)
- [7. 更新表格](#7-更新表格)

## 1. 导入PyMySQL库

```python
import pymysql
```

## 2. 使用Python连接MySQL数据库

-- 参数host：mysql服务器所在的主机的ip；
-- 参数user：用户名
-- 参数password：密码
-- 参数port：连接的mysql主机的端口，默认是3306
-- 参数database：连接的数据库名
-- 参数charset：当读取数据出现中文会乱码的时候，需要我们设置一下编码

```python
mydb = pymysql.connect(
  host="localhost", #默认用主机名
  user="root",  #默认用户名
  password="123456"   #mysql密码
  ,charset='utf8',  #编码方式
  database="xdd",
)
print(mydb)
```

 <pymysql.connections.Connection object at 0x0000023AEA89A148>  

## 3. Cursor游标对象的一些常用方法

1）cursor用来执行命令的方法
execute(query, args)：执行单条sql语句，接收的参数为sql语句本身和使用的参数列表，返回值为受影响的行数；
executemany(query, args)：执行单条sql语句，但是重复执行参数列表里的参数，返回值为受影响的行数；
2）cursor用来接收返回值的方法
fetchone()：返回一条结果行；
fetchmany(size)：接收size条返回结果行。如果size的值大于返回的结果行的数量，则会返回cursor.arraysize条数据；
fetchall()：接收全部的返回结果行；

### 3.1. 创建表

```python
mycursor = mydb.cursor()
mycursor.execute("CREATE TABLE user (name VARCHAR(255), address VARCHAR(255))")
mycursor.execute("SHOW TABLES")
for x in mycursor:
    print(x)
```

 ('user',)

### 3.2. 添加数据

```python
# 添加单行数据
sql = "INSERT INTO user (name, address) values (%s,%s)"
val = ("丹丹", "江科大")
mycursor.execute(sql,val)
mydb.commit()
print(mycursor.rowcount,"添加表格成功.")
```

1 添加表格成功.

```python
#添加多行数据
sql = "INSERT INTO user (name, address) VALUES (%s, %s)"
val = [('牛魔王',9000),('铁扇公主',8000),('玉皇大帝',6000)]
mycursor.executemany(sql, val)
mydb.commit()
print(mycursor.rowcount, "全部添加成功.")
```

 3 全部添加成功.

### 3.3. 查询表格

```python
#查表，返回所有
mycursor.execute("select * from user")
myresult = mycursor.fetchall()
for x in myresult:
    print(x)
```

('丹丹', '江科大')
    ('牛魔王', '9000')
    ('铁扇公主', '8000')
    ('玉皇大帝', '6000'）

```python
#返回结果第一行
mycursor.execute("select * from user")
myresult = mycursor.fetchone()
print(myresult)
```

 ('丹丹', '江科大')

## 4. SQL注入

通过参数化,就可以防止sql注入. 我们将参数的拼接交给直接交给execute,而不是自己拼接好了再交给execute. 这就是sql语句的参数化.如果execute发现有参数, 他内部就会做 防止sql注入。

```python
#防止SQL注入
sql = "select * from user where address = %s"
adr = ("8000",)
mycursor.execute(sql,adr)
myresult = mycursor.fetchone()
print(myresult)
```

('铁扇公主', '8000')

## 5. 排序

```python
#升序ASC逆序DESC
mycursor.execute("select * from user order by name DESC")
myresult = mycursor.fetchall()
for x in myresult:
    print(x)
```

 ('铁扇公主', '8000')
    ('玉皇大帝', '6000')
    ('牛魔王', '9000')
    ('丹丹', '江科大')

## 6. 删除记录

```python
#删除记录
mycursor.execute("delete from user where address = '6000'")
mydb.commit()
print(mycursor.rowcount,"记录已删除")
```

1 记录已删除

## 7. 更新表格

```python
#更新表
mycursor.execute("update user set address = 'Canyon 123' where address = 'Valley 345'")
mydb.commit()
print(mycursor.rowcount,"更新成功")
```

0 更新成功
