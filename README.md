[TOC]

# 基于Django框架的口罩识别WEB端实现

## 小组成员

- 组长：哈斯巴特
- 组员：王睿潇、杨鑫、施皓天



## 先看这个

- 本项目用于四川大学本科课程作业，并不适用于其他商业用途
- 有问题可联系QQ：847573508
- 本文件夹中只包含识别模块代码，不包含训练模块（数据集过大）

## 系统概述

- 现如今疫情形势严峻，需要全民佩戴口罩来缓解形势，在公共场合中需要对戴口罩进行检测。而传统人员检测是一种枯燥单调的重复性劳作，这样的工作效率是很低的，而且人流量大的时候，也容易忽视个体，造成工作上的疏忽。所以需要智能口罩检测系统来代替检测，避免人员聚集同时保持高效率。基于现如今疫情形势及社会需求，开发了本系统。



## 运行系统

- 可在主流操作系统Windows上运行，在Windows10上测试通过
- 后续已移植到Mac OS上，但需要更改部分代码，本文件夹中代码运行环境为Windows10



## 环境配置

- 项目制作选择了 Anaconda 的 root 环境， 确保python环境中安装了django、tensorflow、numpy、opencv、pymysql、threading、winsound、os、time、smtplib、email等必要包



## 操作流程

- 项目选择mysql作为数据库，该项目代码为本机运行，需要本机打开mysql服务，并建立对应user表，并更改数据库连接信息（在views.py中）

- 项目运行时在终端打开**mask__detect**文件夹(注：project/mask_detect )，然后在终端输入以下代码打开Django的WEB服务 确保文件夹不要进错 manage.py隶属于该文件夹

  ``` 
  python manage.py runserver 0.0.0.0:8000
  ```

- 服务运行成功后在浏览器中输入 127.0.0.1即可进入登陆界面

- 进入登录页面以后，可以选择新用户注册、老用户找回密码以及正常登录功能

  > - 新用户注册功能
  >   - 要求输入账号、密码与邮箱并将数据存到数据库中
  > - 老用户找回密码功能
  >   - 要求输入邮箱（会由本人邮箱847573508@qq.com发送验证码）在网页中输入与之相符的验证码进行确认 确认成功会创建界面提示账号密码
  > - 正常登录功能
  >   - 输入用户名与密码后，与数据库信息进行验证、验证成功后进入主页面

- 进入index主界面后有两个功能，获取视像头视频与下线功能

  > - 获取摄像头视频功能
  >
  >   - 点击获取摄像头视频按钮 会调用 project/mask_detect/detect_mask_video.py,在系统中（非本网页）生成调用摄像头并实时返回处理后视频流的窗口，在窗口中可以识别是否佩戴口罩，若未佩戴口罩则会有系统提示音进行报警
  >
  >   - 键盘输入 'q' 或者 'Q'即可退出识别窗口
  >
  > - 下线功能
  >
  >   - 点击下线按钮会清除登录信息并回到登陆界面

  