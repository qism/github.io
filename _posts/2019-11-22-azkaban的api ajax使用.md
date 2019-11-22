---
layout:     post
title:      azkaban用api ajax实现一键打包上传执行流
subtitle:   指定flow parameter只能UI方式？
date:       2019-11-22
author:     qism
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:    
        - 任务调度工具
        - azkaban
---

# 一、azkaban

任务调度工具，任务定时，任务依赖什么的都能轻松搞定，具体实在不想抄了，小伙伴自己百度吧~~

常规使用流程是：建立job->添加依赖->增加配置文件（比如邮件监控）->打包->上传->schedule

本文记录两个小冰工作中碰到的问题:

	1、每次有更改，需要UI删除包，重新打包，上传，schedule，是否能自动化？

	2、如果是muti-executor的情况下，az会调度执行节点，也就是执行多次，执行节点可能不同。若需要指定执行节点，除了UI指定flow parameter（手动，不影响流全局），还能采取什么方式？

上述两个问题都可以用az的***api ajax***解决
-----------------------------------------------------------

# 二、自动化

实现一键打包，构建project，上传，执行

```sh
host=http://azkaban.xxx.com
username="xx"
password="xxx"
project="az_task_bi_test"
description="@Brief: azkaban测试  @Owner: iceberg"

if [[ $# -ge 1 ]]; then
	project=$1
fi

## 已经存在则先删除
if [[ -f az_task_bi_test.zip ]]; then
	rm -rf az_task_bi_test.zip
fi
## 压缩 job  
## 我的测试文件是以dw、biz命名的，熟悉数仓的小伙伴应该很清楚；配置文件中只写了邮件地址
zip ./az_task_bi_test.zip dw/* biz/* *.job notifyemails.properties

## 申请 session_id
session_id=`curl -s -k -X POST --data "action=login&username=${username}&password=${password}" ${host} | \

python -c 'import json,sys; data = json.load(sys.stdin); print (data["status"]=="success" and data["session.id"] or "error");'`

## 是否获取 session_id 失败
if [[ "error" = "$session_id" ]]; then
	echo "登录失败"
	exit
fi

#首先删除同名的项目
curl -k --get --data "session.id=${session_id}&delete=true&project=${project}" ${host}/manager

#新建一个project，用Python解析输出结果
curl -k -X POST --data "session.id=${session_id}&name=${project}&description=${description}" ${host}/manager?action=create | \

python -c 'import json,sys; data = json.load(sys.stdin); print (data.get("error") and "error: " + data["error"] or data.get("status"));'

#上传项目，用Python解析输出结果
curl -s -k -H "Content-Type: multipart/mixed" -X POST --form 'session.id='${session_id} --form 'ajax=upload' --form 'file=@az_task_bi_test.zip;type=application/zip' --form 'project='${project} ${host}/manager | \
python -c 'import json,sys; data = json.load(sys.stdin); print (data.get("error") and "error: " + data["error"] or data);'
```

# 三、指定执行节点

因为UI 指定flow parameter的方式不会影响流的全局参数，也就是说是个临时的参数，那么如果需要一个全局参数，但是又想schedule，怎么办？

第一个想尝试的方法是在配置文件中，非常遗憾，小冰看了很久的文档，az不支持

第二个是想尝试的在执行时限定条件，az不支持

第三个是模拟UI操作，太复杂

第四个，api ajax，其实和第三个本质是一样的

小冰碰到的问题是，想要某个任务指定在某个节点跑，直接上代码：


```sh
host=http://azkaban.xxx.com
project="az_task_bi_test"
username="xxx"
password="xx"
useExecutor=10  #执行节点 按照需求来，我这边只是举个例子

## 申请 session_id
session_id=`curl -s -k -X POST --data "action=login&username=${username}&password=${password}" ${host} | \

python -c 'import json,sys; data = json.load(sys.stdin); print (data["status"]=="success" and data["session.id"] or "error");'`

## 是否获取 session_id 失败
if [[ "error" = "$session_id" ]]; then
	echo "登录失败"
	exit
fi

#获取flow_name
flow_name=`curl -k --get --data "session.id=${session_id}&ajax=fetchprojectflows&project=${project}" ${host}/manager | \

python -c 'import json,sys; data = json.load(sys.stdin); print (data["flows"][0]["flowId"] or "error");'`

if [[ "error" = "$flow_name" ]]; then
	echo "获取流失败"
	exit
fi

#执行流
curl -k --get --data 'session.id='${session_id} --data 'ajax=executeFlow' --data 'flowOverride[useExecutor]='${useExecutor} --data 'project='${project} --data 'flow='${flow_name} ${host}/executor
```

还有添加schedule，获取流执行情况等等很多操作，具体可以参考azkaban的官方文档

over~~


