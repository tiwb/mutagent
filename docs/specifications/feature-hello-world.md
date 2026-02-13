接下来准备进行mutagent的初次真实启动，我需要：
1. 允许通过python -m mutagent来启动mutagent， 启动后会询问用户输入，输入完成后agent给出反馈。然后循环。
2. main支持读取环境变量中的ANTHROPIC_BASE_URL，ANTHROPIC_AUTH_TOKEN，ANTHROPIC_MODEL，也支持读取mutagent.json下的配置，mutagent.json下配置env的方法同claude code。
3. 给出合理的系统提示词，让agent知道自己的定位跟使命，我会尝试让它开始自我迭代进化。