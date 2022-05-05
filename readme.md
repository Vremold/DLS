## Development Library Search System
这篇工作主要是从现有的前端开源仓库中挖掘结构化的功能语义信息和非结构化的功能语义信息，并基于此面向用户query提供检索服务
### 数据准备
本工作爬取了NPM网站中所有可见的仓库信息，以及它们在GitHub网站中的信息，包括readme信息。爬取NPM网站使用Python requests库完成，总共爬取41166个仓库。（当然，这些信息是不全的，因为顺着NPM的目录页面向下爬取，到一定深度后NPM网站便不返回数据了【笑】），GitHub网站的相关信息使用GitHub Restful API完成爬取。数据已经经过基本清洗并放在Data目录下，看看pathutil.py文件应该就知道是怎么存放的了
### 结构化功能语义信息挖掘
本工作使用