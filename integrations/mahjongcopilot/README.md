# MahjongCopilot 本地兼容改动

独立宿主源码来自 https://github.com/latorc/MahjongCopilot ，基线为
`31be3de`。根目录的 `MahjongCopilot/` 是用户本地独立 checkout，不纳入此仓库。

本目录的 [补丁](s70-compatibility.patch) 完整保存本次同步时该 checkout 的
`bot/local/engine.py`、`bot/local/model.py`、README 及新增加载测试改动。
内容涉及 GN、categorical policy 和既有模型接口兼容；不是宿主完整运行验收。

在上述基线的独立干净 checkout 中先执行 `git apply --check`，再按需应用补丁。
本次只导出补丁，不修改、重置或提交用户本地的独立宿主。
许可证沿用上游，宿主依赖和原生扩展按其 README 准备。
