# -*- coding: utf-8 -*-
"""
billing 注入补丁 —— 给公司 mcli 网关用的透明适配层。

为什么要这个文件？
  公司内部 Claude 网关（mcli.sankuai.com）要求每个请求的 system prompt 里
  必须带一段 billing 标识，否则返回 "Requests are not allowed"。
  本文件在 SDK 层自动把这段标识塞进 system，示例脚本本身不用改、保持干净。

用法：
  python -m examples.python._billing_patch <要运行的脚本路径>
  例如：python -m examples.python._billing_patch examples/python/01_hello_claude.py

类比：就像给你的电话卡自动加拨一个运营商前缀，你照常拨号就行，前缀在后台帮你加。
"""
from anthropic import NotGiven
from anthropic.resources import messages as _msgs

BILLING = "x-anthropic-billing-header: cc_version=2.1.117.bf6; cc_entrypoint=sdk-cli; cch=a77b8;"


def _inject_system(system):
    """把 billing 标识塞进 system，已有的话就不重复加。"""
    # 没传 system：直接用 billing 当 system
    if system is NotGiven or system is None:
        return [{"type": "text", "text": BILLING}]
    # system 是字符串：转成两个文本块（billing + 原文）
    if isinstance(system, str):
        return [{"type": "text", "text": BILLING}, {"type": "text", "text": system}]
    # system 是列表：检查是否已有 billing，没有就插到最前面
    if isinstance(system, list):
        if any(isinstance(b, dict) and "billing-header" in b.get("text", "") for b in system):
            return system
        return [{"type": "text", "text": BILLING}] + system
    return system


# --- 同步版：包一层 create / stream ---
_orig_create = _msgs.Messages.create
_orig_stream = _msgs.Messages.stream


def _create(self, **kwargs):
    kwargs["system"] = _inject_system(kwargs.get("system", NotGiven))
    return _orig_create(self, **kwargs)


def _stream(self, **kwargs):
    kwargs["system"] = _inject_system(kwargs.get("system", NotGiven))
    return _orig_stream(self, **kwargs)


_msgs.Messages.create = _create
_msgs.Messages.stream = _stream

# --- 异步版：同样包一层 ---
_orig_async_create = _msgs.AsyncMessages.create
_orig_async_stream = _msgs.AsyncMessages.stream


async def _async_create(self, **kwargs):
    kwargs["system"] = _inject_system(kwargs.get("system", NotGiven))
    return await _orig_async_create(self, **kwargs)


def _async_stream(self, **kwargs):
    kwargs["system"] = _inject_system(kwargs.get("system", NotGiven))
    return _orig_async_stream(self, **kwargs)


_msgs.AsyncMessages.create = _async_create
_msgs.AsyncMessages.stream = _async_stream


def _run():
    import runpy
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m examples.python._billing_patch <脚本路径>")
        sys.exit(1)
    runpy.run_path(sys.argv[1], run_name="__main__")


if __name__ == "__main__":
    _run()
