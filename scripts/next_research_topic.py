#!/usr/bin/env python3
"""研究主题轮换脚本。读取 research_topics.md，返回下一个未研究的主题。"""

import sys
from pathlib import Path

TOPICS_FILE = Path(__file__).parent.parent / "docs" / "research_topics.md"
FINDINGS_FILE = Path(__file__).parent.parent / "docs" / "research_findings.md"


def get_researched_topics():
    """从 findings 文件中提取已研究过的主题关键词。"""
    if not FINDINGS_FILE.exists():
        return set()
    text = FINDINGS_FILE.read_text()
    # 提取 ## 标题行作为已研究主题
    researched = set()
    for line in text.split("\n"):
        if line.startswith("## "):
            researched.add(line.strip("# ").strip())
    return researched


def get_all_topics():
    """从 topics 文件中提取所有主题。"""
    if not TOPICS_FILE.exists():
        return []
    text = TOPICS_FILE.read_text()
    topics = []
    in_list = False
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("## 待研究主题"):
            in_list = True
            continue
        if in_list and line.startswith("##"):
            break
        if in_list and line and line[0].isdigit() and "." in line:
            # 提取 "1. topic text"
            topic = line.split(".", 1)[1].strip()
            topics.append(topic)
    return topics


def get_next_topic():
    """返回下一个未研究的主题。如果全部研究完，从头循环。"""
    all_topics = get_all_topics()
    researched = get_researched_topics()
    
    # 过滤已研究的（模糊匹配：主题关键词出现在已研究列表中）
    remaining = []
    for t in all_topics:
        # 取前10个字符作为关键词匹配
        key = " ".join(t.split()[:3])
        if not any(key.lower() in r.lower() for r in researched):
            remaining.append(t)
    
    if remaining:
        return remaining[0]
    elif all_topics:
        # 全部研究完，从头开始
        return all_topics[0]
    else:
        return "quantitative factor investing A-share China"


if __name__ == "__main__":
    topic = get_next_topic()
    print(topic)
