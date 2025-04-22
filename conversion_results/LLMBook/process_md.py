import re

def process_markdown(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 删除所有<span>标签
    content = re.sub(r'<span.*?</span>', '', content, flags=re.DOTALL)

    # 删除指定标题前的内容
    target_heading = '## **背景与基础知识**'
    index = content.find(target_heading)
    if index != -1:
        content = content[index:]

    # 提升章节标题级别
    # 处理## **第x章...** 改为# **第x章...**
    content = re.sub(
        r'^##\s+(\*\*第[一二三四五六七八九十\d]+章\b.*?\*\*)',
        r'# \1',
        content,
        flags=re.MULTILINE
    )

    # 处理#### **x.x.x...** 改为### **x.x.x...**
    content = re.sub(
        r'^####\s+(\*\*(?:[\d一二三四五六七八九十]+\.){2,}[\d一二三四五六七八九十]+\b.*?\*\*)',
        r'### \1',
        content,
        flags=re.MULTILINE
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

# 使用示例
process_markdown('LLMBook.md', 'processedBook.md')