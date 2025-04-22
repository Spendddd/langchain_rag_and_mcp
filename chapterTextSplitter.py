from chineseRecursiveTextSplitter import ChineseRecursiveTextSplitter
import re

# 自定义拆分器
class ChapterTextSplitter(ChineseRecursiveTextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=100, separator=r"第\d+章.*\n", **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.pattern = re.compile(separator)

    def split_text(self, text):
        # 找到所有章节标题的位置
        chapters = self.pattern.finditer(text)
        chapter_starts = [m.start() for m in chapters]
        chapter_starts.append(len(text))

        # 按章节分割文本
        splits = []
        for i in range(len(chapter_starts) - 1):
            start = chapter_starts[i]
            end = chapter_starts[i + 1]
            chunk = text[start:end]
            splits.append(chunk)

        # 进一步拆分每个章节为更小的块
        final_splits = []
        for chunk in splits:
            sub_splits = self.split_text_recursive(chunk, self.chunk_size, self.chunk_overlap)
            final_splits.extend(sub_splits)

        return final_splits

    def split_text_recursive(self, text, chunk_size, chunk_overlap):
        splits = []
        current = 0
        while current < len(text):
            end = current + chunk_size
            if end >= len(text):
                end = len(text)
            else:
                # 尝试在块内找到最近的句子结束
                end = text.rfind('.', current, end)
                if end == -1:
                    end = current + chunk_size
            chunk = text[current:end].strip()
            if chunk:
                splits.append(chunk)
            current = end
        return splits