import srt
from typing import List

def load_srt_file(file_path: str) -> List[srt.Subtitle]:
    """
    加载并解析 SRT 文件。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        srt_content = file.read()
    return list(srt.parse(srt_content))

def save_str_file(file_path: str, subtitles: List[srt.Subtitle]) -> None:
    """保存字幕到SRT文件，确保使用原始序号"""
    # 确保字幕序号保持不变
    formatted_srt = srt.compose(subtitles, reindex=False)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_srt)
