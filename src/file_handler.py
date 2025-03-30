import srt
from typing import List
from .config import load_config

def load_srt_file(file_path: str) -> List[srt.Subtitle]:
    """
    加载并解析 SRT 文件。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        srt_content = file.read()
    return list(srt.parse(srt_content))

def save_str_file(file_path: str, subtitles: List[srt.Subtitle]) -> None:
    """保存字幕到SRT文件，根据配置决定是否重新编号"""
    # 从配置文件获取是否重新编号的设置
    config = load_config()
    reindex = config.get('output', {}).get('reindex_subtitles', False)

    # 根据配置决定是否重新编号
    formatted_srt = srt.compose(subtitles, reindex=reindex)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_srt)
