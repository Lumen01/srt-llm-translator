import argparse
import os
import asyncio

from src.translate import translate_subtitles

async def main():
    parser = argparse.ArgumentParser(description="使用 LLM 的 SRT 字幕文件翻译器")
    parser.add_argument("--source-lang", type=str, default="'自动检测源语言'", help="源语言 (例如: 英语)")
    parser.add_argument("--target-lang", type=str, required=True, help="目标语言 (例如: 中文)")
    parser.add_argument("--file", type=str, help="源 SRT 文件路径")
    parser.add_argument("--folder", type=str, help="包含 SRT 文件的文件夹路径")
    args = parser.parse_args()

    if args.file and args.folder:
        raise ValueError("请只指定 --file 或 --folder 其中之一，不能同时使用")

    if args.folder:
        for filename in os.listdir(args.folder):
            if filename.endswith(".srt"):
                file_path = os.path.join(args.folder, filename)
                await translate_subtitles(source_srt_file=file_path, 
                                          source_language=args.source_lang, 
                                          target_language=args.target_lang)
    elif args.file:
        await translate_subtitles(source_srt_file=args.file, 
                                  source_language=args.source_lang, 
                                  target_language=args.target_lang)
    else:
        raise ValueError("必须提供以下参数之一：--file 或 --folder")

if __name__ == "__main__":
    asyncio.run(main())
