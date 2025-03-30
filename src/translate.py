import os
import asyncio
import re
from typing import List, Dict
import srt
from openai import AsyncOpenAI
from .file_handler import load_srt_file, save_str_file
from .config import load_config
from .logger import logger

class SubtitleTranslator:
    def __init__(self):
        config = load_config()
        self.semaphore = asyncio.Semaphore(config['translation']['max_concurrent_calls'])
        self.llm_client = AsyncOpenAI(
            api_key=config['openai']['api_key'],
            base_url=config['openai']['api_url'],
        )
        self.model = config['openai']['model']
        self.max_tokens = config['openai'].get('max_tokens', 2000)
        self.temperature = config['openai'].get('temperature', 0.3)
        self.batch_size = config['translation']['batch_size']
        self.total_tokens = 0
        self.start_time = None

        # 定义翻译规则
        self.translation_rules = """
            1. 保持字幕序号和时间轴格式不变
            2. 只翻译内容部分
            3. 保留内容中的所有标签和格式标记
            4. 保持换行符不变
            5. 返回格式必须与输入完全一致
            6. 不要解释，不要生成任何额外的文本
            7. 不要翻译人物名称
        """

    def _format_subtitle_batch(self, entries: List[srt.Subtitle]) -> str:
        """格式化字幕批次，保持原始格式"""
        batch_header = f"""请严格保持以下字幕的序号和时间轴格式，只翻译内容部分：
{len(entries)}条字幕，格式示例：
1
00:00:01,000 --> 00:00:03,000
这是要翻译的内容

翻译要求：
{self.translation_rules}

待翻译字幕：
"""
        formatted_entries = []
        for entry in entries:
            formatted_entries.append(
                f"{entry.index}\n"
                f"{entry.start} --> {entry.end}\n"
                f"{entry.content}"
            )
        return batch_header + "\n\n".join(formatted_entries)

    async def _parse_translated_text(self, translated_text: str, original_entries: List[srt.Subtitle], source_language: str, target_language: str) -> List[srt.Subtitle]:
        """解析翻译后的文本，保持原始时间轴"""
        translated_blocks = translated_text.strip().split('\n\n')

        # 创建原始字幕的索引映射，用于后续查找
        original_map = {entry.index: i for i, entry in enumerate(original_entries)}

        # 初始化结果列表，先用原始字幕填充
        result = original_entries.copy()

        # 解析翻译后的文本块
        for block in translated_blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                try:
                    # 获取字幕序号
                    subtitle_index = int(lines[0])
                    # 获取翻译后的内容 (跳过序号和时间轴行)
                    content = '\n'.join(lines[2:])

                    # 查找对应的原始字幕在列表中的位置
                    if subtitle_index in original_map:
                        original_position = original_map[subtitle_index]
                        original_entry = original_entries[original_position]
                        # 创建新的字幕条目，保持原始序号和时间轴
                        result[original_position] = srt.Subtitle(
                            index=original_entry.index,
                            start=original_entry.start,
                            end=original_entry.end,
                            content=content
                        )

                        # 添加调试日志
                        logger.debug(f"成功解析字幕 #{subtitle_index}:\n"
                                    f"原始内容: {original_entry.content}\n"
                                    f"翻译内容: {content}")
                except (ValueError, IndexError) as e:
                    logger.error(f"解析字幕块失败: {block}\n错误: {str(e)}")
                    continue

        return result

    async def translate_batch(self, entries: List[srt.Subtitle], batch_index: int, total_batches: int, source_language: str, target_language: str) -> List[srt.Subtitle]:
        start_index = entries[0].index
        end_index = entries[-1].index
        batch_start_time = asyncio.get_event_loop().time()

        print(f"处理第 {batch_index + 1}/{total_batches} 批字幕 (范围 {start_index}-{end_index})")
        logger.info(f"处理第 {batch_index + 1}/{total_batches} 批字幕 (范围 {start_index}-{end_index})")

        async with self.semaphore:
            try:
                batch_text = self._format_subtitle_batch(entries)
                logger.debug(f"第 {batch_index + 1} 批待翻译内容：\n{batch_text}")  # 添加输入内容日志

                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
                                你是一个专业的字幕翻译器，请严格遵循以下规则：
                                {self.translation_rules}
                                翻译从{source_language}到{target_language}
                            """
                        },
                        {
                            "role": "user",
                            "content": batch_text
                        }
                    ]
                )

                translated_text = response.choices[0].message.content.strip()
                # 添加详细的日志检查
                logger.debug(f"第 {batch_index + 1} 批翻译返回内容：\n{translated_text}")

                if not translated_text:
                    logger.error(f"批次 {batch_index + 1} 翻译返回空内容")
                    return entries

                # 检查返回的格式是否正确
                if len(translated_text.strip().split('\n\n')) != len(entries):
                    logger.error(f"批次 {batch_index + 1} 翻译返回内容格式错误")
                    logger.debug(f"错误内容：\n{translated_text}")
                    return entries

                # 计算并记录 tokens 使用情况
                completion_tokens = response.usage.completion_tokens
                self.total_tokens += completion_tokens

                batch_time = asyncio.get_event_loop().time() - batch_start_time
                print(f"第 {batch_index + 1}/{total_batches} 批翻译完成，耗时 {batch_time:.1f} 秒，输出 tokens {completion_tokens}")
                logger.info(f"第 {batch_index + 1}/{total_batches} 批翻译完成，耗时 {batch_time:.1f} 秒，输出 tokens {completion_tokens}")

                translated_entries = await self._parse_translated_text(
                    translated_text,
                    entries,
                    source_language,
                    target_language
                )

                # 添加翻译结果验证
                if len(translated_entries) != len(entries):
                    logger.error(f"批次 {batch_index + 1} 翻译结果数量不匹配：预期 {len(entries)}，实际 {len(translated_entries)}")

                    # 找出缺失的字幕条目
                    translated_indices = {entry.index for entry in translated_entries}
                    missing_entries = [entry for entry in entries if entry.index not in translated_indices]

                    logger.info(f"尝试重新翻译 {len(missing_entries)} 条缺失的字幕")

                    # 对缺失的条目单独翻译
                    try:
                        retry_text = self._format_subtitle_batch(missing_entries)
                        retry_response = await self.llm_client.chat.completions.create(
                            model=self.model,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"""
                                        你是一个专业的字幕翻译器，请严格遵循以下规则：
                                        {self.translation_rules}
                                        翻译到目标语言 {target_language}
                                    """
                                },
                                {
                                    "role": "user",
                                    "content": retry_text
                                }
                            ]
                        )

                        # 计算并记录补充翻译的 tokens
                        self.total_tokens += retry_response.usage.completion_tokens

                        retry_translated_text = retry_response.choices[0].message.content.strip()
                        retry_entries = await self._parse_translated_text(
                            retry_translated_text,
                            missing_entries,
                            source_language,
                            target_language
                        )

                        # 合并翻译结果
                        all_entries = {entry.index: entry for entry in translated_entries}
                        all_entries.update({entry.index: entry for entry in retry_entries})

                        # 按原始顺序重建结果列表
                        final_entries = [all_entries.get(entry.index, entry) for entry in entries]

                        if len(final_entries) == len(entries):
                            logger.info("成功补充翻译缺失的字幕")
                            return final_entries

                    except Exception as e:
                        logger.error(f"补充翻译失败: {str(e)}")

                    # 如果补充翻译失败，返回原始字幕
                    return entries

                return translated_entries

            except Exception as e:
                logger.error(f"批次 {batch_index + 1}/{total_batches} 翻译失败: {str(e)}")
                return entries

    async def translate_all(self, subtitles: List[srt.Subtitle], source_language: str, target_language: str) -> List[srt.Subtitle]:
        self.start_time = asyncio.get_event_loop().time()
        total_subtitles = len(subtitles)
        total_batches = (total_subtitles + self.batch_size - 1) // self.batch_size

        logger.info(f"开始翻译，共 {total_subtitles} 条字幕，分 {total_batches} 批处理")

        # 创建任务列表
        tasks = []
        for i in range(0, total_subtitles, self.batch_size):
            batch = subtitles[i:i + self.batch_size]
            batch_index = i // self.batch_size

            tasks.append(
                self.translate_batch(
                    batch,
                    batch_index,
                    total_batches,
                    source_language,
                    target_language
                )
            )

        # 并行执行所有任务
        translated_batches = await asyncio.gather(*tasks)
        translated_subtitles = [sub for batch in translated_batches for sub in batch]

        # 确保返回的字幕保持原始序号
        for i, sub in enumerate(translated_subtitles):
            if i < len(subtitles):
                # 确保序号与原始字幕一致
                translated_subtitles[i] = srt.Subtitle(
                    index=subtitles[i].index,  # 使用原始序号
                    start=sub.start,
                    end=sub.end,
                    content=sub.content
                )

        total_time = asyncio.get_event_loop().time() - self.start_time
        # 记录日志并同时打印到控制台
        logger.info(f"翻译完成")
        logger.info(f"总共 {total_subtitles} 条字幕, {total_time:.1f} 秒, {self.total_tokens} tokens.")
        logger.info(f"平均 {(total_subtitles/total_time):.2f} eps，{(self.total_tokens/total_time):.2f} tps, {(self.total_tokens/total_subtitles):.1f} tpe.")

        # 同样的信息也直接打印到控制台
        print(f"总共 {total_subtitles} 条字幕, {total_time:.1f} 秒, {self.total_tokens} tokens.")
        print(f"平均 {(total_subtitles/total_time):.2f} eps，{(self.total_tokens/total_time):.2f} tps, {(self.total_tokens/total_subtitles):.1f} tpe.")

        return translated_subtitles

async def translate_subtitles(source_srt_file, source_language='auto', target_language='简体中文'):
    config = load_config()
    print(f"开始翻译 SRT 文件 '{source_srt_file}'")
    logger.info(f"开始翻译 SRT 文件 '{source_srt_file}'")
    logger.info("\n=== 翻译函数环境变量 ===")
    logger.info(f"环境 OPENAI_API_URL: {config['openai']['api_url']}")
    logger.info(f"环境 OPENAI_API_KEY: {config['openai']['api_key'][:8]}...")
    logger.info(f"环境 OPENAI_API_MODEL: {config['openai']['model']}")
    logger.info(f"环境 OPENAI_API_MAX_TOKENS: {config['openai']['max_tokens']}")
    logger.info(f"环境 OPENAI_API_TEMPERATURE: {config['openai']['temperature']}")
    logger.info(f"环境 TRANSLATION_BATCH_SIZE: {config['translation']['batch_size']}")
    logger.info(f"环境 TRANSLATION_MAX_CONCURRENT_CALLS: {config['translation']['max_concurrent_calls']}")
    logger.info(f"环境 TRANSLATION_PARALLEL_WORKERS: {config['translation']['parallel_workers']}")

    # Load subtitle file
    subtitles = load_srt_file(source_srt_file)

    # Initialize translator and process all subtitles
    translator = SubtitleTranslator()
    translated_subtitles = await translator.translate_all(subtitles, source_language, target_language)

    # Save translated file
    target_srt_file = source_srt_file.replace('.srt', f'.{target_language}.srt')
    if translated_subtitles:
        save_str_file(target_srt_file, translated_subtitles)
    else:
        raise Exception("没有可保存的字幕内容")

    print(f"翻译后的字幕已保存到 '{target_srt_file}'")
    return target_srt_file
