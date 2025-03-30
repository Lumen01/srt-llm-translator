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

        # 获取价格配置
        self.prompt_cache_hit_price = config['openai'].get('prompt_cache_hit_tokens_price_per_1m', 0.5)
        self.prompt_cache_miss_price = config['openai'].get('prompt_cache_miss_tokens_price_per_1m', 2.0)
        self.completion_price = config['openai'].get('completion_tokens_price_per_1m', 8.0)

        self.total_tokens = 0
        self.prompt_tokens = 0  # 添加输入token计数
        self.completion_tokens = 0  # 添加输出token计数
        self.prompt_cache_hit_tokens = 0  # 添加缓存命中token计数
        self.prompt_cache_miss_tokens = 0  # 添加缓存未命中token计数
        self.start_time = None


        # 定义翻译规则
        self.system_prompt = """你是一位专业的字幕翻译家，请严格遵循以下规则：
1.保持字幕序号、时间轴、字幕标签或格式标记或换行
2.不要翻译人物名称
3.只翻译台词内容，包括标点符号的格式
4.不要解释，不要生成任何额外的文本
5.输出字幕的条目和格式须与输入一致
6.'_TL_'是时间轴标记保持在第二行
7.将字幕从{source_language}翻译到{target_language}
8.字幕示例：
1
_TL_
<i>This is a Sample.</i>
11.返回示例:
1.
_TL_
<i>这是一个示例</i>
        """

    def _get_formatted_system_prompt(self, source_language: str, target_language: str) -> str:
        """根据源语言和目标语言格式化系统提示"""
        return self.system_prompt.format(
            source_language=source_language,
            target_language=target_language
        )

    def _format_subtitle_batch(self, entries: List[srt.Subtitle]) -> str:
        """格式化字幕批次，保持原始格式"""
        formatted_entries = []
        for entry in entries:
            # 使用特殊标记替换时间轴
            formatted_entries.append(
                f"{entry.index}\n"
                f"_TL_\n"
                f"{entry.content}"
            )
        return "\n\n".join(formatted_entries)

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

                    # 检查第二行是否为我们的特殊标记
                    if lines[1] != "_TL_":
                        logger.warning(f"字幕 #{subtitle_index} 的时间轴标记不正确: {lines[1]}")

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
                        logger.info(f"成功解析字幕 #{subtitle_index}:\n"
                                    f"原文: {original_entry.content}\n"
                                    f"译文: {content}")
                except (ValueError, IndexError) as e:
                    logger.error(f"解析字幕块失败: {block}\n错误: {str(e)}")
                    continue

        return result

    async def translate_batch(self, entries: List[srt.Subtitle], batch_index: int, total_batches: int, source_language: str, target_language: str) -> List[srt.Subtitle]:
        start_index = entries[0].index
        end_index = entries[-1].index
        batch_start_time = asyncio.get_event_loop().time()

        async with self.semaphore:
            try:
                batch_text = self._format_subtitle_batch(entries)
                logger.debug(f"第 {batch_index + 1} 批待翻译内容：\n{batch_text}")  # 添加输入内容日志

                # 使用辅助方法获取格式化的系统提示
                formatted_system_prompt = self._get_formatted_system_prompt(source_language, target_language)

                # 构建消息列表
                messages = [
                    {
                        "role": "system",
                        "content": formatted_system_prompt
                    },
                    {
                        "role": "user",
                        "content": batch_text
                    }
                ]

                # 记录完整提交信息
                logger.debug(f"原始提交信息： {messages}")

                # 发送翻译请求
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=messages
                )

                # 提取翻译结果
                translated_text = response.choices[0].message.content.strip()

                # 并记录本轮接口调用 tokens 使用情况
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                # 获取缓存命中和未命中的tokens (如果API返回中包含这些字段)
                prompt_cache_hit_tokens = getattr(response.usage, 'prompt_cache_hit_tokens', 0)
                prompt_cache_miss_tokens = getattr(response.usage, 'prompt_cache_miss_tokens', 0)

                # 累计各类 tokens
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.prompt_cache_hit_tokens += prompt_cache_hit_tokens
                self.prompt_cache_miss_tokens += prompt_cache_miss_tokens
                self.total_tokens += prompt_tokens + completion_tokens

                # 添加详细的日志检查
                logger.debug(f"第 {batch_index + 1} 批（范围 {start_index}-{end_index}）字幕译文：\n{translated_text}")

                # 检查返回内容是否为空，如果为空则返回原始字幕
                if not translated_text:
                    logger.error(f"第 {batch_index + 1} 批（范围 {start_index}-{end_index}）字幕译文为空")
                    # 注释掉此行，让方法继续执行，验证补充翻译是否有效
                    # return entries

                # 检查返回的格式是否正确，如果格式错误则返回原始字幕
                if len(translated_text.strip().split('\n\n')) != len(entries):
                    logger.error(f"第 {batch_index + 1} 批（范围 {start_index}-{end_index}）字幕译文条目缺失！预期 {len(entries)}，实际 {len(translated_text.strip().split('\n\n'))}")
                    logger.error(f"错误内容：\n{translated_text}")
                    # 注释掉此行，让方法继续执行，验证补充翻译是否有效
                    # return entries

                # 记录翻译完成日志
                batch_time = asyncio.get_event_loop().time() - batch_start_time
                logger.info(f"第 {batch_index + 1}/{total_batches} 批（范围 {start_index}-{end_index}）翻译完成，耗时 {batch_time:.1f} 秒，输入 tokens {prompt_tokens}，输出 tokens {completion_tokens}")

                # 解析翻译结果
                translated_entries = await self._parse_translated_text(
                    translated_text,
                    entries,
                    source_language,
                    target_language
                )

                # 添加翻译结果验证
                if len(translated_entries) != len(entries):
                    logger.error(f"第 {batch_index + 1} 批（范围 {start_index}-{end_index}）字幕译文条目不匹配：预期 {len(entries)}，实际 {len(translated_entries)}")

                    # 找出缺失的字幕条目
                    translated_indices = {entry.index for entry in translated_entries}
                    missing_entries = [entry for entry in entries if entry.index not in translated_indices]

                    logger.info(f"尝试重新翻译 {len(missing_entries)} 条缺失的字幕")

                    # 对缺失的条目单独翻译
                    try:
                        retry_text = self._format_subtitle_batch(missing_entries)

                        # 使用辅助方法获取格式化的系统提示
                        formatted_system_prompt = self._get_formatted_system_prompt(source_language, target_language)

                        # 构建消息列表
                        messages=[
                            {
                                "role": "system",
                                "content": formatted_system_prompt
                            },
                            {
                                "role": "user",
                                "content": retry_text
                            }
                        ]

                        # 记录完整提交信息
                        logger.debug(f"原始提交信息： {messages}")
                        retry_response = await self.llm_client.chat.completions.create(
                            model=self.model,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            messages=messages
                        )

                        # 计算并记录补充翻译的 tokens
                        retry_prompt_tokens = retry_response.usage.prompt_tokens
                        retry_completion_tokens = retry_response.usage.completion_tokens
                        # 获取缓存命中和未命中的tokens
                        retry_prompt_cache_hit_tokens = getattr(retry_response.usage, 'prompt_cache_hit_tokens', 0)
                        retry_prompt_cache_miss_tokens = getattr(retry_response.usage, 'prompt_cache_miss_tokens', 0)

                        # 累计各类tokens
                        self.prompt_tokens += retry_prompt_tokens
                        self.completion_tokens += retry_completion_tokens
                        self.prompt_cache_hit_tokens += retry_prompt_cache_hit_tokens
                        self.prompt_cache_miss_tokens += retry_prompt_cache_miss_tokens
                        self.total_tokens += retry_prompt_tokens + retry_completion_tokens

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
                logger.error(f"批次 {batch_index + 1}/{total_batches} （范围 {start_index}-{end_index}）翻译失败: {str(e)}")
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
        logger.info(f"总共 {total_subtitles} 条字幕, {total_time:.1f} 秒")
        logger.info(f"tokens 输入/输出/合计 {self.prompt_tokens}/{self.completion_tokens}/{self.total_tokens})")
        logger.info(f"平均 {(total_subtitles/total_time):.2f} eps，{(self.completion_tokens/total_time):.2f} tps, {(self.completion_tokens/total_subtitles):.1f} tpe")

        # 计算预估费用
        # 如果没有缓存命中/未命中的详细数据，则假设所有prompt tokens都是未命中的
        if self.prompt_cache_hit_tokens == 0 and self.prompt_cache_miss_tokens == 0:
            self.prompt_cache_miss_tokens = self.prompt_tokens

        hit_cost = self.prompt_cache_hit_tokens * self.prompt_cache_hit_price / 1000000
        miss_cost = self.prompt_cache_miss_tokens * self.prompt_cache_miss_price / 1000000
        completion_cost = self.completion_tokens * self.completion_price / 1000000
        total_cost = hit_cost + miss_cost + completion_cost

        logger.info(f"输入 tokens：命中 {self.prompt_cache_hit_tokens}/{hit_cost:.4f}元, 未命中 {self.prompt_cache_miss_tokens}/{miss_cost:.4f}元")
        logger.info(f"输出 tokens: {self.completion_tokens}/{completion_cost:.4f}元")
        logger.info(f"总费用: {total_cost:.4f}元")
        logger.info(f"注意：以上费用根据 DeepSeek API 文档预估，仅供参考。北京时间 00:30-08:30 为优惠时段，价格减半。")

        return translated_subtitles

async def translate_subtitles(source_srt_file, source_language='auto', target_language='简体中文'):
    config = load_config()
    logger.info(f"开始翻译 SRT 文件 '{source_srt_file}'")
    logger.info(f"环境 OPENAI_API_URL: {config['openai']['api_url']}")
    logger.info(f"环境 OPENAI_API_KEY: {config['openai']['api_key'][:8]}...")
    logger.info(f"环境 OPENAI_API_MODEL: {config['openai']['model']}")
    logger.info(f"环境 OPENAI_API_MAX_TOKENS: {config['openai']['max_tokens']}")
    logger.info(f"环境 OPENAI_API_TEMPERATURE: {config['openai']['temperature']}")
    logger.info(f"环境 TRANSLATION_BATCH_SIZE: {config['translation']['batch_size']}")
    logger.info(f"环境 TRANSLATION_MAX_CONCURRENT_CALLS: {config['translation']['max_concurrent_calls']}")
    logger.info(f"环境 TRANSLATION_PARALLEL_WORKERS: {config['translation']['parallel_workers']}")

    # 加载原始字幕
    subtitles = load_srt_file(source_srt_file)

    # 初始化翻译器
    translator = SubtitleTranslator()

    # 获取格式化的系统提示
    formatted_system_prompt = translator._get_formatted_system_prompt(source_language, target_language)
    logger.info(f"模型 System Prompt:\n{formatted_system_prompt}")

    # 翻译字幕
    translated_subtitles = await translator.translate_all(subtitles, source_language, target_language)

    # 保存翻译后的字幕
    target_srt_file = source_srt_file.replace('.srt', f'.{target_language}.srt')
    if translated_subtitles:
        save_str_file(target_srt_file, translated_subtitles)
    else:
        raise Exception("没有可保存的字幕内容")

    if config['output']['reindex_subtitles']:
        logger.info(f"字幕已重新编号。")

    logger.info(f"翻译后的字幕已保存到 '{target_srt_file}'")
    return target_srt_file
