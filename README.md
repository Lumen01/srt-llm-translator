# LLM 字幕翻译器

## 概述

LLM 字幕翻译器是一款基于 Python 的工具，使用大型语言模型实现字幕的跨语言翻译。该工具会保留字幕的原始时间戳，输出目标语言的字幕文件。该工具使用符合 OpenAI API 标准的 API 进行翻译，国内可使用 DeepSeek、火山引擎等

**更新**

update-20250330

- 改进批量处理功能，巨幅提升翻译效率
- 增加配置文件，丰富配置项，方便对翻译参数进行微调
- 增加日志功能，可输出详细日志方便 debug

**环境要求**

- Python 3.x
- LLM 服务 API 密钥
- 所需 Python 包：
  - `openai`
  - `srt`

## 安装使用

**安装**

1. 克隆仓库：

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. 安装依赖包：
    ```bash
    pip install -r requirements.txt
    ```

3. 修改配置文件：
    ```bash
    cp config.sample.yml config.yml
    vim comfig.yml
    ```

**使用方式**

```bash
# 翻译单个 SRT 文件：
python srt_llm_translator.py --target-lang <目标语言代码> --file <源文件.srt>
# 批量翻译多个SRT文件：
python srt_llm_translator.py --target-lang <目标语言代码> --folder <目录路径>
```

`<目标语言代码>` 除了可以是标准的代码，比如 `zh-CN`、`zh-TW`、`zh-HK`、`yue`、`es`，也可以是口语化的内容，比如`中文`、`粤语`、`日文`，甚至方言也可以，比如`广东话`、`河南话`、`四川话`。

```bash
# 翻译单个 SRT 文件为普通话
python srt_llm_translator.py --target-lang "zh-CN" --file sample/sample.srt

# 翻译单个 SRT 文件为粤语
python srt_llm_translator.py --target-lang "zh-HK" --file sample/sample.srt
# 或
python srt_llm_translator.py --target-lang "yue" --file sample/sample.srt
```

*※注意：翻译为方言有可能因为语法问题导致条目的缺失，但是程序会以原字幕补充。*

参数说明

- `--target-lang`: 目标语言代码（如`en`表示英语，`es`表示西班牙语）
- `--source-lang`: 可选的源语言代码，默认自动检测
- `--file`: 源SRT文件路径
- `--folder`: 包含多个SRT文件的目录路径

## 成本与性能
源项目对单条字幕进行提交翻译，即使采用并发请求也不能大幅提升翻译速度，而且容易触发接口并发限制。

本项目改进了批量翻译的逻辑，采用一次调用提交多条字幕（通过`batch_size`控制）同时可以设定并行工作线程数量（通过 `parallel_workers` 控制）的方式大幅提升翻译性能。

以下是对一个 512 个条目的英文字幕内容进行简体中文的测试结果：

| batch_size | parallel_workers | 耗时         | 速率                      |
| ---------- | ---------------- | ------------ | ------------------------- |
| 8          | 4                | 55.1 sec     | 9.30 eps，277.61 tps      |
| 8          | 8                | 53.0 sec     | 9.67 eps，288.30 tps      |
| 8          | 16               | 52.2 sec     | 9.81 eps，293.04 tps      |
| 16         | 8                | 44.7 sec     | 11.45 eps，340.20 tps     |
| 16         | 16               | 42.4 sec     | 12.09 eps，358.29 tps     |
| **32**     | **8**            | **40.5 sec** | **12.65 eps，371.66 tps** |
| 32         | 16               | 41.8 sec     | 12.24 eps，360.64 tps     |

- 从测试结果来看
  - `batch_size` 过小的情况下是，提高 `parallel_workers` 对效率提升不大。
  - `batch_size` 设为 32 同时 `parallel_workers` 设为 8 的情况下比较理想。
  - 假设 2 小时的电影台词数为测试量的4倍即 2048 条台词来计算，可以在 3分钟内能完成字幕翻译。
- 测试使用 DeepSeek 模型，请参考[官方文档](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。
- 不同服务商对于并发请求限制不一样，请根据官方文档来设置并发参数 `max_concurrent_calls` 以实现效率最大化。
