import os
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yml')

    # 首先尝试从配置文件加载
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'openai': {
                'api_key': None,
                'api_url': "https://api.openai.com/v1",
                'model': "gpt-4-turbo-preview"
            },
            'translation': {
                'batch_size': 10,
                'max_concurrent_calls': 20,
                'parallel_workers': 10  # 新增：并行工作线程数
            },
            'logging': {
                'level': 'INFO'  # 可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
            }
        }

    # 环境变量优先级高于配置文件
    config['openai']['api_key'] = os.getenv('OPENAI_API_KEY', config['openai']['api_key'])
    config['openai']['api_url'] = os.getenv('OPENAI_API_URL', config['openai']['api_url'])
    config['openai']['model'] = os.getenv('OPENAI_MODEL', config['openai']['model'])
    config['translation']['batch_size'] = int(os.getenv('BATCH_SIZE', config['translation']['batch_size']))
    config['translation']['max_concurrent_calls'] = int(os.getenv('MAX_CONCURRENT_CALLS', config['translation']['max_concurrent_calls']))
    config['translation']['parallel_workers'] = int(os.getenv('PARALLEL_WORKERS', config['translation']['parallel_workers']))
    config['logging']['level'] = os.getenv('LOGGING_LEVEL', config['logging']['level'])

    return config