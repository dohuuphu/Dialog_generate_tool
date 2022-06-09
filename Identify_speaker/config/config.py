import yaml


def get_config():
    with open('env.yml', encoding='utf-8') as cfgFile:
        config_app = yaml.safe_load(cfgFile)
        # cfgFile.close()
    return config_app