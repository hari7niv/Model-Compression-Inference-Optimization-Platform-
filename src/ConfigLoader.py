import yaml

class ConfigLoader:
    
    @staticmethod
    def load(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        return config