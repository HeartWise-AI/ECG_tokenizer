
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.parser import HeartWiseParser
from utils.config.heartwise_config import HeartWiseConfig


def main(config: HeartWiseConfig):
    print(config)

if __name__ == "__main__":
    config: HeartWiseConfig = HeartWiseParser.parse_config()
    main(config)
