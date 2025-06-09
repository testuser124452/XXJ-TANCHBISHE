import sys
import datetime
from colorama import Fore, Style, init

init(autoreset=True)


def log(msg, level='INFO', to_file=True):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    color = {
        'INFO': Fore.GREEN,
        'WARN': Fore.YELLOW,
        'ERROR': Fore.RED,
        'DEBUG': Fore.CYAN,
    }.get(level.upper(), Fore.WHITE)
    output = f"{color}[{now}] [{level.upper()}] {msg}{Style.RESET_ALL}"
    print(output, file=sys.stderr if level == 'ERROR' else sys.stdout)
    # 写到本地日志文件
    if to_file:
        with open("train_log.txt", "a", encoding="utf8") as f:
            f.write(f"[{now}] [{level.upper()}] {msg}\n")
