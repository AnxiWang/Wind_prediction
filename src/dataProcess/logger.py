import logging
from logging import handlers
import os


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, level='info', when='D', backcount=50):
        self.fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        log_path = os.path.join(os.path.abspath(os.getcwd()), 'wrf_data_analysis_logs')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        filename = os.path.join(log_path, 'wrf_data_analysis.log')
        self.logger = logging.getLogger(filename)
        # 设置日志格式
        format_str = logging.Formatter(self.fmt)
        # 往屏幕上输出
        sh = logging.StreamHandler()
        # 设置屏幕上显示的格式
        sh.setFormatter(format_str)
        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))
        #往文件里写入#指定间隔时间自动生成文件的处理器
        # rf = handlers.RotatingFileHandler(filename, maxBytes=2048 * 1024, backupCount=50)
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒、 M 分、H 小时、D 天、W 每星期（interval==0时代表星期一）、 midnight 每天凌晨
        tr = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backcount, encoding='utf-8')
        # 设置文件里写入的格式
        tr.setFormatter(format_str)
        # 把对象加到logger里
        self.logger.addHandler(sh)
        self.logger.addHandler(tr)


log = Logger()