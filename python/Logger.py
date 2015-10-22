from enum import Enum
class LogType(Enum):
	PRINT = 1
	FILE = 2
class Logger:
	def __init__(self, log_type=LogType.PRINT, log_path=''):
		self.type = log_type

	def log(self,message):
		if self.type == LogType.PRINT:
			print message
		else:
			pass
			#TODO file
	def close(self):
		pass
		#TODO close file 
