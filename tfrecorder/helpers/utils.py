import logging
import logging.handlers

import os
import time
import random
import string
import sys

def elapsed_since(last_time):
	"""
	Convenience function that returns a formatted string giving the time elapsed since last_time.
	"""
	return time.strftime('%H:%M:%S', time.gmtime(time.time() - last_time))

def eta_based_on_elapsed_time(done, total, start_time):

	seconds_to_go = int((time.time() - start_time)*(total-done)/done)
	days_to_go = seconds_to_go // (3600*24)

	if days_to_go > 0:
		seconds_to_go = seconds_to_go - days_to_go * 3600*24
		time_to_go = '%s:%s' % (days_to_go, time.strftime('%H:%M:%S', time.gmtime(seconds_to_go)))

	else:
		time_to_go = time.strftime('%H:%M:%S', time.gmtime(seconds_to_go))

	return time_to_go


def generate_random_string(k=6):
	return ''.join(random.choices(string.ascii_lowercase, k=k))


def get_logger(name=None, level=logging.INFO, log_to_console=True, logs_directory_path=None, logs_filename='log.txt'):

	if name is None:
		name = __name__

	log = logging.getLogger(name)

	#log.handlers = [] # as Log is usually a static object, it might have been already initialized, so we re-init it.
	if not len(log.handlers):
		# if this logger has never been configured, then we add the handlers

		# configure a new logger
		log.setLevel(level)

		if log_to_console:
			handler = logging.StreamHandler(sys.stdout)
			handler.setLevel(level)
			log.addHandler(handler)

		if logs_directory_path:
			# rotating
			max_bytes = 10*1024*1024 # 10 MB
			backup_count = 2
			handler = logging.handlers.RotatingFileHandler(os.path.join(logs_directory_path, logs_filename),
			                                               maxBytes=max_bytes,
			                                               backupCount=backup_count)
			#handler = logging.FileHandler(os.path.join(logs_directory_path, logs_filename))
			handler.setLevel(level)
			log.addHandler(handler)

	return log