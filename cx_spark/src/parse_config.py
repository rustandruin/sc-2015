from collections import defaultdict
import ConfigParser
import ast
import os

def load_configuration():
	config_params = defaultdict(dict)
	cur_path = os.path.dirname(os.path.realpath(__file__))

	config = ConfigParser.RawConfigParser()
	settings_file = os.path.join(cur_path,'conf/settings.cfg')
	config.read(settings_file)
	sections = config.sections()
	for section in sections:
		options = config.options(section)
		for option in options:
			config_params[section][option] = config.get(section, option)
	return config_params
