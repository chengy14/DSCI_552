#
import enum

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/24/2019 3:43 PM'

ROOT_PATH = "..\\assets"
SPLASH = "\\"
FULL_COLUMNS = ['time', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
TIME_DOMAIN = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']


class DATASET_LABEL(enum.Enum):
	BENDING1 = 0
	BENDING2 = 0
	CYCLING = 1
	LYING = 2
	SITTING = 3
	STANDING = 4
	WALKING = 5
