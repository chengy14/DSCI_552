#
import enum

__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/24/2019 3:43 PM'

ROOT_PATH = "..\\assets"
SPLASH = "\\"
MFCC_FILE_PATH = "Frogs_MFCCs.csv"
FULL_COLUMNS = ["MFCCs_ 1", "MFCCs_ 2", "MFCCs_ 3", "MFCCs_ 4", "MFCCs_ 5", "MFCCs_ 6", "MFCCs_ 7", "MFCCs_ 8",
				"MFCCs_ 9", "MFCCs_10", "MFCCs_11", "MFCCs_12", "MFCCs_13", "MFCCs_14", "MFCCs_15", "MFCCs_16",
				"MFCCs_17", "MFCCs_18", "MFCCs_19", "MFCCs_20", "MFCCs_21", "MFCCs_22", "Family", "Genus", "Species",
				"RecordID"]
LABELS_NAME = ["Family", "Genus", "Species"]

Y_LABEL = [['Leptodactylidae', 'Bufonidae', 'Dendrobatidae', 'Hylidae'],
		   ['Adenomera', 'Ameerega', 'Dendropsophus', 'Hypsiboas', 'Leptodactylus', 'Osteocephalus', 'Rhinella',
			'Scinax'],
		   ['AdenomeraAndre', 'AdenomeraHylaedactylus', 'Ameeregatrivittata', 'HylaMinuta', 'HypsiboasCinerascens',
			'HypsiboasCordobae', 'LeptodactylusFuscus', 'OsteocephalusOophagus', 'Rhinellagranulosa',
			'ScinaxRuber']]

MODEL_NAMES = [
			'Chain 1', 'Chain 2', 'Chain 3', 'Chain 4', 'Chain 5', 'Chain 6', 'Chain 7', 'Chain 8', 'Chain 9',
			'Chain 10', 'Ensemble']

