#
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '9/24/2019 3:43 PM'

ROOT_PATH = "..\\assets"
CRIME = "\\communities.csv"
APS_TRAIN = "\\aps_failure_training_set.csv"
APS_TEST = "\\aps_failure_test_set.csv"
APS_SHRINK = "\\aps_failure_shrink_set.csv"
SPLASH = "\\"

CRIME_FULL_COLUMNS = ["state", "county", "community", "communityname string", "fold", "population", "householdsize",
					  "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
					  "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf",
					  "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap",
					  "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov",
					  "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy",
					  "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce",
					  "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
					  "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg",
					  "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig",
					  "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",
					  "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous",
					  "PersPerRentOccHous",
					  "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup",
					  "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone",
					  "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian",
					  "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg",
					  "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85",
					  "PctSameCity85",
					  "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop",
					  "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol",
					  "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",
					  "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens",
					  "PctUsePubTrans",
					  "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
					  "PolicBudgPerPop", "ViolentCrimesPerPop"]

CRIME_REMAIN_COLUMNS = ['population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp',
						'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome',
						'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire',
						'medFamInc',
						'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap',
						'HispPerCap',
						'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
						'PctUnemployed',
						'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
						'MalePctDivorce',
						'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
						'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg',
						'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig',
						'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
						'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous',
						'PersPerRentOccHous',
						'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
						'PctHousOccup',
						'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone',
						'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian',
						'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg',
						'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85',
						'PctSameCity85',
						'PctSameState85', 'LandArea', 'PopDens', 'PctUsePubTrans', 'LemasPctOfficDrugUn',
						'ViolentCrimesPerPop']

APS_FULL_COLUMNS = ["class", "aa_000", "ab_000", "ac_000", "ad_000", "ae_000", "af_000", "ag_000", "ag_001", "ag_002",
					"ag_003", "ag_004", "ag_005", "ag_006", "ag_007", "ag_008", "ag_009", "ah_000", "ai_000", "aj_000",
					"ak_000", "al_000", "am_0", "an_000", "ao_000", "ap_000", "aq_000", "ar_000", "as_000", "at_000",
					"au_000", "av_000", "ax_000", "ay_000", "ay_001", "ay_002", "ay_003", "ay_004", "ay_005", "ay_006",
					"ay_007", "ay_008", "ay_009", "az_000", "az_001", "az_002", "az_003", "az_004", "az_005", "az_006",
					"az_007", "az_008", "az_009", "ba_000", "ba_001", "ba_002", "ba_003", "ba_004", "ba_005", "ba_006",
					"ba_007", "ba_008", "ba_009", "bb_000", "bc_000", "bd_000", "be_000", "bf_000", "bg_000", "bh_000",
					"bi_000", "bj_000", "bk_000", "bl_000", "bm_000", "bn_000", "bo_000", "bp_000", "bq_000", "br_000",
					"bs_000", "bt_000", "bu_000", "bv_000", "bx_000", "by_000", "bz_000", "ca_000", "cb_000", "cc_000",
					"cd_000", "ce_000", "cf_000", "cg_000", "ch_000", "ci_000", "cj_000", "ck_000", "cl_000", "cm_000",
					"cn_000", "cn_001", "cn_002", "cn_003", "cn_004", "cn_005", "cn_006", "cn_007", "cn_008", "cn_009",
					"co_000", "cp_000", "cq_000", "cr_000", "cs_000", "cs_001", "cs_002", "cs_003", "cs_004", "cs_005",
					"cs_006", "cs_007", "cs_008", "cs_009", "ct_000", "cu_000", "cv_000", "cx_000", "cy_000", "cz_000",
					"da_000", "db_000", "dc_000", "dd_000", "de_000", "df_000", "dg_000", "dh_000", "di_000", "dj_000",
					"dk_000", "dl_000", "dm_000", "dn_000", "do_000", "dp_000", "dq_000", "dr_000", "ds_000", "dt_000",
					"du_000", "dv_000", "dx_000", "dy_000", "dz_000", "ea_000", "eb_000", "ec_00", "ed_000", "ee_000",
					"ee_001", "ee_002", "ee_003", "ee_004", "ee_005", "ee_006", "ee_007", "ee_008", "ee_009", "ef_000",
					"eg_000"]
