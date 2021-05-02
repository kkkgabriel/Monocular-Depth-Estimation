'''
# not used, no time yo. 
This file stores constants and global configuration variables
'''
import pickle

# constants
MODELS_DIR = 'models/'
MAIN_RESULTS_DIR = 'results/'
ALL_RESULTS_FILE = MAIN_RESULTS_DIR + 'all_results.pkl'

# settable vars
hyperparams = {
	'preprocessing': {},
	"dataloader": {
		"params": {
			'batch_size': 32,
			'shuffle': True
		}
	},
	"optimizer": {},
	"loss_function": {},
	"model": {}
}

def empty_results():
	output = open(ALL_RESULTS_FILE, 'wb')
	pickle.dump([], output)
	output.close()