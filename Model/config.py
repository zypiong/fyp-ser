"""
Configuration file: using this one to keep all the paths in one place for various imports.

TRAINING_FILES_PATH = Path of the training files. Here there are

- the RAVDESS dataset files (Folders Actor_01 to Actor_24
- the TESS dataset renamed files (Folders Actor_25 and Actor_26)

SAVE_DIR_PATH = Path of the joblib features created with create_features.py

MODEL_DIR_PATH = Path for the keras model created with neural_network.py

TESS_ORIGINAL_FOLDER_PATH = Path for the TESS dataset original folder (used by tess_pipeline.py)

"""
import sys
import os
import pathlib

working_dir_path = pathlib.Path().absolute()
one_level_up_path = os.path.dirname(working_dir_path)

if sys.platform.startswith('win32'):
    MODEL_DIR_PATH = str(working_dir_path) + '\\model\\'
    TESS_ORIGINAL_PATH = str(working_dir_path) + \
        '\\TESS_Toronto_emotional_speech_set_data\\'
    RAVDESS_ORIGINAL_PATH = str(working_dir_path) + '\\Ravdess_data\\'
    DATASET_DIR_PATH = str(working_dir_path) + '\\datasets\\'
    EXAMPLES_PATH = str(working_dir_path) + '\\examples\\'
    CHECKPOINTS_DIR_PATH = str(working_dir_path) + '\\checkpoints\\'
    VIDEOS_FILE_PATH = str(one_level_up_path) + '\\videos\\'
    AUDIO_ONLY_PATH = str(one_level_up_path) + '\\AudioOnly\\'
else:
    MODEL_DIR_PATH = str(working_dir_path) + '/model/'
    TESS_ORIGINAL_PATH = str(working_dir_path) + \
        '/TESS_Toronto_emotional_speech_set_data/'
    RAVDESS_ORIGINAL_PATH = str(working_dir_path) + '/Ravdess_data/'
    DATASET_DIR_PATH = str(working_dir_path) + '/datasets/'
    EXAMPLES_PATH = str(working_dir_path) + '/examples/'
    CHECKPOINTS_DIR_PATH = str(working_dir_path) + '/checkpoints/'
    VIDEOS_FILE_PATH = str(one_level_up_path) + '/videos/'
    AUDIO_ONLY_PATH = str(one_level_up_path) + '/AudioOnly/'
