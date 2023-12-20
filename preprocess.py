import os
import pickle
import librosa
import numpy as np

class AudioHTRF(): 
    """
    Audio class that contains all the htrf features needed. 
    """
    pass


class Loader(): 
    """
    Loading an audio file. 
    """
    def __init__(self, sample_rate, mono):
        self.sample_rate = sample_rate 
        self.mono = mono            # check if it fits the htrf option

    def load(self, file_path): 
        signal = librosa.load(file_path, 
                              sr=self.sample_rate, 
                              mono=self.mono)[0]
        return signal
    

class ZeroPadder():
    """
    Padding an array if needed. 
    """
    def __init__(self, mode="contant"): 
        self.mode = mode 

    def left_pad(self, array, num_missing_items): 
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
        return padded_array
    
    def right_pad(self, array, num_missing_items):  # we are going to use this one
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array
    

class LogSpectogramExtractor():
    """
    Extracts log spectograms (dB) from a time-series signal.
    """
    def __init__(self, frame_size, hop_size):
        self.frame_size = frame_size
        self.hop_size = hop_size

    def extract(self, signal): 
        stft = librosa.stft(signal, 
                            n_fft=self.frame_size, 
                            hop_length=self.hop_size)[:-1]
        spectogram = np.abs(stft)
        log_spectogram = librosa.amplitude_to_db(spectogram)
        return log_spectogram
    
class CochleagramExtractor(): 
    """
    Extracts cochleagrams from a time-series signal.
    """
    def __init__(self, frame_size, hop_size):
        self.frame_size = frame_size
        self.hop_size = hop_size

    def extract(self, signal): 
        pass
    
class MFCCExtractor(): 
    """
    Extracts the Mel Frequency Ceptrum Coeffiecients from a time-series signal.
    """
    def __init__(self, frame_size, hop_size):
        self.frame_size = frame_size
        self.hop_size = hop_size

    def extract(self, signal): 
        pass
    

class MinMaxNormalizer(): 
    """
    Applies min max normalization to an array.
    """
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array): 
        """
        Normalizing between (0,1). 
        """
        norm_array = (array - array.min())/(array.max() - array.min())
        norm_array = norm_array*(self.max - self.min) + self.min
        return array
    
    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min)/(self.max - self.min) 
        array = array*(original_max - original_min) + original_min
        return array


class Saver(): 
    """
    Stores features and min and max values. 
    """
    def __init__(self, signal_dir): 
        self.signal_dir = signal_dir 

    def generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.signal_dir, file_name)
        return save_path

    def save_hrir_data(self, signal_dict): 
        save_path = os.path.join(self.signal_dir, "hrirs.pkl")
        self._save_hrir_data(signal_dict, save_path)

    def save_norm_hrir_data(self, signal_dic): 
        save_path = os.path.join(self.signal_dir, "norm_hrirs.pkl")
        self._save_norm_hrir_data(signal_dic, save_path)

    def save_feature_data(self, signal_dic):
        save_path = os.path.join(self.signal_dir, "hrtf_features.pkl")
        self._save_feature_data(signal_dic, save_path)

    def save_norm_feature_data(self, signal_dic):
        save_path = os.path.join(self.signal_dir, "norm_hrtf_features.pkl")
        self._save_norm_feature_data(signal_dic, save_path)

    @staticmethod
    def _save_hrir_data(data, save_path): 
        with open(save_path, "wb") as file: 
            pickle.dump(data, file)

    @staticmethod
    def _save_norm_hrir_data(data, save_path): 
        with open(save_path, "wb") as file: 
            pickle.dump(data, file)

    @staticmethod
    def _save_feature_data(data, save_path):
        with open(save_path, "wb") as file: 
            pickle.dump(data, file)

    @staticmethod
    def _save_norm_feature_data(data, save_path):
        with open(save_path, "wb") as file:
            pickle.dump(data, file)


class PreProcessingPipeline(): 
    """
    Processes audio files in a directory applying the following steps: 
        1.- loading
        2.- padding (if needed)
        3.- extracting log spectogram
        4.- normalizing (spectogram)
        5.- saving
        6.- storing min and max values for all spectograms
    """
    def __init__(self): 
        self.padder = None
        self.extractor= None
        self.normalizer = None
        self.saver = None
        self._loader = None
        self._num_samples = None  
        self.hrir_to_store = {} 
        self.norm_hrir_to_store = {}
        self.features_to_store = {}
        self.norm_feature_to_store = {}

    @property
    def loader(self): 
        return self._loader
    
    @loader.setter
    def loader(self, loader): 
        self._loader = loader

    def process(self, audio_files_dir):
        """
        Looping through all the audio files. 
        """
        for root, dir, files in os.walk(audio_files_dir): 
            for file in files: 
                file_path = os.path.join(root, file)
                self.process_file(file, file_path)
                print(f"Processed file {file_path}")

        self.saver.save_hrir_data(self.hrir_to_store)
        self.saver.save_norm_hrir_data(self.norm_hrir_to_store)
        self.saver.save_feature_data(self.features_to_store)
        self.saver.save_norm_feature_data(self.norm_feature_to_store)

    def process_file(self, file, file_path): 
        """
        Processes signal, normalized signal and extracts one feature.
        """
        signal = self.loader.load(file_path)
        elev, azimuth, channel = self.elev_azimuth_channel(file)
        coordinates = (elev, azimuth)
        self.store_hrir(file, signal, coordinates, channel)

        min_max_val_signal = (signal.min(), signal.max())
        norm_signal = self.normalizer.normalize(signal)
        self.store_norm_hrir(file, norm_signal, min_max_val_signal,
                             coordinates, channel)

        feature = self.extractor.extract(signal)
        self.store_features(file, feature, coordinates, channel)

        min_max_val_feature = (feature.min(), feature.max())
        norm_feature = self.normalizer.normalize(feature)  
        self.store_norm_features(file, norm_feature, min_max_val_feature, 
                                 coordinates, channel)                
    
    def store_hrir(self, file_name, signal, coordinates, channel): 
        """
        Specific for each HRIR's in the dataset.
        """
        self.hrir_to_store[file_name] = {
            "signal": signal, "coordinates": coordinates,
            "channel": channel
        }
        
    def store_norm_hrir(self, file_name, norm_signal, min_max_val, 
                        coordinates, channel):
        """
        Specific for each HRIR's in the dataset. (hrir signal normalized)
        """
        self.norm_hrir_to_store[file_name] = {
            "norm_signal": norm_signal, "min_max": min_max_val, 
            "coordinates": coordinates, "channel": channel
        }

    def store_features(self, file_name, feature, coordinates, channel): 
        """
        Specific for each HRTF's in the dataset.
        """
        self.features_to_store[file_name] = {
            "feature": feature, "coordinates": coordinates, 
            "channel": channel
        }

    def store_norm_features(self, file_name, norm_feature, min_max_val, 
                            coordinates, channel):
        """
        Specific for each HRTF's in the dataset. (hrtf signal normalized)
        """
        self.norm_feature_to_store[file_name] = {
            "norm_feature": norm_feature, "min_max": min_max_val, 
            "coordinates": coordinates, "channel": channel
        }


    def elev_azimuth_channel(self, file_path):
        """
        Extracts the elevation and azimuth coordinates and channel from the hrir file's names. 
        """
        file_path = file_path[:len(file_path)-5]        # removes: a.wav 
        file_path = file_path.split("e")                # examples: L-10e265a ; L-10e265a; L0e015a
        elev = int(file_path[0][1:len(file_path[0])])   # removes the channel: L or R
        azimuth = int(file_path[1])
        channel = file_path[0][0]
        if channel == "L": 
            channel = "left"
        elif channel == "R": 
            channel = "right"
        return elev, azimuth, channel
    
    
    def is_padding_necessary(self, signal): 
        if len(signal) < self._num_samples: 
            return True
        else: 
            return False
        
    def apply_padding(self, signal): 
        num_missing_samples = self._num_samples - len(signal)
        padded_signal_r = self.zero_padder.right_pad(signal, num_missing_samples)
        padded_signal_l = self.zero_padder.right_pad(signal, num_missing_samples)
        return padded_signal_r, padded_signal_l



if __name__ == "__main__": 
    frame_size = 256
    hop_size = 64
    sample_rate = 44100
    mono = True

    # Directories: 
    # might want to add: "\\Users\\juanp\\OneDrive - Universidad Católica de Chile\\Desktop\\iPre\\iPre HTRFs Interpolation\\
    # also: "hrtf_database_interpolation\\
    DATASETS_SAVE_DIR = "dataset_pkl_file"
    FILES_DIR = "hrtf_dataset" 

    loader = Loader(sample_rate, mono)
    padder = ZeroPadder()
    log_spectogram_extractor = LogSpectogramExtractor(frame_size, hop_size)
    min_max_norm = MinMaxNormalizer(0, 1)
    saver = Saver(DATASETS_SAVE_DIR)

    preprocessing_pipeline = PreProcessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectogram_extractor
    preprocessing_pipeline.normalizer = min_max_norm
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)