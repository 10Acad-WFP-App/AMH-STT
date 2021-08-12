import sys
import os
import librosa as lb
from librosa.core import audio
import pandas as pd
from json import dump
sys.path.insert(0, '../scripts')
try:
    from audio_explorer import AudioExplorer
except:
    from scripts.audio_explorer import AudioExplorer

try:
    from logger_creator import CreateLogger
except:
    from scripts.logger_creator import CreateLogger

e_logger = CreateLogger('AudioPreprocessor')
e_logger = e_logger.get_default_logger()


class AudioLoader(AudioExplorer):
    def __init__(self, directory:str, audio_dir:str=r'/wav/*.wav', tts_file:str=r'/trsTrain.txt'):
        self.label_target = {}
        try:
            super().__init__(directory,audio_dir,tts_file)
            e_logger.info('Successfully Inherited AudioExplorer Class')
        except Exception as e:
            e_logger.exception("Failed Inheriting")
        self.load_label_target()
        # so now we want to load the audio and transcription
        # but first we need to change the load
                
    def load_audio(self):
        
        try:
            audio_name = []
            audio_mode = []
            audio_frequency = []
            audio_ts_data = []
            audio_duration = []
            has_TTS = []
            tts = []
            
            for audio_file in self.audio_files_dir_list:
                audio_data, audio_freq = lb.load(audio_file,sr=None)
                name = audio_file.split('wav')[-2]
                name = name[1:-1].strip()
                audio_name.append(name)
                # Time in seconds
                audio_duration.append(round(lb.get_duration(audio_data,sr=16000),3))
                # Audio Sampling Rate
                audio_frequency.append(audio_freq)
                # Audio Mode (Mono, Stereo)
                audio_mode.append('Mono' if len(audio_data.shape) == 1 else 'Stereo')
                # Audio time series data
                audio_ts_data.append(audio_data)
                # TTS
                tts_status = self.check_tts_exist(name)
                has_TTS.append(tts_status)
                # Add Transcription
                if(tts_status):
                    tts.append(self.tts_dict[name])
                else:
                    tts.append(None)
                
            self.df = pd.DataFrame()
            self.df['Name'] = audio_name
            self.df['Duration'] = audio_duration
            self.df['Channels'] = audio_mode
            self.df['SamplingRate'] = audio_frequency
            self.df['TimeSeriesData'] = audio_ts_data
            self.df['HasTTS'] = has_TTS
            self.df['TTS'] = tts

        except Exception as e:
            e_logger.exception('Failed to Load Audio Files')

    def load_label_target(self):
        for audio_file in self.audio_files_dir_list:
            name = audio_file.split('wav')[-2]
            name = name[1:-1].strip()
            tts_status = self.check_tts_exist(name)
                # Add Transliteration
            if(tts_status):
                self.label_target[name] = self.tts_dict[name]

    def export_tts(self, directory: str) -> None:
        #export to valid and train json files
        partition_index =len(self.label_target)*0.8
        train_dict = {key: self.label_target[key] for i,key in zip(range(1000),self.label_target.keys()) if i < partition_index}
        valid_dict = {key: self.label_target[key] for i,key in zip(range(1000),self.label_target.keys()) if i >= partition_index}
        try:
            with open(os.path.join(directory,"train.json"), "w") as export_file:
                dump(train_dict, export_file, indent=4, sort_keys=True)
            with open(os.path.join(directory,"valid.json"), "w") as export_file:
                dump(valid_dict, export_file, indent=4, sort_keys=True)

            e_logger.info(
                f'Successfully Exported Transliteration as JSON file to {directory}" train.json and test.json".')

        except FileExistsError as e:
            e_logger.exception(
                f'Failed to create {directory} train or test.json, it already exists.')
        except Exception as e:
            e_logger.exception('Failed to Export Transliteration as JSON File.')

    def get_audio_info(self) -> pd.DataFrame:
        try:
            return self.df.drop(columns=['TTS','TimeSeriesData'],axis=1)
        except Exception as e:
            e_logger.exception('Failed to return Audio Information')

    def get_audio_info_with_data_tts(self) -> pd.DataFrame:
        try:
            self.df
        except Exception as e:
            e_logger.exception('Failed to return Audio Information')

    def get_audio_info_with_data(self) -> pd.DataFrame:
        try:
            return self.df.drop('TTS',axis=1)
        except Exception as e:
            e_logger.exception('Failed to return Audio Information')

if __name__ == "__main__":
    al = AudioLoader(directory='../data/train')
    print(al.export_tts("./"))