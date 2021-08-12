import muda
import jams
import glob
import pathlib
class AudioManipulation:
   
  

    def __init__(self):
        pass

    def time_stretching(self, file, suffix_file, stretching_array):
        jam = jams.JAMS()
        jam = muda.load_jam_audio(jam, file)
        pth = pathlib.Path(file)
        parent = pth.parent
        file_name = pth.stem
        suffix = pth.suffix
        bg_transform = muda.deformers.TimeStretch(stretching_array)
        for i, jam_out in enumerate(bg_transform.transform(jam)):
             muda.save('{}/{}_{}_{:02d}{}'.format(parent, file_name, suffix_file,i, suffix),'{}/{}_{}_{:02d}.{}'.format(parent, file_name, suffix_file,i, "jams"),jam_out)

    def pitch_shifting(self, file, suffix_file, shifting_array):
        jam = jams.JAMS()
        jam = muda.load_jam_audio(jam, file)
        pth = pathlib.Path(file)
        parent = pth.parent
        file_name = pth.stem
        suffix = pth.suffix
        bg_transform = muda.deformers.PitchShift(shifting_array)
        for i, jam_out in enumerate(bg_transform.transform(jam)):
             muda.save('{}/{}_{}_{:02d}{}'.format(parent, file_name, suffix_file,i, suffix),'{}/{}_{}_{:02d}.{}'.format(parent, file_name, suffix_file,i, "jams"),jam_out)

    def dynamic_range_compression(self, file, suffix_file, presets):
        jam = jams.JAMS()
        jam = muda.load_jam_audio(jam, file)
        pth = pathlib.Path(file)
        parent = pth.parent
        file_name = pth.stem
        suffix = pth.suffix
        bg_transform = muda.deformers.DynamicRangeCompression(presets)
        for i, jam_out in enumerate(bg_transform.transform(jam)):
             muda.save('{}/{}_{}_{:02d}{}'.format(parent, file_name, suffix_file,i, suffix),'{}/{}_{}_{:02d}.{}'.format(parent, file_name, suffix_file,i, "jams"),jam_out)
    def background_noise_addition(self, file, suffix_file, bg_noises):
        # create an empty jam
        jam = jams.JAMS()
        jam = muda.load_jam_audio(jam, file)
        pth = pathlib.Path(file)
        parent = pth.parent
        file_name = pth.stem
        suffix = pth.suffix
        bg_transform = muda.deformers.BackgroundNoise(n_samples=1, files=bg_noises)
        for i, jam_out in enumerate(bg_transform.transform(jam)):
             muda.save('{}/{}{}{:02d}{}'.format(parent, file_name, suffix_file,i, suffix),'{}/{}_{}_{:02d}.{}'.format(parent, file_name, suffix_file,i, "jams"),jam_out)

if __name__ == "__main__":
    y = AudioManipulation()
  
    files = glob.glob("../data/train/wav/*/*.wav")
    bg_noises=["173955__saphe__street-scene-3.wav","207208__jormarp__high-street-of-gandia-valencia-spain.wav"]
    for i in files:
        # now this is all the files.
        y.time_stretching(i, "ts", [0.81, 0.93])
        y.pitch_shifting(i, "ps", [-2, -1, 1, 2])
        y.dynamic_range_compression(i, "drc", ["radio",  "music standard"])
        y.background_noise_addition(i, "bn", bg_noises)
