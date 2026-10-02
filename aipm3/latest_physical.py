"""Reproduce the two physical inputs used by the recommended 9-feature N head."""
import subprocess
import numpy as np

def physical_updates(video_path):
    import imageio_ffmpeg
    executable=imageio_ffmpeg.get_ffmpeg_exe()
    audio=subprocess.run([executable,'-v','error','-threads','1','-i',str(video_path),
        '-vn','-ac','1','-ar','8000','-f','f32le','pipe:1'],capture_output=True,check=True)
    sound=np.frombuffer(audio.stdout,dtype='<f4');n=len(sound)//800
    rms=np.sqrt((sound[:n*800].reshape(n,800)**2).mean(axis=1)) if n else np.zeros(1)
    db=20*np.log10(rms+1e-8);active=db[db>db.max()-40]
    dynamic=float(np.quantile(active,.9)-np.quantile(active,.1)) if len(active) else 0.
    video=subprocess.run([executable,'-v','error','-threads','1','-i',str(video_path),
        '-an','-vf','fps=8,scale=160:90','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True)
    rgb=np.frombuffer(video.stdout,np.uint8).reshape(-1,90,160,3)
    gray=rgb.mean(3).astype(np.float32)/255
    delta=np.abs(np.diff(gray,axis=0)).mean((1,2))
    return {'phys__audio_dynamic_range_db':dynamic,'phys__motion_mean':float(np.median(delta)) if len(delta) else 0.}
