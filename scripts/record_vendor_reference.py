"""Record a physical external-policy probe with explicit provenance captions."""
import argparse
import json
import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vendor_reference_probe import VendorGraph, run


class Recording:
    def __init__(self, output, live):
        self.output=output;self.live=live;self.renderer=None;self.viewer=None
        self.threads=[];self.frames=0;self.writer=None;self.start=None

    def __call__(self, info, data, step, phase, hold, monitor):
        if self.renderer is None:
            self.renderer=mujoco.Renderer(info.model,height=480,width=640)
            self.camera=mujoco.MjvCamera();self.camera.distance=2.9
            self.camera.azimuth=125;self.camera.elevation=-20
            self.options=mujoco.MjvOption();self.options.geomgroup[3]=0
            self.writer=imageio.get_writer(self.output/'adapted_vendor_recovery.mp4',fps=25)
            if self.live:
                from mujoco import viewer
                existing=set(threading.enumerate());self.viewer=viewer.launch_passive(info.model,data)
                self.threads=[t for t in threading.enumerate() if t not in existing and t.name.endswith('(_launch_internal)')]
                self.viewer.cam.distance=2.9;self.viewer.cam.azimuth=125;self.viewer.cam.elevation=-20
                self.viewer.opt.geomgroup[3]=0
            self.start=time.monotonic()
        self.camera.lookat[:]=[data.qpos[0],data.qpos[1],.55]
        if step%2==1:
            self.renderer.update_scene(data,camera=self.camera,scene_option=self.options)
            frame=Image.new('RGB',(640,592),(16,23,31));frame.paste(Image.fromarray(self.renderer.render()),(0,80))
            draw=ImageDraw.Draw(frame);font=ImageFont.load_default(size=16)
            draw.text((12,7),'ADAPTED VENDOR POLICY | EXTERNAL PRETRAINED',fill=(255,220,95),font=font)
            draw.text((12,29),'Supine -> standing | real MuJoCo dynamics',fill='white',font=font)
            draw.text((12,51),f'Time {data.time:5.2f} s | clean hold {hold:.2f} s | limits '+('PASS' if monitor.ok else 'FAIL'),fill='white',font=font)
            draw.text((12,568),'Not trained in our PPO experiments | no state teleporting',fill=(255,220,95),font=font)
            self.writer.append_data(np.asarray(frame));self.frames+=1
            if self.frames in [1,150,190,230,290,375]:frame.save(self.output/f'frame_{self.frames:04}.png')
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.cam.lookat[:]=self.camera.lookat
            self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  'EXTERNAL VENDOR POLICY + LOCAL ADAPTER\nNot our trained PPO\nSimulation time\nClean standing hold\nTrajectory limits',
                                  f'\n\n{data.time:.2f} s\n{hold:.2f} s\n'+('PASS' if monitor.ok else 'FAIL')))
            self.viewer.sync()
        if self.live:time.sleep(max(0.,(step+1)*.02-(time.monotonic()-self.start)))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            for thread in self.threads:
                thread.join(timeout=5.)
                if thread.is_alive():raise RuntimeError('Viewer failed to close')
        if self.renderer is not None:self.renderer.close()
        if self.writer is not None:self.writer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-dir',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--timescale',type=float,default=1.);p.add_argument('--output',required=True);p.add_argument('--live',action='store_true');p.add_argument('--expected',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'recorded_result.json').exists():raise ValueError('Choose fresh output')
    recording=Recording(out,a.live)
    try:result,states=run(VendorGraph(a.asset_dir),'vendor_actor',a.timescale,a.seed,'feedback',on_step=recording)
    finally:recording.close()
    expected=json.loads(Path(a.expected).read_text())
    crosschecks={key:result[key]==expected[key] for key in ['standing_return_pass','terminal_hold','max_clean_hold','max_pelvis_height','limits','target_adjustment_steps','max_target_adjustment_rad','final_checks','final_stance']}
    crosschecks['full_15_seconds']=recording.frames==375
    assert all(crosschecks.values()),crosschecks
    (out/'recorded_result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'recording_checks.json').write_text(json.dumps({'frames':recording.frames,'fps':25,'physical_resimulation':True,'checks':crosschecks},indent=2)+'\n')
    print(json.dumps({'checks':crosschecks,'viewer_closed':True}),flush=True)


if __name__=='__main__':main()
