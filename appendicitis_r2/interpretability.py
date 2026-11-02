"""Research Grad-CAM visualization; not a validated localization model."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from .data import load_image
from .model import create_model


def gradcam(checkpoint,image_path,out,roi=None):
    from PIL import Image
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    c=cp['config'];model=create_model(c)
    if not hasattr(model,'gradcam_target'):
        raise ValueError('Grad-CAM supported for r2_cnn only')
    model.load_state_dict(cp['model_state']);model.eval()
    image=load_image(image_path,c,roi=roi)
    x=torch.tensor(np.repeat(image[None,None],int(c['data']['channels']),axis=1),dtype=torch.float32)
    activation=[];gradient=[]
    hook_f=model.gradcam_target().register_forward_hook(lambda _m,_i,o:activation.append(o))
    hook_g=model.gradcam_target().register_full_backward_hook(lambda _m,_gin,gout:gradient.append(gout[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits=model(x);cls=int(logits.argmax(1).item())
        logits[0,cls].backward()
        weights=gradient[-1].detach().mean(dim=(-2,-1),keepdim=True)
        cam=torch.relu((weights*activation[-1].detach()).sum(dim=1))[0].cpu().numpy()
        cam=cv2.resize(cam,(image.shape[1],image.shape[0]))
        if cam.max()>cam.min():cam=(cam-cam.min())/(cam.max()-cam.min())
        else:cam=np.zeros_like(cam)
        heat=cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET)
        grey=cv2.cvtColor(np.uint8(image*255),cv2.COLOR_GRAY2BGR)
        overlay=cv2.addWeighted(grey,0.55,heat,0.45,0)
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(out),overlay)
        return {'predicted_class':cls,'probability_positive':float(logits.softmax(-1)[0,1].item())}
    finally:
        hook_f.remove();hook_g.remove()
