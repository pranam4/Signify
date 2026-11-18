"""Reproducible 30-second Signify feature film. No external media required."""
from pathlib import Path
import math
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'frontend/public/media/signify-30s.mp4'
W,H,FPS = 1280,720,30
INK='#102e2d'; LIME='#d6efae'; WHITE='#f4f7ef'; MUTED='#a2bcb1'
FONTS=Path('C:/Windows/Fonts')
def font(size,bold=False): return ImageFont.truetype(str(FONTS/('segoeuib.ttf' if bold else 'segoeui.ttf')),size)
def text(d,xy,s,size=22,fill=WHITE,bold=False): d.text(xy,s,font=font(size,bold),fill=fill)
def pill(d,x,y,s,active=False):
    width=d.textlength(s,font=font(17))+28
    d.rounded_rectangle((x,y,x+width,y+37),18,fill=LIME if active else '#23453e',outline='#476751')
    text(d,(x+14,y+6),s,17,INK if active else '#c8ddce')
    return width+10
def card(d,box,fill='#1c3b35'): d.rounded_rectangle(box,22,fill=fill,outline='#3c5b4b',width=1)
def line(d,a,b,fill=LIME,width=4): d.line((a,b),fill=fill,width=width)
def hand(d,cx,cy,s=1):
    pts=[(-35,58),(-54,16),(-46,7),(-26,29),(-29,-35),(-18,-42),(-10,8),(-7,-57),(5,-59),(10,7),(20,-44),(31,-41),(27,14),(43,-22),(54,-16),(43,44),(20,73),(-13,77)]
    coords=[(cx+x*s,cy+y*s) for x,y in pts]
    d.polygon(coords,fill='#305747'); d.line(coords+[coords[0]],fill=LIME,width=3)
    for x,y in pts[::2]: d.ellipse((cx+x*s-3,cy+y*s-3,cx+x*s+3,cy+y*s+3),fill=LIME)
def panel(d,label):
    card(d,(640,155,1205,590)); text(d,(670,178),label,16,MUTED)
    d.ellipse((1168,186,1175,193),fill=LIME)
def scene(t):
    n=min(5,int(t/5)); u=t-n*5
    im=Image.new('RGB',(W,H),INK); d=ImageDraw.Draw(im)
    for i in range(7):
        x=1080+int(18*math.sin(t*.4)); r=110+i*54
        d.ellipse((x-r,320-r,x+r,320+r),outline='#193a34',width=1)
    text(d,(65,40),'signify.',32,LIME,True)
    text(d,(960,53),'A LITTLE CLOSER, TOGETHER',12,MUTED)
    labels=['CONNECTION, REIMAGINED','YOUR HANDS. YOUR VOICE.','SPEAK. TYPE. CONNECT.','MEET YOUR COMPANION','BUILT AROUND YOU','SMALL SIGNS. BIG POSSIBILITIES.']
    text(d,(65,170),f'0{n+1}  /  {labels[n]}',14,LIME,True)
    titles=[['Every conversation','deserves a bridge.'],['Turn signs','into conversation.'],['Your words.','A new way to see.'],['Say hello','to Ray.'],['More expression.','More connection.'],['Learn it.','Make it yours.']]
    offset=int(25*(1-min(1,u/.6))**3)
    for j,s in enumerate(titles[n]): text(d,(62,220+j*66+offset),s,47,WHITE if j==0 else LIME,True)
    descriptions=[['Two-way sign language translation.','One thoughtful space to connect.'],['Live camera recognition becomes','editable text and spoken words.'],['Record your voice, use the live mic,','upload audio, or simply type.'],['A 3D signing companion with expressive','gestures and alphabet fingerspelling.'],['English, Hindi and Kannada.','Spatial cues and facial expressions.'],['Explore reference posters. Capture samples.','Import, review labels and train your model.']]
    for j,s in enumerate(descriptions[n]): text(d,(65,385+j*31),s,21,MUTED)
    tags=[['SIGNS → SPEECH','SPEECH → SIGNS'],['EDIT · COPY · UNDO','LISTEN ALOUD'],['VOICE + AUDIO','QUICK PHRASES'],['REPLAY · LOOP · SPEED','ROTATE + ZOOM'],['DIALECT SELECTION','GESTURE SHORTCUTS'],['PERSONAL SIGN LIBRARY','CUSTOM TRAINING']]
    x=65
    for tag in tags[n]: x+=pill(d,x,495,tag)
    if n==0:
        panel(d,'THE TRANSLATION STUDIO')
        hand(d,775,330,1.05)
        for i in range(15):
            h=15+45*abs(math.sin(t*2+i*.4)); x=978+i*10
            d.rounded_rectangle((x,340-h/2,x+5,340+h/2),3,fill=LIME)
        text(d,(875,325),'↔',45,LIME)
        text(d,(692,472),'Your hands',26,WHITE,True); text(d,(987,472),'Your voice',26,WHITE,True)
        text(d,(693,527),'Different ways to express. One conversation.',18,MUTED)
    elif n==1:
        panel(d,'SIGN → TEXT & SPEECH')
        d.rounded_rectangle((670,225,915,480),14,fill='#112b29')
        hand(d,790,330+6*math.sin(t*2),1.1)
        for x,y in [(680,236),(884,236),(680,447),(884,447)]:
            line(d,(x,y),(x+20,y)); line(d,(x,y),(x,y+20))
        card(d,(940,260,1180,430),'e' if False else '#e4efcf')
        text(d,(960,280),'LIVE TRANSLATION',12,'#4e6d46',True)
        text(d,(958,325),'Hello!',42,INK,True)
        text(d,(960,389),'Text + spoken output',14,'#4e6d46')
        text(d,(674,510),'Camera guides · Mirror · Image analysis',17,MUTED)
        text(d,(674,543),'Light & contrast controls · Sign stability',17,MUTED)
    elif n==2:
        panel(d,'YOUR MESSAGE')
        card(d,(672,232,1175,355),'#e4efcf')
        msg='Hello, thank you!'[:min(17,int(u*14))]
        text(d,(700,264),msg,34,INK,True)
        for i in range(35):
            h=12+52*abs(math.sin(t*3+i*.65)); x=714+i*12
            d.rounded_rectangle((x,410-h/2,x+5,410+h/2),3,fill=LIME)
        x=678
        for tag in ['Hello','Thank you','Please help']: x+=pill(d,x,475,tag,tag=='Hello')
        text(d,(679,540),'Editable transcript → sign sequence',20,MUTED)
    elif n==3:
        panel(d,'RAY / INTERACTIVE 3D SIGNING')
        # Editorial character illustration, not a simulation of a real sign.
        d.ellipse((821,469,1059,503),fill='#294a40')
        d.rounded_rectangle((873,345,1007,483),33,fill='#458d77')
        d.ellipse((893,246,987,355),fill='#ddb08a')
        d.pieslice((889,235,992,326),180,360,fill='#302620')
        for x in [922,956]: d.ellipse((x,294,x+7,302),fill=INK)
        d.arc((926,310,959,334),0,180,fill='#8c5746',width=3)
        line(d,(885,366),(828,421),'#ddb08a',18); line(d,(998,366),(1050,320),'#ddb08a',18)
        hand(d,1060,282,.38)
        text(d,(684,234),'RAY',22,LIME,True)
        text(d,(686,270),'Your personal',17,MUTED); text(d,(686,294),'signing companion',17,MUTED)
        text(d,(690,525),'Studio / Front / Hands',20,WHITE)
        text(d,(690,558),'Seek any moment · Expand the stage',16,MUTED)
    elif n==4:
        panel(d,'COMMUNICATION WITH CONTEXT')
        for i,(s,sub) in enumerate([('English','Listen in your language'),('Hindi','Editable multilingual transcripts'),('Kannada','Voice and text input')]):
            y=232+i*89; card(d,(674,y,1172,y+73),'#25483d')
            text(d,(695,y+10),s,24,LIME,True); text(d,(695,y+43),sub,15,MUTED)
        text(d,(680,529),'Face + hands + body · Directional signing',17,MUTED)
        text(d,(680,556),'Reference videos & cue cards',17,MUTED)
    else:
        panel(d,'YOUR SIGN LIBRARY')
        refs=sorted((ROOT/'frontend/public/signs/ref').glob('*.jpg'))[:3]
        for i,p in enumerate(refs):
            x=677+i*166; card(d,(x,238,x+150,414),'#f0f3e6')
            thumb=Image.open(p).convert('RGB'); thumb.thumbnail((134,135)); im.paste(thumb,(x+(150-thumb.width)//2,246))
            text(d,(x+12,386),p.stem.replace('_',' ').title()[:15],14,INK,True)
        text(d,(680,447),'CAPTURE → REVIEW → TRAIN',23,LIME,True)
        text(d,(680,491),'Build recognition around your vocabulary.',18,MUTED)
        pill(d,680,536,'Start your next conversation',True)
    d.line((65,635,1205,635),fill='#385548',width=1)
    d.line((65,635,65+int(1140*t/30),635),fill=LIME,width=3)
    text(d,(65,659),'A bridge between signs and words.',16,MUTED)
    text(d,(1000,659),'SIGNIFY  /  EXPLORE',14,LIME,True)
    # Dip-to-color transitions and opening/closing fade.
    fade=min(1,u/.25,(5-u)/.22,t/.35,(30-t)/.45)
    if fade<1: im=Image.blend(Image.new('RGB',(W,H),INK),im,max(0,fade))
    return im

OUT.parent.mkdir(parents=True,exist_ok=True)
container=av.open(str(OUT),'w')
video=container.add_stream('libx264',rate=FPS); video.width=W; video.height=H; video.pix_fmt='yuv420p'; video.options={'crf':'20','preset':'fast'}
audio=container.add_stream('aac',rate=48000); audio.layout='stereo'
# Original soft electronic score: warm chords, bell arpeggio, restrained pulse.
sr=48000; samples=np.arange(sr*30)/sr; music=np.zeros_like(samples)
chords=[[130.81,164.81,196],[110,130.81,164.81],[87.31,110,130.81],[98,123.47,146.83]]
for beat in range(60):
    start=beat*.5; local=samples-start; mask=(local>=0)&(local<1.8); q=local[mask]
    chord=chords[(beat//8)%4]; note=chord[beat%3]*4
    music[mask]+=.10*np.sin(2*np.pi*note*q)*np.exp(-q*5)
    if beat%2==0: music[mask]+=.09*np.sin(2*np.pi*(60*q+2*(1-np.exp(-q*30))))*np.exp(-q*14)
for k in range(8):
    start=k*4; mask=(samples>=start)&(samples<min(start+4,30)); q=samples[mask]-start
    for freq in chords[k%4]: music[mask]+=.035*np.sin(2*np.pi*freq*q)*np.minimum(q*3,1)*np.minimum((4-q)*2,1)
music*=np.minimum(samples,1)*np.minimum((30-samples)/1.5,1)
for i in range(FPS*30):
    frame=av.VideoFrame.from_image(scene(i/FPS)); frame.pts=i
    for packet in video.encode(frame): container.mux(packet)
    a=music[i*1600:(i+1)*1600].astype('float32')
    af=av.AudioFrame.from_ndarray(np.stack([a,a]),format='fltp',layout='stereo'); af.sample_rate=sr; af.pts=i*1600
    for packet in audio.encode(af): container.mux(packet)
    if i%150==0: print(f'Rendered {i//30}/30 seconds',flush=True)
for packet in video.encode(): container.mux(packet)
for packet in audio.encode(): container.mux(packet)
container.close()
scene(16).save(ROOT/'frontend/public/media/signify-poster.jpg')
print(OUT)
