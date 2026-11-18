"""Prepare editable SVG boards for Figma. Does not connect to or author in Figma."""
from pathlib import Path
from html import escape

DEST=Path(__file__).parent
scenes=[
 ('01-opening','00–03 SEC','Different expressions.','One connection.','Two-way sign language translation. A little closer, together.', ['SIGN TO SPEECH','SPEECH TO SIGN'], 'One thoughtful space', ['Your hands → words','Your words → signs']),
 ('02-sign-to-speech','03–08 SEC','Your hands.','Your voice.','Express yourself naturally. Keep the conversation moving.', ['EDIT · COPY · UNDO','LISTEN · GESTURE SHORTCUTS'], 'Live translation', ['Camera + image recognition','Framing · Mirror · Light & contrast','Sign stability → editable transcript']),
 ('03-speech-to-sign','08–12 SEC','Say it','your way.','Every conversation can begin a little differently.', ['QUICK PHRASES','EDITABLE MESSAGE'], 'Four ways to begin', ['Record voice · Live microphone','Upload audio · Type your words','Your message → sign sequence']),
 ('04-ray','12–17 SEC','Meet','Ray.','Your expressive 3D signing companion.', ['SIGNING + FINGERSPELLING','EXPRESSIVE FACE + HANDS'], 'A closer look, at your pace', ['Replay · Loop · Speed · Seek','Studio · Front · Hands views','Rotate · Zoom · Expand']),
 ('05-languages','17–21 SEC','More ways','to connect.','Language, expression and context come together.', ['DIALECT SELECTION','REFERENCE VIDEO + CUE CARDS'], 'Beyond words alone', ['English · Hindi · Kannada','Hands + face + body','Spatial cues · Directional signing']),
 ('06-library','21–26 SEC','A library that','grows with you.','Build your vocabulary. Make recognition your own.', ['POSTER REFERENCES','PERSONAL SIGN COLLECTION'], 'Capture → review → train', ['Webcam bursts · Image/folder import','Review labels · Manage samples','Train your recognition model']),
]
def txt(x,y,s,size=32,fill='#F4F7EF',weight=400):
 return f'<text x="{x}" y="{y}" fill="{fill}" font-family="Segoe UI, Inter, sans-serif" font-size="{size}" font-weight="{weight}">{escape(s)}</text>'
def base():
 return ['<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">','<rect width="1920" height="1080" fill="#102E2D"/>', '<circle cx="1570" cy="420" r="480" fill="#163B34"/>','<circle cx="1580" cy="420" r="360" fill="none" stroke="#33594A" stroke-width="2"/>', txt(100,105,'signify.',46,'#D6EFAE',700)]
for slug,timing,a,b,desc,tags,title,items in scenes:
 p=base(); p += [txt(1580,95,timing,21,'#A2BCB1'),txt(100,260,'EVERY CONVERSATION MATTERS',20,'#D6EFAE',600),txt(94,395,a,83,'#F4F7EF',700),txt(94,498,b,83,'#D6EFAE',700),txt(100,590,desc,28,'#B7CCC0')]
 for i,tag in enumerate(tags):
  y=690+i*65
  p += [f'<rect x="100" y="{y}" width="640" height="50" rx="25" fill="#24473C" stroke="#4D6E52"/>',txt(125,y+33,tag,22,'#D6EFAE',500)]
 p += ['<rect x="1000" y="260" width="820" height="570" rx="35" fill="#0B2525" opacity="0.4"/>','<rect x="980" y="240" width="820" height="570" rx="35" fill="#204238" stroke="#54765C"/>',txt(1030,315,title,34,'#E3F1D3',600)]
 for i,item in enumerate(items):
  y=370+i*125
  p += [f'<rect x="1025" y="{y}" width="730" height="96" rx="18" fill="#D6EFAE"/>',txt(1060,y+59,item,28,'#173A2D',500)]
 p += ['<path d="M100 950 H1820" stroke="#4D6E52"/>',txt(100,1008,'A bridge between signs and words.',25,'#A2BCB1'),txt(1450,1008,'SIGNIFY / FEATURE FILM',20,'#D6EFAE'),'</svg>']
 (DEST/f'{slug}.svg').write_text('\n'.join(p),encoding='utf-8')
p=base()
p += ['<circle cx="790" cy="490" r="340" fill="none" stroke="#73945F" stroke-width="2" opacity="0.35"/>','<circle cx="1130" cy="490" r="340" fill="none" stroke="#B4D48F" stroke-width="2" opacity="0.35"/>',txt(1530,95,'26–30 SEC',21,'#A2BCB1'),txt(630,305,'DIFFERENT EXPRESSIONS. ONE CONNECTION.',22,'#B9D39D',600),txt(417,480,'Every voice',126,'#F4F7EF',700),txt(615,625,'belongs.',126,'#D6EFAE',700),'<rect x="652" y="727" width="616" height="90" rx="45" fill="#D6EFAE"/>',txt(712,784,'Start your next conversation.',31,'#173A2D',600),txt(664,965,'A bridge between signs and words.',28,'#B7CCC0'),'</svg>']
(DEST/'07-every-voice-belongs.svg').write_text('\n'.join(p),encoding='utf-8')
print('Prepared seven editable 1920 × 1080 SVG boards for Figma import.')
