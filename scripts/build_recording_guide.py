#!/usr/bin/env python3
"""Build the illustrated phone/IMU guide from the four repository illustrations.

Run from any directory with Python, reportlab and Pillow installed. The default
output is arm_checker/guide/PhoneToPanda_Phone_IMU_Recording_Guide.pdf.
"""
from pathlib import Path
import math
import argparse
import reportlab
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'arm_checker' / 'guide' / 'illustrations'
OUTPUT = ROOT / 'arm_checker' / 'guide' / 'PhoneToPanda_Phone_IMU_Recording_Guide.pdf'
W, H = A4
M = 42
CW = W - 2*M
INK = colors.HexColor('#183039')
MUTED = colors.HexColor('#50656F')
TEAL = colors.HexColor('#087E85')
PALE = colors.HexColor('#EAF4F3')
ORANGE = colors.HexColor('#B66531')
WARM = colors.HexColor('#FBF2E8')
LINE = colors.HexColor('#CAD8DC')
WHITE = colors.white
FONTS = Path(reportlab.__file__).parent / 'fonts'
pdfmetrics.registerFont(TTFont('Field', str(FONTS/'Vera.ttf')))
pdfmetrics.registerFont(TTFont('Field-Bold', str(FONTS/'VeraBd.ttf')))
pdfmetrics.registerFontFamily('Field', normal='Field', bold='Field-Bold')
STYLES = {
 'body': ParagraphStyle('body', fontName='Field', fontSize=10.3, leading=15.1, textColor=INK),
 'small': ParagraphStyle('small', fontName='Field', fontSize=8.5, leading=12, textColor=MUTED),
 'bold': ParagraphStyle('bold', fontName='Field-Bold', fontSize=10.3, leading=15.1, textColor=INK),
 'head': ParagraphStyle('head', fontName='Field-Bold', fontSize=13.2, leading=18, textColor=INK),
 'white': ParagraphStyle('white', fontName='Field-Bold', fontSize=10.5, leading=15, textColor=WHITE),
}

class Guide:
 def __init__(self):
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  self.c = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=1)
  self.c.setTitle('PhoneToPanda | Phone + IMU recording guide')
  self.c.setAuthor('PhoneToPanda')
  self.c.setSubject('Right-arm phone placement, calibration, timed movements and paired recording procedure')
  self.n = 0
  self.bounds = []

 def text(self,x,y,text,size=10,color=INK,bold=False,align='left'):
  self.c.setFillColor(color); self.c.setFont('Field-Bold' if bold else 'Field',size)
  if align=='center': self.c.drawCentredString(x,H-y,text)
  elif align=='right': self.c.drawRightString(x,H-y,text)
  else: self.c.drawString(x,H-y,text)

 def p(self,text,x,y,width=CW,style='body'):
  para=Paragraph(text,STYLES[style]); _,height=para.wrap(width,H)
  assert y+height<790, (self.n,text[:80],y+height)
  para.drawOn(self.c,x,H-y-height)
  self.bounds.append({'page':self.n,'top':y,'bottom':y+height,'text':text})
  return y+height

 def box(self,x,y,w,h,fill=PALE,stroke=None,r=6):
  self.c.setFillColor(fill);self.c.setStrokeColor(stroke or fill)
  self.c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=bool(stroke))

 def line(self,x1,y1,x2,y2,color=LINE,width=1,dash=None):
  self.c.setStrokeColor(color);self.c.setLineWidth(width);self.c.setDash(dash or [])
  self.c.line(x1,H-y1,x2,H-y2);self.c.setDash([])

 def circle(self,x,y,r,color=TEAL,filled=True):
  self.c.setFillColor(color);self.c.setStrokeColor(color)
  self.c.circle(x,H-y,r,fill=int(filled),stroke=1)

 def arrow(self,a,b,color=TEAL,width=1.6):
  self.line(*a,*b,color,width)
  ang=math.atan2(b[1]-a[1],b[0]-a[0])
  for d in (-.45,.45):
   self.line(*b,b[0]-7*math.cos(ang+d),b[1]-7*math.sin(ang+d),color,width)

 def begin(self,section,title,subtitle):
  if self.n:self.c.showPage()
  self.n+=1
  self.c.bookmarkPage('page'+str(self.n));self.c.addOutlineEntry(title,'page'+str(self.n),0)
  self.c.setFillColor(TEAL);self.c.rect(0,H-8,W,8,fill=1,stroke=0)
  self.text(M,32,'PHONE TO PANDA  /  RECORDING FIELD GUIDE',8,MUTED,True)
  self.text(W-M,32,f'{self.n:02d} / 11',8,MUTED,align='right')
  self.text(M,66,section.upper(),9,TEAL,True)
  self.text(M,101,title,24,INK,True)
  self.p(subtitle,M,115,CW)
  self.line(M,801,W-M,801)
  self.text(M,820,'Right arm only  |  15 October 2026',8,MUTED)
  self.text(W-M,820,'Phone video + separate IMU session',8,MUTED,align='right')

 def image(self,name,top,width=CW,x=M):
  from PIL import Image
  path=ASSETS/name
  with Image.open(path) as im: height=width*im.height/im.width
  self.c.drawImage(str(path),x,H-top-height,width=width,height=height,mask='auto')
  return top+height

 def note(self,title,body,top,fill=PALE,height=None):
  hp=Paragraph(title,STYLES['bold']);_,hh=hp.wrap(CW-28,H)
  bp=Paragraph(body,STYLES['body']);_,bh=bp.wrap(CW-28,H)
  h=height or 25+hh+bh
  self.box(M,top,CW,h,fill)
  self.p(title,M+14,top+10,CW-28,'bold')
  self.p(body,M+14,top+14+hh,CW-28)
  return top+h

 def step(self,num,title,body,top):
  self.circle(M+10,top+10,10)
  self.text(M+10,top+13,str(num),9,WHITE,True,'center')
  end=self.p('<b>'+title+'</b> '+body,M+29,top,CW-29)
  return end+13

 def table(self,rows,top,widths,small=False):
  converted=[[Paragraph(str(cell),STYLES['bold' if i==0 else ('small' if small else 'body')])
              for cell in row] for i,row in enumerate(rows)]
  t=Table(converted,colWidths=widths,hAlign='LEFT')
  t.setStyle(TableStyle([
   ('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
   ('LINEBELOW',(0,0),(-1,0),1,TEAL),('LINEBELOW',(0,1),(-1,-1),.4,LINE),
   ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
   ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
  _,height=t.wrap(CW,H)
  assert top+height<786,(self.n,'table',top+height)
  t.drawOn(self.c,M,H-top-height)
  return top+height

 def check(self,label,top):
  self.c.setStrokeColor(TEAL);self.c.setLineWidth(1)
  self.c.rect(M,H-top-11,10,10,stroke=1,fill=0)
  return self.p(label,M+21,top-2,CW-21)+10

 def save(self):
  assert self.n==11
  self.c.save()

def build():
 g=Guide()
 # 1. Start with a useful overview, not a decorative cover.
 g.begin('Start here','Record your right arm',
  'A practical guide for filming a pick-and-place movement while your two right-arm sensors record the same take.')
 end=g.image('phone-placement.png',160,width=490,x=(W-490)/2)
 g.p('Illustrative setup. The rear phone camera faces you and the table. Hardware appearance is schematic; keep your existing wiring.',M,end+7,CW,'small')
 y=end+48
 y=g.step(1,'Set the view.', 'Fix the phone in landscape at your front-right; include your shoulder, elbow, wrist, hand, object and tray.',y)
 y=g.step(2,'Calibrate and check.', 'Open Arm Studio on your PC (page 10). Use the right upper-arm and forearm sensors; finish table alignment and mounted neutral.',y)
 y=g.step(3,'Record one pilot.', 'Start the phone, choose 30 seconds in Arm Studio, then start its 5-second countdown. Follow the 28-second sequence on page 5 and review both files.',y)
 g.note('Do the pilot before the full set',
  'One good paired take is the first goal. Only after it passes the checks on page 8 should you collect the 12-take plan on page 9.',y+1)
 g.p('Find it quickly: phone 2  |  sensors 3  |  start/stop 4  |  timing 5  |  movement 6  |  occlusion 7  |  checks 8  |  takes 9  |  saving 10  |  field checklist 11',M,757,CW,'small')

 # 2. Exact geometric guidance in a deterministic top view.
 g.begin('01 / Camera and workspace','Put the phone here',
  'Stand at the table. Place the phone diagonally in front of your RIGHT side, about 30-45 degrees away from directly in front of you.')
 g.box(M,162,CW,244,colors.HexColor('#F5F8F8'),LINE)
 g.text(M+15,182,'TOP VIEW - NOT TO SCALE',8,TEAL,True)
 g.box(145,220,220,90,colors.HexColor('#E1EBED'),LINE,3)
 g.text(252,238,'TABLE',9,MUTED,True,'center')
 g.box(278,263,18,18,ORANGE,r=1);g.box(210,258,38,26,PALE,TEAL,2)
 g.arrow((274,272),(251,272),TEAL)
 g.text(230,302,'20-30 cm task spacing',8,MUTED,align='center')
 g.circle(254,359,15,INK,False)
 g.line(224,341,284,341,INK,3)
 g.line(284,341,299,316,TEAL,4)
 g.arrow((254,337),(254,316),INK)
 g.p('YOU<br/>Facing the table',190,379,130,'small')
 g.p('Your right side',314,349,110,'small')
 g.box(383,191,48,25,INK,r=3)
 g.line(407,218,407,241,MUTED,2)
 g.line(407,241,392,252,MUTED,1)
 g.line(407,241,422,252,MUTED,1)
 g.line(382,198,167,217,TEAL,1,[3,3])
 g.line(383,211,281,368,TEAL,1,[3,3])
 g.arrow((381,225),(335,267),TEAL)
 g.p('PHONE<br/>Rear lens<br/>toward you',445,194,95,'small')
 g.p('Start 1.5-2 m away;<br/>adjust for full framing.',392,309,144,'small')
 y=g.table([
  ['Setting','Use this starting point'],
  ['Position','Stable tripod or fixed support. Lens near lower-chest height, tilted slightly downward toward the table.'],
  ['Camera','Rear camera, landscape, normal 1x view. Start with 1080p / 30 fps. Use 60 fps only if the preview stays bright and sharp; keep it consistent.'],
  ['Framing','Include the arm straight down AND the highest reach. Leave space above the shoulder and around the hand path. The wrist must never leave the frame.'],
  ['Light and focus','Light from the front or side; avoid a bright window behind you. Lock focus/exposure if available. Keep zoom and position fixed.'],
 ],427,[100,CW-100])
 g.note('Rehearse once while watching the phone preview',
  'Bend your elbow, reach to the cube, carry it to the tray and return your arm down. If anything is cropped, move the phone back and repeat the check.',y+16)

 # 3. Sensor identity and calibration.
 g.begin('02 / Sensors and calibration','Prepare the right arm',
  'Use only the two right-arm sensors for the arm estimate. Keep the ESP powered while moving the sensors from the table to your arm.')
 end=g.image('sensor-calibration.png',160,width=CW)
 for x,label in [(M,'A  TABLE ALIGNMENT'),(M+172,'B  SECURE MOUNTING'),(M+344,'C  NEUTRAL POSE')]:
  g.text(x,end+17,label,8,TEAL,True)
 y=end+33
 y=g.step(1,'Check identity and lengths.', 'Upper arm = ID 3 (board S4); forearm = ID 4 (S5). Move each alone to confirm. Enter your measured shoulder-to-elbow and elbow-to-wrist lengths in cm.',y)
 y=g.step(2,'Wait for startup readiness.', 'After powering the ESP, keep both boards still for at least 10 continuous seconds. Wait until both sensor cards show Fresh.',y)
 y=g.step(3,'Capture the table pose.', 'Lay both boards chip/+Z side up. Point both actual +X arrows toward the direction you will face while recording. Click Table alignment; hold until complete, at least 2 seconds.',y)
 y=g.step(4,'Mount, then capture neutral.', 'Mount one above the elbow and one on the forearm, clear of joints. Face the same direction the +X arrows pointed. Keep the right arm straight down; click Neutral pose and hold until complete, at least 3 seconds.',y)
 g.p('Check slow elbow bends and forward lifts. Repeat both calibration steps after remounting, an ESP restart, changed assignments or a changed input source. Board drawings illustrate orientation, not connector wiring or the axis layout of every sensor model.',M,y,CW,'small')

 # 4. Timing gesture and exact interface controls.
 g.begin('03 / Start and stop','Make a matched recording',
  'Phone video and the IMU session must capture the SAME physical movement. They are separate recordings; their clocks are not automatically synchronized.')
 end=g.image('timing-gesture.png',158,width=490,x=(W-490)/2)
 for x,label in [(M+81,'ARM DOWN'),(M+255,'BEND ELBOW'),(M+428,'ARM DOWN')]:
  g.text(x,end+17,label,8,TEAL,True,'center')
 g.p('Timing gesture: one gentle elbow bend to about a right angle, then return down. Keep the upper arm close to your side and the torso still. Repeat it near the end of every take.',M,end+30,CW)
 y=end+82
 y=g.step(1,'Phone first.', 'Press video Record. Say the take ID aloud, for example "P zero zero, clear view". Keep the phone fixed.',y)
 y=g.step(2,'Choose 30 seconds, then start.', 'In Arm Studio, select 30 seconds under Record for (default 15; 10 also available). Confirm LIVE and both Fresh. Click Start recording - 5s; return to neutral before the countdown reaches zero.',y)
 y=g.step(3,'Follow the sequence; let it save.', 'Start page 5 at countdown zero. Keep Arm Studio open: it automatically saves after about 30 seconds. Stop the phone separately, then check Saved sessions. Stop &amp; save remains available to finish early.',y)
 g.note('The two elbow gestures are alignment cues',
  'Match the elbow movements in video and sensor data afterward. The countdown gives preparation time; it does not synchronize the phone. Speaking the take ID helps match files.',y+2)

 # 5. Full timing script.
 g.begin('04 / Timed take','Follow this 28-second sequence',
  'Choose 30 seconds in Arm Studio. Be in neutral when its 5-second countdown reaches zero: that is 0 s below. The phone starts earlier and stops separately, so its file will be longer.')
 rows=[['Time','Your movement','Keep clear in the video'],
  ['0-3 s','Stand still with the right arm relaxed straight down.','Whole right arm and shoulder.'],
  ['3-6 s','Bend the elbow comfortably, then return down. Keep the upper arm still.','Timing gesture 1: elbow and wrist.'],
  ['6-8 s','Hold neutral without moving the straps or cables.','Right arm; no obstruction.'],
  ['8-11 s','Reach to the cube and grasp gently with your right hand.','Hand approaching and making contact.'],
  ['11-15 s','Lift about 8-10 cm and carry smoothly toward the tray.','Arm and hand path; optional occlusion only as described on page 7.'],
  ['15-18 s','Lower into the tray, release visibly, then move fingers away.','Object touching tray and fingers opening.'],
  ['18-21 s','Bring the hand back and return the arm to neutral.','Complete return path.'],
  ['21-24 s','Repeat the same elbow bend and return.','Timing gesture 2: elbow and wrist.'],
  ['24-28 s','Hold neutral and stay still. Remain still until the 30-second automatic save; then stop the phone.','Final pose and any residual drift.']]
 y=g.table(rows,168,[57,268,CW-325])
 y=g.note('Comfort and complete movements matter more than exact seconds',
  'These are rehearsal targets. Use a helper or spoken timer. Do not rush to fit. The 10- and 15-second options suit shorter checks, not this full sequence. Keep pacing consistent and leave the page open for automatic save.',y+17)
 g.p('For the pilot P00, use a clear view. One pick-and-place per take is enough. Reset the cube only after both recordings have stopped.',M,y+16,CW)

 # 6. Large movement storyboard.
 g.begin('05 / Movement detail','Reach, lift, carry, release',
  'Use a small light object and a shallow tray. Place them about 20-30 cm apart, within comfortable reach of your right hand.')
 end=g.image('pick-and-place.png',162,width=CW)
 g.p('Read the illustration from top-left to top-right, then bottom-left to bottom-right. The images show movement stages, not exact motion measurements.',M,end+9,CW,'small')
 y=end+46
 y=g.step(1,'Reach and grasp.', 'Keep your feet planted and torso upright. Approach slowly enough that the camera can see when your fingers contact the cube.',y)
 y=g.step(2,'Lift and carry.', 'Lift only enough to clear the table, about 8-10 cm. Move smoothly toward the tray; do not swing the whole body or reach behind yourself.',y)
 y=g.step(3,'Place and release.', 'Lower the cube, open your fingers clearly, then return the arm down. A small pause at placement makes the event easier to review.',y)
 g.note('Keep the shoulder reference stable',
  'The checker assumes a fixed shoulder. Move at the shoulder and elbow naturally, but avoid stepping, twisting the torso or leaning forward. If the reach needs a lean, bring the object and tray closer.',y+1)

 # 7. Partial occlusion, with unambiguous optical vs physical obstruction.
 g.begin('06 / Optional occluded takes','Hide the view, not the path',
  'First complete the clear pilot. In occluded takes, block the camera view of the moving forearm or hand briefly during the carry, not the physical reach.')
 g.box(M,166,CW,246,colors.HexColor('#F5F8F8'),LINE)
 g.text(M+14,187,'TOP VIEW - OPTICAL OBSTRUCTION ONLY',8,TEAL,True)
 g.box(102,228,242,90,colors.HexColor('#E1EBED'),LINE,3)
 g.circle(155,272,9,ORANGE)
 g.box(272,258,38,29,PALE,TEAL,2)
 g.arrow((173,272),(260,272),TEAL,2)
 g.text(216,307,'Physical carry path stays clear',8,INK,align='center')
 g.circle(203,366,14,INK,False)
 g.text(203,395,'YOU',9,INK,True,'center')
 g.box(467,239,43,24,INK,r=3)
 g.text(489,284,'PHONE',8,INK,True,'center')
 g.line(465,251,221,271,TEAL,1,[3,3])
 g.line(397,222,397,282,ORANGE,7)
 g.p('Card between lens<br/>and carry region',369,308,156,'small')
 g.arrow((418,300),(401,281),ORANGE,1)
 g.p('Keep the card outside the table workspace. A helper moves it into the camera view only during the middle of the carry.',M+14,201,CW-28,'small')
 y=g.table([
  ['Clear take','Occluded take'],
  ['No deliberate obstruction. Shoulder, elbow, wrist, object and tray stay visible.','Same phone, task and pacing. Hide only part of the moving forearm/hand for about 2 seconds, roughly 12-14 s in the script.'],
  ['Both elbow timing gestures remain visible.','Both timing gestures, pickup and placement MUST remain visible. Remove the obstruction before release.'],
  ['Use this for P00 and the clear trials.','Rehearse with a helper. If alone, a fixed narrow screen can work only after checking the camera preview and full reach.']
 ],434,[CW/2,CW/2])
 g.note('Do not improvise an occlusion that changes the task',
  'Do not move the phone, cover the lens with your hand, touch a sensor, hide the entire body or force a different reach. If the card blocks your movement, move the card farther from the working area.',y+17)
 g.p('A single camera cannot reveal the true hidden motion. These takes can test continuity and visible recovery; hidden-motion accuracy requires an independent reference.',M,754,CW,'small')

 # 8. Honest go/no-go criteria.
 g.begin('07 / Pilot review','Check P00 before recording more',
  'Review the phone clip and the saved IMU session. A moving graphic is a useful sanity check, but it does not establish physical accuracy or synchronized timing.')
 y=171
 for num,(title,body) in enumerate([
  ('Two real files exist.','The phone clip plays and the IMU session contains real LIVE data, not Demo. Keep both original files.'),
  ('The frame is complete.','Shoulder, elbow, wrist, hand, object and tray stay visible throughout the clear take. Both timing gestures are visible.'),
  ('The image is usable.','The hand and object are sharp enough to inspect; no severe blur, glare, dark shadows or camera shake hides the grasp.'),
  ('The stream stays usable.','Replay the IMU session and look for frozen poses, sudden resets, unexplained jumps or missing data. Both selected sensor timestamps should advance.'),
  ('Motion agrees qualitatively.','An elbow bend increases the displayed bend; a forward lift follows your forward motion. Return down and note any offset. Do not judge only by screen-left/screen-right.'),
  ('The pair can be aligned.','Identify the two elbow gestures in both recordings. If either gesture is missing or ambiguous, repeat the pilot before collecting the study set.')],1):
  y=g.step(num,title,body,y)
 g.note('About the yellow USB recovery message',
  'It records that recovery happened earlier in the connection; it is not proof of a current sensor fault. Preserve the warning and recovery counts. Investigate repeated errors or gaps before treating a recording as validated comparison data.',y+3,fill=WARM)
 g.text(M,608,'PILOT NOTES',9,TEAL,True)
 g.text(M,634,'Phone file:',10)
 g.line(M+75,637,W-M,637)
 g.text(M,663,'IMU session:',10)
 g.line(M+75,666,W-M,666)
 g.text(M,692,'Ready / repeat - reason:',10)
 g.line(M+150,695,W-M,695)
 g.p('If the pilot fails: keep the failed attempt, note the reason, correct the setup and record P00_attempt2. Do not hide a valid recording just because a later algorithm performs badly.',M,754,CW,'small')

 # 9. Small, balanced acquisition plan.
 g.begin('08 / Collection plan','One pilot, then twelve takes',
  'This is a small feasibility set, not a guaranteed training-data requirement. It gives six clear and six occluded takes for a later controlled comparison.')
 y=g.table([
  ['Block','Take order','Use'],
  ['Pilot','P00 - clear view','Check files, movement, camera framing and timing gestures. Keep outside the test set.'],
  ['Development','D01 clear\nD02 occluded\nD03 occluded\nD04 clear\nD05 clear\nD06 occluded'.replace('\n','<br/>'),'Use these six takes to choose processing, alignment and technical usability rules.'],
  ['Break','Stop. Check the setup. Repeat table alignment and mounted neutral.','Freeze the chosen processing and acceptance rules before viewing test results. Record any change in setup.'],
  ['Held-out test','T01 occluded\nT02 clear\nT03 clear\nT04 occluded\nT05 occluded\nT06 clear'.replace('\n','<br/>'),'Keep these six full takes out of tuning. Apply the same methods and rules chosen on development.']
 ],168,[86,162,CW-248])
 y=g.note('One take supports all three later analysis variants',
  'Record camera and IMUs together once. Later, use the same take for vision-only, IMU-only and combined processing. Do not collect three different performances and call them a matched comparison.',y+18)
 g.p('Keep phone placement, object, tray, lighting and pacing fixed within each block. Use whole-take or whole-block splits; do not randomly split neighboring frames between tuning and testing.',M,y+16,CW)
 g.p('Keep every attempt. Replace technically incomplete captures under the same condition with an attempt suffix. Record the reason for replacement. A drop or failed grasp can still be valuable if the recording itself is complete.',M,735,CW,'small')

 # 10. File pairing and what is not implemented.
 g.begin('09 / Save and match','Keep the original pair together',
  'The phone video and the downloaded IMU session belong to one take ID. The checker assigns its own session filename, so make the mapping explicit.')
 y=170
 y=g.step(1,'Save the IMU session.', 'Wait for the selected duration to finish and save, or click Stop &amp; save early. In Saved sessions, use Replay to inspect it and Download for the full JSON. Return to live input before the next take.',y)
 y=g.step(2,'Copy the original phone video.', 'Transfer the original MOV/MP4 without compression, trimming or frame-rate changes. Keep audio and timestamps in your private originals; use separate reviewed copies for sharing.',y)
 y=g.step(3,'Create one folder per take.', 'Copy both files into a take folder such as P00_clear. Preserve the original filenames, or use descriptive copies while noting their original names.',y)
 g.box(M,y+2,CW,111,PALE)
 g.text(M+14,y+24,'EXAMPLE FOLDER - NOT AN ACTUAL RECORDING',8,TEAL,True)
 g.text(M+14,y+45,'P00_clear/',11,INK,True)
 g.text(M+30,y+65,'phone_original.MOV',10)
 g.text(M+30,y+83,'imu_original_session.json',10)
 g.text(M+30,y+101,'notes.txt',10)
 y+=133
 y=g.p('<b>Write in the notes:</b> take ID; clear/occluded; original video filename; checker session ID; date and block; phone resolution/fps; measured arm lengths; sensor IDs; calibration completed; any warning, drift, reset, dropped object or repeated attempt.',M,y,CW)
 y=g.note('Do not pair an old video with a new IMU performance',
  'Repeating the same action later creates a different take. The camera and sensors must record the same physical movement, with the timing gestures preserved for later alignment.',y+17)
 g.p('<b>Checker boundary:</b> this tool saves paired data and estimates arm motion. It does not perform automatic camera synchronization, marker-free tracking, vision + IMU fusion or robot-policy training. The separate experimental pilot uses manual timing; the original vision tracker still requires printed markers.',M,y+17,CW)
 g.box(M,674,CW,60,PALE)
 g.text(M+13,691,'OPEN ARM STUDIO - FROM THE PHONETOPANDA REPOSITORY FOLDER',8,TEAL,True)
 g.text(M+13,709,'bash arm_checker/run.sh --port auto --protocol am',9.3,INK,True)
 g.text(M+13,725,'Then open http://127.0.0.1:8765/ on the same PC.',9.3,INK)
 g.p('Two IMUs provide orientation-based arm estimates using a fixed shoulder and your measured lengths. They do not independently measure absolute wrist position, fingers, object contact or grasp success.',M,747,CW,'small')

 # 11. A practical sheet to leave next to the laptop.
 g.begin('10 / Keep beside you','Recording-day checklist',
  'Use this page once you have read the setup instructions. Complete one clear pilot and review it before collecting the remaining takes.')
 y=167
 for label in [
  'Phone fixed at front-right; rear camera; landscape; 1080p at a consistent frame rate.',
  'Full shoulder, elbow, wrist, hand, cube, tray and arm-down pose fit in the preview.',
  'IDs 3 and 4 confirmed; measured lengths entered; mounts secure; cables slack.',
  'Table +X arrows pointed toward your recording-forward direction; mounted neutral captured facing that way.',
  'LIVE selected; both sensor cards Fresh; slow movements checked; no unexplained freeze.',
  'Take ID chosen; Record for = 30 seconds; helper or timer ready; page stays open.']:
  y=g.check(label,y)
 g.box(M,y+6,CW,54,TEAL)
 g.p('START PHONE  >  SELECT 30 s  >  START 5 s COUNTDOWN  >  NEUTRAL BY ZERO',M+14,y+16,CW-28,'white')
 y+=77
 y=g.table([
  ['Seconds','Action'],['0-3','Still'],['3-6','Elbow bend and return'],['6-8','Still'],
  ['8-11','Reach and grasp'],['11-15','Lift and carry'],['15-18','Place and release'],
  ['18-21','Return down'],['21-24','Elbow bend and return'],['24-28','Still']
 ],y,[80,CW-80],small=True)
 g.box(M,y+13,CW,39,TEAL)
 g.p('AUTO-SAVE AT 30 s  >  STOP PHONE  >  CHECK AND MATCH BOTH FILES',M+14,y+24,CW-28,'white')
 g.p('Generated instructional illustrations, not photos of your hardware or experimental evidence. Board shapes are schematic; use the actual printed axes on your sensors. The phone and sensor recordings remain separate files.',M,746,CW,'small')
 g.save()
 print(OUTPUT)

if __name__ == '__main__':
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output', type=Path, default=OUTPUT)
 args = parser.parse_args()
 OUTPUT = args.output.expanduser().resolve()
 build()
