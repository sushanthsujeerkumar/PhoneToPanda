#!/usr/bin/env python3
"""Build the recording guide: python build_guide.py [--output PATH].

Requires reportlab==4.4.9. All illustrations are generated as vector geometry.
No hardware, camera, model or network access is performed.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle

W, H = A4
M = 43
CW = W - 2 * M
INK = colors.HexColor('#1C2C37')
MUTED = colors.HexColor('#516571')
TEAL = colors.HexColor('#087F82')
LIGHT = colors.HexColor('#EDF5F4')
BLUE = colors.HexColor('#4278A1')
CLAY = colors.HexColor('#AD633E')
PALE = colors.HexColor('#F8F3ED')
LINE = colors.HexColor('#CDD9DD')

# ReportLab ships these fonts, so generation does not depend on host font setup.
FONT_DIR = Path(reportlab.__file__).parent / 'fonts'
pdfmetrics.registerFont(TTFont('GuideSans', str(FONT_DIR / 'Vera.ttf')))
pdfmetrics.registerFont(TTFont('GuideSans-Bold', str(FONT_DIR / 'VeraBd.ttf')))
pdfmetrics.registerFontFamily('GuideSans', normal='GuideSans',
                              bold='GuideSans-Bold')

STYLES = {
    'body': ParagraphStyle('body', fontName='GuideSans', fontSize=10.2,
                           leading=14.4, textColor=INK, spaceAfter=0),
    'small': ParagraphStyle('small', fontName='GuideSans', fontSize=8.5,
                            leading=11.6, textColor=MUTED),
    'label': ParagraphStyle('label', fontName='GuideSans-Bold', fontSize=9,
                            leading=12, textColor=INK),
    'heading': ParagraphStyle('heading', fontName='GuideSans-Bold', fontSize=13,
                              leading=17, textColor=INK),
    'white': ParagraphStyle('white', fontName='GuideSans-Bold', fontSize=10,
                            leading=13, textColor=colors.white),
}


class Guide:
    def __init__(self, output: Path):
        self.c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1,
                               invariant=1)
        self.c.setTitle('Right-arm IMU recording guide')
        self.c.setAuthor('PhoneToPanda')
        self.c.setSubject('Camera and two-sensor pilot recording procedure')
        self.page = 0

    def begin(self, section: str, title: str, subtitle: str):
        if self.page:
            self.c.showPage()
        self.page += 1
        c = self.c
        c.setFillColor(TEAL); c.rect(0, H - 12, W, 12, fill=1, stroke=0)
        self.text(M, 37, 'PHONE TO PANDA / RIGHT-ARM CHECKER', 8, MUTED, True)
        self.text(W-M, 37, f'{self.page:02d} / 09', 8, MUTED, align='right')
        self.text(M, 68, section.upper(), 9, TEAL, True)
        title_size = min(25, 25 * CW / pdfmetrics.stringWidth(title, 'GuideSans-Bold', 25))
        self.text(M, 101, title, title_size, INK, True)
        self.p(subtitle, M, 113, CW, 'body')
        c.setStrokeColor(LINE); c.line(M, 46, W-M, 46)
        self.text(M, H-29, 'Recording procedure v1.0 - 11 October 2026', 8, MUTED)
        self.text(W-M, H-29, 'Proposed pilot; physical validation pending', 8,
                  MUTED, align='right')

    def text(self, x, top, s, size=10, color=INK, bold=False, align='left'):
        c=self.c; c.setFillColor(color)
        c.setFont('GuideSans-Bold' if bold else 'GuideSans', size)
        if align == 'right': c.drawRightString(x, H-top, s)
        elif align == 'center': c.drawCentredString(x, H-top, s)
        else: c.drawString(x, H-top, s)

    def p(self, s, x, top, width=CW, style='body'):
        p=Paragraph(s, STYLES[style]); _,h=p.wrap(width,H)
        p.drawOn(self.c,x,H-top-h)
        return top+h

    def box(self, x, top, width, height, fill=LIGHT, stroke=None, radius=7):
        c=self.c; c.setFillColor(fill); c.setStrokeColor(stroke or fill)
        c.roundRect(x,H-top-height,width,height,radius,stroke=bool(stroke),fill=1)

    def line(self, a, b, color=INK, width=2, dash=None):
        c=self.c;c.setStrokeColor(color);c.setLineWidth(width)
        c.setDash(dash or [])
        c.line(a[0],H-a[1],b[0],H-b[1]);c.setDash([])

    def circle(self,x,top,r,color=INK,fill=True):
        self.c.setFillColor(color);self.c.setStrokeColor(color)
        self.c.circle(x,H-top,r,fill=int(fill),stroke=1)

    def arrow(self,a,b,color=TEAL,width=1.7):
        self.line(a,b,color,width)
        ang=math.atan2(b[1]-a[1],b[0]-a[0]);n=7
        for d in (-.46,.46):
            self.line(b,(b[0]-n*math.cos(ang+d),b[1]-n*math.sin(ang+d)),color,width)

    def label(self,s,x,top,width=140):
        return self.p(s,x,top,width,'small')

    def callout(self,title,body,top,fill=LIGHT):
        self.box(M,top,CW,70,fill)
        self.p(title,M+14,top+10,CW-28,'label')
        self.p(body,M+14,top+28,CW-28,'body')

    def step(self,n,title,body,top):
        self.circle(M+11,top+10,11,TEAL)
        self.text(M+11,top+13,str(n),9,colors.white,True,'center')
        self.p(title,M+31,top-1,CW-31,'heading')
        return self.p(body,M+31,top+21,CW-31)+18

    def table(self,rows,top,widths,small=False):
        style='small' if small else 'body'
        converted=[[Paragraph(str(v),STYLES['label' if i==0 else style])
                    for v in row] for i,row in enumerate(rows)]
        t=Table(converted,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),LIGHT),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),1,TEAL),
            ('LINEBELOW',(0,1),(-1,-1),.5,LINE),
            ('LEFTPADDING',(0,0),(-1,-1),9),
            ('RIGHTPADDING',(0,0),(-1,-1),9),
            ('TOPPADDING',(0,0),(-1,-1),8),
            ('BOTTOMPADDING',(0,0),(-1,-1),8),
        ]))
        _,height=t.wrap(CW,H);t.drawOn(self.c,M,H-top-height)
        return top+height

    def person(self,x,y,scale=1,arm='down',right_color=TEAL):
        # Front view: anatomical right is on the reader's left.
        s=scale
        self.circle(x,y,13*s,INK,False)
        self.line((x,y+13*s),(x,y+75*s),INK,3)
        self.line((x-28*s,y+27*s),(x+28*s,y+27*s),INK,3)
        self.line((x,y+75*s),(x-20*s,y+123*s),INK,3)
        self.line((x,y+75*s),(x+20*s,y+123*s),INK,3)
        self.line((x+28*s,y+27*s),(x+35*s,y+70*s),MUTED,3)
        self.line((x+35*s,y+70*s),(x+35*s,y+112*s),MUTED,3)
        shoulder=(x-28*s,y+27*s);elbow=(x-35*s,y+70*s)
        wrist=(x-35*s,y+112*s) if arm=='down' else (x-79*s,y+63*s)
        self.line(shoulder,elbow,right_color,5)
        self.line(elbow,wrist,right_color,5)
        for px,py in (shoulder,elbow,wrist):self.circle(px,py,4*s,right_color)
        return shoulder,elbow,wrist

    def sensor(self,x,y,label):
        self.box(x-9,y-12,18,24,TEAL,INK,2)
        self.arrow((x,y+6),(x,y-7),colors.white,1)
        self.text(x+15,y+3,label,9,INK,True)

    def camera(self,x,y,size=1):
        self.box(x-19*size,y-11*size,38*size,22*size,INK,radius=3)
        self.circle(x+10*size,y,3*size,colors.white)
        self.line((x,y+11*size),(x,y+36*size),MUTED,1)
        self.line((x,y+36*size),(x-13*size,y+47*size),MUTED,1)
        self.line((x,y+36*size),(x+13*size,y+47*size),MUTED,1)

    def save(self):
        self.c.save()


def build(output: Path):
    g=Guide(output)

    # 1: concise field overview.
    g.begin('01 / Purpose and kit','Record a usable pilot',
            'A practical camera + two-IMU capture procedure for the right arm. '
            'Start with one short rehearsal; collect the study takes only after it passes.')
    g.box(M,171,CW,126,LIGHT)
    g.text(M+17,197,'1 PILOT',17,TEAL,True)
    g.text(M+179,197,'12 USABLE TAKES',17,TEAL,True)
    g.text(M+17,220,'about 20 seconds',10,INK)
    g.text(M+179,220,'6 clear + 6 briefly occluded',10,INK)
    g.p('Six development takes first. Recalibrate, then capture six held-out '
        'test takes in a separate block. This is a small pilot design, not a '
        'statistical power calculation or an established training dataset.',M+17,242,CW-34)
    top=324
    top=g.step(1,'Prepare the capture space',
        'Phone on a fixed support, existing ESP32 rig, two selected sensors, '
        'secure straps, laptop, small object, target container and a clear table.',top)
    top=g.step(2,'Check and calibrate the right arm',
        'Verify sensor identity, allow startup bias capture, align both boards '
        'on the table, then capture mounted neutral with the arm straight down.',top)
    top=g.step(3,'Record both systems and keep the evidence',
        'Start the phone and IMU recording separately. Include visible motion '
        'pulses before and after the task. Save original video and session JSON '
        'under one take ID, including imperfect or failed takes.',top)
    g.box(M,556,CW,66,colors.HexColor('#F6F8F9'),LINE)
    g.text(M+14,577,'START THE LIVE CHECKER FROM THE REPOSITORY ROOT',8,TEAL,True)
    g.text(M+14,598,'bash arm_checker/run.sh --port auto --protocol am',9.2,INK,True)
    g.label('Then open http://127.0.0.1:8765/ . Phone recording is separate.',M+14,606,CW-28)
    g.callout('Two recordings, not automatic synchronization',
        'The checker does not record camera video, synchronize clocks or fuse '
        'camera and IMU data. Later manual alignment is a separate analysis step.',650)
    g.p('What this can support: checking feasibility, sensor identity, continuity '
        'and failure cases. What it cannot yet establish: accurate 3D hand '
        'position, grasp timing, robot control, or a benefit from sensor fusion.',M,735,CW,'small')

    # 2: camera geometry, front and top views.
    g.begin('02 / Camera placement','Keep the whole task in view',
        'Use landscape 1080p at 30 or 60 fps. Keep one frame rate for all takes. '
        'A fixed diagonal view from the wearer\'s front-right is a practical starting point.')
    g.box(M,172,CW,218,colors.HexColor('#F6F8F9'),LINE)
    g.text(M+13,192,'TOP VIEW',9,TEAL,True)
    g.box(M+145,240,202,75,colors.HexColor('#E2E8EA'),LINE)
    g.text(M+246,250,'table',10,MUTED,align='center')
    g.circle(M+190,283,8,CLAY);g.text(M+176,307,'object',8,CLAY)
    g.box(M+294,275,24,21,LIGHT,TEAL,2);g.text(M+290,307,'target',8,TEAL)
    g.circle(M+232,349,18,INK,False)
    g.line((M+254,338),(M+280,309),TEAL,4)
    g.label('wearer faces table',M+262,344,115)
    g.camera(M+440,229,.85)
    g.line((M+424,233),(M+160,270),BLUE,1,[4,3])
    g.line((M+425,239),(M+247,350),BLUE,1,[4,3])
    g.label('Phone at wearer\'s<br/>front-right',M+369,295,128)
    g.arrow((M+437,286),(M+440,271),BLUE,1)
    g.label('Start around<br/>1.5-2 m away.<br/>Adjust for complete<br/>framing; distance is<br/>not a guarantee.',M+14,252,113)
    g.box(M,411,CW,211,colors.HexColor('#F6F8F9'),LINE)
    g.text(M+13,431,'PHONE FRAME / FRONT VIEW',9,TEAL,True)
    g.person(M+262,452,.83,'bend')
    g.line((M+55,570),(M+459,570),MUTED,2)
    g.circle(M+164,560,7,CLAY);g.box(M+370,548,26,22,LIGHT,TEAL,2)
    g.label('Right shoulder, elbow and wrist<br/>stay inside the frame throughout.',M+16,454,164)
    g.arrow((M+142,489),(M+231,482),BLUE,1)
    g.label('Include object, target and the<br/>entire hand path. Leave margin.',M+298,590,190)
    g.p('<b>Before pressing Record:</b> test the full reach, avoid backlighting, '
        'keep exposure/focus stable if your phone permits, and make the hand '
        'large enough to inspect. Keep the phone position, zoom and table layout '
        'unchanged within each capture block.',M,646)
    g.callout('Raw video versus the current tracker',
        'Old visual markers are not required to save a raw pilot video. The '
        'current PhoneToPanda vision tracker does require markers; a marker-free '
        'replacement is not implemented.',709,PALE)

    # 3: sensor mounting.
    g.begin('03 / Wearable setup','Identify first, then mount',
        'Use the right upper-arm and right forearm records from the existing rig. '
        'Secure mounts are part of the measurement, not a cosmetic detail.')
    g.box(M,170,235,346,colors.HexColor('#F6F8F9'),LINE)
    g.text(M+15,192,'FRONT VIEW - ANATOMICAL RIGHT',8,TEAL,True)
    shoulder,elbow,wrist=g.person(M+139,232,1.42)
    g.sensor(M+94,300,'3')
    g.sensor(M+89,361,'4')
    g.label('Upper-arm sensor<br/>on outer segment;<br/>clear of elbow hinge',M+12,216,115)
    g.arrow((M+72,256),(M+86,292),TEAL,1)
    g.label('Forearm sensor<br/>on back/outer side;<br/>clear of wrist motion',M+12,406,115)
    g.arrow((M+61,403),(M+83,375),TEAL,1)
    g.p('<b>Default AM IDs</b><br/>3 = right upper arm<br/>4 = right forearm',M+255,174,245)
    g.p('Move one loose board at a time and watch its live orientation or gyro '
        'readout. Confirm the physical board-to-record mapping before putting '
        'it on the arm.',M+255,236,245)
    g.p('Use a repeatable location and record it. Tighten enough to prevent '
        'sliding, while remaining comfortable. Do not put a board across a '
        'joint. Leave cable slack through the full reach.',M+255,326,245)
    g.p('Keep all existing wiring and firmware unchanged. This checker selects '
        'two records from the five-record stream. No flashing or rewiring is '
        'part of this procedure.',M+255,428,245)
    g.p('<b>Measure the model lengths.</b> Use approximate shoulder-to-elbow and '
        'elbow-to-wrist joint-centre distances, in centimetres. Enter those '
        'numbers in the checker\'s cm fields (for example, 30 cm and 26 cm) '
        'and on the session sheet. They set the displayed segment lengths; '
        'the sensors do not measure them.',M,542)
    g.callout('Fixed shoulder: keep the torso still',
        'The viewer anchors the shoulder at one point. It does not measure '
        'torso translation, chest-relative shoulder angles, independent hand '
        'motion or finger closure.',635)
    g.p('The earlier rig\'s reported inward/outward reversal has not been '
        'physically diagnosed. Verify IDs and movement direction; do not assume '
        'a new viewer has fixed it.',M,725,CW,'small')

    # 4: calibration.
    g.begin('04 / Calibration','Three holds before a take',
        'Use Table alignment, then Neutral pose in the viewer. A healthy stream '
        'and successful calibration are prerequisites, not proof of physical accuracy.')
    g.box(M,174,CW,136,LIGHT)
    for i,(title,seconds,detail) in enumerate([
        ('STARTUP BIAS','10 s','Boards stationary'),
        ('TABLE ALIGN','2 s','Flat and parallel'),
        ('MOUNTED NEUTRAL','3 s','Right arm straight down')]):
        x=M+15+i*169
        g.text(x,197,title,9,TEAL,True)
        g.text(x,233,seconds,25,INK,True)
        g.p(detail,x,247,153,'small')
    g.p('The table and neutral captures also need at least 20 distinct fresh '
        'paired samples. Motion or invalid data can restart or prevent capture; '
        'wait for the actual completion state.',M+15,278,CW-30,'small')
    g.box(M,332,245,157,colors.HexColor('#F6F8F9'),LINE)
    g.text(M+12,354,'TABLE ALIGN',9,TEAL,True)
    for x in (M+65,M+151):
        g.box(x,382,35,46,TEAL,INK,3)
        g.arrow((x+17,416),(x+17,389),colors.white)
    g.label('Chip/+Z up. Both +X axes toward the same<br/>intended forward direction. Keep power on.',M+12,449,220)
    g.box(M+262,332,CW-262,157,colors.HexColor('#F6F8F9'),LINE)
    g.text(M+276,354,'MOUNTED NEUTRAL / N-POSE',9,TEAL,True)
    g.person(M+370,378,.63)
    g.label('Arm down; elbow straight;<br/>torso still; palm toward thigh.',M+276,462,221)
    top=516
    top=g.step(1,'Wait for firmware bias and fresh data',
        'Existing firmware needs ten continuous quiet seconds after startup. '
        'Check both selected IDs and valid/fresh indicators.',top)
    top=g.step(2,'Align on the table, then mount and capture neutral',
        'The table capture aligns relative heading. Mounted neutral assigns '
        'sensor mounting offsets so both arm segments point down (-Z).',top)
    g.callout('Repeat both captures when the reference changes',
        'Recalibrate after remounting, a device restart, changed sensor IDs or '
        'a changed input source. For the separate test block, recalibrate before '
        'capturing its first take.',704,PALE)

    # 5: 20 s take and independent clock diagram.
    g.begin('05 / One take','A repeatable 20-second sequence',
        'Start phone video first, then start IMU recording. Allow a little video '
        'lead-in. Stop IMU recording after the end hold, then stop the phone.')
    stages=[(0,3,'Still',LIGHT),(3,5,'Pulse 1',colors.HexColor('#CEE6E6')),
            (5,13,'Pick - move - place',colors.HexColor('#DDE8F0')),
            (13,15,'Still',LIGHT),(15,17,'Pulse 2',colors.HexColor('#CEE6E6')),
            (17,20,'Still',LIGHT)]
    x0=M+7;total=CW-14
    for start,end,label,fill in stages:
        x=x0+total*start/20;w=total*(end-start)/20
        g.box(x,198,w-2,64,fill,radius=2)
        g.p(label,x+5,213,w-10,'small')
        g.text(x,283,str(start)+' s',8,MUTED)
    g.text(x0+total,283,'20 s',8,MUTED,align='right')
    g.p('<b>Pulse:</b> with the upper arm fairly still, make one clear, comfortable '
        'elbow bend and return. Keep the forearm visible. Use the same gesture '
        'before and after the task; avoid tiny movements that are hard to identify '
        'in a 30 fps video.',M,313)
    g.p('<b>Task:</b> pick up one small object, move it along the same clear path '
        'and place it in the target. Aim for about eight seconds; do not rush to '
        'fit an exact duration. For occluded takes, hide the camera view for '
        'about two seconds during the carry only (page 7).',M,399)
    g.box(M,491,CW,128,colors.HexColor('#F6F8F9'),LINE)
    g.text(M+13,514,'SEPARATE TIMELINES',9,TEAL,True)
    for y,label in [(548,'Phone video'),(587,'IMU session')]:
        g.text(M+14,y+3,label,9,INK,True)
        g.line((M+127,y),(W-M-16,y),MUTED,1)
        for xx in [M+175,M+398]:g.line((xx,y-11),(xx,y+11),TEAL,3)
    for xx in [M+175,M+398]:g.line((xx,534),(xx,600),TEAL,1,[3,3])
    g.label('Pulse 1',M+151,601,70);g.label('Pulse 2',M+374,601,70)
    g.p('<b>Later manual alignment:</b> identify the same visible bend and its '
        'IMU angular-motion feature. Estimate an offset using the first pulse; '
        'compare the second pulse to check residual timing disagreement. Keep '
        'both original files and write down the chosen frames/times and uncertainty.',M,642)
    g.p('These gestures provide rough correspondence, not measured calibrated '
        'synchronization. The phone, ESP32 and host have independent clocks. '
        'The checker records their available timestamps separately; it does '
        'not align camera frames automatically.',M,737,CW,'small')

    # 6: capture plan.
    g.begin('06 / Capture plan','Balance conditions, reserve a block',
        'Use the same wearer, object, target and basic movement for the first pilot. '
        'Keep clear and occluded takes balanced within each block.')
    g.callout('First: one 20-second pilot, labelled P00',
        'Inspect video framing, both IDs, calibration, session playback and '
        'both visible pulses. If anything fails, correct it and repeat the '
        'pilot before starting development takes.',170)
    rows=[['Order','Take IDs','Conditions','Use'],
          ['1','D01 / D02','Clear / Occluded','Development'],
          ['2','D03 / D04','Occluded / Clear','Development'],
          ['3','D05 / D06','Clear / Occluded','Development'],
          ['Break','Recalibrate','New capture block','Freeze analysis choices'],
          ['4','T01 / T02','Occluded / Clear','Held-out test'],
          ['5','T03 / T04','Clear / Occluded','Held-out test'],
          ['6','T05 / T06','Occluded / Clear','Held-out test']]
    end=g.table(rows,266,[48,110,186,CW-344],True)
    g.p('<b>Target: 12 usable paired takes.</b> Six development takes (3 clear, '
        '3 occluded), then six test takes (3 clear, 3 occluded). Conditions are '
        'specified before capture. A take is a paired video and IMU session, '
        'not a single frame.',M,end+24)
    g.p('<b>Keep the test block separate.</b> Recalibrate before it and record '
        'that block identity. Choose processing, alignment rules and usability '
        'criteria using development data. Do not tune the method against test '
        'outcomes and still call those outcomes unseen.',M,end+101)
    g.callout('Retain failures and replacements',
        'Keep every attempt with a rejection reason. A corrupt or missing '
        'recording may be replaced under the same condition; append an attempt '
        'suffix such as D02_a02. Do not discard a valid take because the method fails.',701,PALE)

    # 7: occlusion.
    g.begin('07 / Occlusion condition','Hide the view, keep the path clear',
        'Create a short, repeatable camera occlusion during carry. Keep start/end '
        'poses, timing pulses, pickup and placement visible.')
    g.box(M,173,CW,280,colors.HexColor('#F6F8F9'),LINE)
    g.text(M+14,196,'TOP VIEW / ONE CAMERA',9,TEAL,True)
    g.camera(M+48,380,.85)
    g.box(M+225,219,239,164,colors.HexColor('#E2E8EA'),LINE)
    g.circle(M+255,302,8,CLAY)
    g.box(M+421,290,26,24,LIGHT,TEAL,2)
    g.arrow((M+269,302),(M+409,302),TEAL,3)
    g.text(M+313,285,'hand + object path',9,TEAL,True)
    g.line((M+60,365),(M+350,302),BLUE,1,[4,3])
    g.line((M+167,336),(M+184,375),CLAY,6)
    g.label('Occluder between camera and path;<br/>not in the wearer\'s reach space',M+67,217,190)
    g.arrow((M+164,250),(M+174,326),CLAY,1)
    g.label('About 2 seconds of hidden carry.<br/>A helper may move a card; do not<br/>move the camera or touch the wearer.',M+226,401,238)
    g.p('Use a card or screen outside the hand path, or ask a helper to cover '
        'that line of sight briefly. Do not ask the wearer to reach behind an '
        'obstacle. Keep cables slack and the object securely supported. Use '
        'the same planned timing across occluded takes.',M,480)
    g.p('Check that the intended arm/hand area actually disappears from the '
        'camera for the chosen interval. Write down its approximate start and '
        'end. Leave the timing pulses and task endpoints unobstructed.',M,569)
    g.callout('Occlusion is not hidden ground truth',
        'One camera cannot reveal hidden motion. For later error measurement, '
        'use an optional second, unoccluded reference camera and verify its '
        'timing and geometry.',649,PALE)
    g.p('Continuity during occlusion can be inspected qualitatively. Numerical '
        'accuracy claims require an independent reference and a defined evaluation '
        'method; the animated arm is not that reference.',M,736,CW,'small')

    # 8: file management and record sheet.
    g.begin('08 / Files and record sheet','Make each pair traceable',
        'Keep original phone files and exported session JSON together under a '
        'single take ID. Do not overwrite originals after trimming or alignment.')
    g.box(M,173,CW,102,LIGHT)
    g.text(M+15,195,'EXAMPLE DIRECTORY',9,TEAL,True)
    g.text(M+15,218,'capture_20261011 / development / D01_a01 /',10,INK,True)
    g.text(M+27,240,'video_original.mp4    imu_session.json    notes.txt',10,INK)
    g.p('Use test / T01_a01 / for the reserved block. Preserve the original '
        'phone filename in notes, even if you also create a descriptive copy.',M+15,252,CW-30,'small')
    rows=[['Record for each take','Value / space for notes'],
          ['Take ID / block / condition','______________________________________'],
          ['Original video + session ID','______________________________________'],
          ['Date / wearer / mounting notes','______________________________________'],
          ['IDs / upper and forearm lengths (cm)','______________________________________'],
          ['Camera fps / frame size / position','______________________________________'],
          ['Calibration completed / issues','______________________________________'],
          ['Pulse 1 / pulse 2 visible?','______________________________________'],
          ['Occlusion interval, if used','______________________________________'],
          ['Drops / stale flags / task outcome','______________________________________'],
          ['Usable? / reason / replacement ID','______________________________________']]
    end=g.table(rows,295,[216,CW-216],True)
    g.p('<b>Before moving on:</b> replay or inspect the saved IMU session, open '
        'the original phone clip, confirm both gestures and the full task are '
        'present, and check that neither selected sensor became invalid. Save '
        'notes while the take is still easy to remember.',M,end+20)
    g.p('Do not pair an old video with a newly recorded IMU movement. A valid '
        'pair comes from the same physical event. Exported host and ESP32 '
        'timestamps retain their original clock domains.',M,745,CW,'small')

    # 9: failure checks, scope and exact reviewer start.
    g.begin('09 / Review before analysis','Pass the checks, keep the limits',
        'Use this page after the rehearsal and after each capture block. '
        'A plausible-looking skeleton alone is not sufficient evidence.')
    rows=[['If you see this','Check / record this'],
          ['No stream or wrong device','Close other serial readers; confirm the selected port and protocol. Do not reflash as a first step.'],
          ['Wrong limb direction','Verify each physical sensor ID separately, then repeat table and mounted calibration. Preserve a mismatch recording.'],
          ['Capture will not complete','Hold still; check fresh paired samples, sensor validity, gyro motion and mount stability.'],
          ['Pose freezes or jumps','Inspect stale/invalid states, device restarts and cable movement. Record the fault; do not label held geometry as fresh motion.'],
          ['Video or pulse is missing','Keep the attempt and note the reason; repeat the same planned condition with a new attempt suffix.'],
          ['Drift during a take','Keep the evidence. Inspect neutral return and pulse disagreement; do not silently correct the test result.']]
    end=g.table(rows,172,[147,CW-147],True)
    g.p('<b>Measurement boundary.</b> Two six-axis IMUs do not observe absolute '
        'heading, global translation, finger closure or independent wrist/hand '
        'pose. The fixed-shoulder geometry depends on calibration and measured '
        'segment lengths. There is no Panda inverse-kinematics or robot-control '
        'connection in this checker.',M,end+22)
    g.p('<b>What is still unverified.</b> Mounted direction correctness, '
        'physical angle accuracy, stationary drift and end-to-end latency '
        'need measured tests. The earlier inward/outward symptom remains '
        'unresolved. This guide proposes data collection; it reports no '
        'new physical validation or camera-IMU fusion result.',M,end+115)
    g.box(M,682,CW,90,LIGHT)
    g.text(M+14,704,'REVIEWER START - FROM REPOSITORY ROOT',9,TEAL,True)
    g.text(M+14,727,'bash arm_checker/run.sh --demo',11,INK,True)
    g.p('Open http://127.0.0.1:8765/ . See arm_checker/README.md and '
        'hardware_setup.md for live serial input, calibration and the adapter '
        'contract. Demo input is synthetic.',M+14,740,CW-28,'small')
    g.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,
                        default=Path(__file__).with_name('recording_guide.pdf'))
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    build(args.output)
    print(args.output)


if __name__=='__main__':
    main()
