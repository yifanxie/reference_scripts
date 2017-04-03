import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PixelbyPixelClassifier as PP
from operator import itemgetter, attrgetter
import gc
import pdb
import objgraph
import sys

## Key refrence that is being looked for
keyref="JC5A"

##keyref="4525"
##
##inputfolder="./SamplingDrawings/"
##imgname="A579-42504_8.png"


##inputfolder="./Drawing_PNG/JC5A_Drawing/"
inputfolder="./Drawing_PNG/"
outputfolder="./OCR_Output/"
##imgname="JA579-42505.png"
imgname="A572-40069.png"
##imgname="JA579-42503.png"


folderpath=inputfolder
maxh=200
maxw=200
minh=20

mx=0
my=0
click=False




imgfdpath="./74KTrain/"
datafile=imgfdpath+"PbyP_Training.data"
modelpath="./StatModel/"

##outputpath="./OCR_Output/"


method=2


##matchlist=np.array(['A4','I1', 'O0', 'Z2', 'S5'])
matchlist=np.array(['I1', 'O0', 'Z2', 'S5'])

loadmodel=False

##PbyP feature extraction setting
case_sensitive_setting=True
fs="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
keeprest=False
chrmatch=False

############################################################################################################################################
## Save the SVM model
def SaveModel(model, modelpath):
    modelfile="svm_Global.xml"
    svm_modelfile=modelpath+modelfile
    print "saving SVM model"
    model.save(svm_modelfile)

############################################################################################################################################
## Loading SVM model for machine
def LoadModel(modelpath):
    modelfile="svm_unc.xml"
    svm_modelfile=modelpath+modelfile
    model=cv2.SVM()
    model.load(svm_modelfile)
    return model

############################################################################################################################################
## Display a given bounding rectangle for a given contour
def DisplayImgRect(img, rect, scale=1):
    roi=img[rect[0][0]:rect[0][1]+1, rect[0][1]:rect[1][1]+1]
    roi=cv2.resize(roi,None,fx=scale,fy=scale,interpolation = cv2.INTER_CUBIC)
    while (1):
        cv2.imshow('roi',roi)
##        cv2.setMouseCallback("in", CaptureMouse,img)
        k=cv2.waitKey(33)
        if k==27:    # Esc key to stop
            cv2.destroyAllWindows()
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print k # else print its value


############################################################################################################################################
## Display an image
def DisplayImg(img, scale=1):
    img=cv2.resize(img,None,fx=scale,fy=scale,interpolation = cv2.INTER_CUBIC)
    while (1):
        cv2.imshow('roi',img)
##        cv2.setMouseCallback("in", CaptureMouse,img)
        k=cv2.waitKey(33)
        if k==27:    # Esc key to stop
            cv2.destroyAllWindows()
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print k # else print its value

############################################################################################################################################
## Take a given ndarray representation of an image as input
## Output the grayscale image in ndarrary representation
def GetROI(img):
    startRow=row
    startCol=col
    endRow=row+height
    endCol=col+width
    if startRow<0:
        startRow=0
    if startCol<0:
        startCol=0
    if endRow>img.shape[0]-1:
        endRow=img.shape[0]-1
    if endCol>img.shape[1]-1:
        endCol=img.shape[1]-1
    gray = cv2.cvtColor(orgimg,cv2.COLOR_BGR2GRAY)
    roi=gray[startRow:endRow, startCol:endCol]
    return roi

############################################################################################################################################
## Plot the histogram of a given ndarray by value
def PlotHist(data):
##    hist=data
    valuerange=np.zeros(data.max()-data.min()+1)
    for count in xrange(0,valuerange.size):
        value=data.min()+count
        valuerange[count]=len(data[data==value])

    plt.plot(valuerange[0:1000])
    plt.show()

############################################################################################################################################
## Gather information about the contours in an image, this include the following:
## area, height, width, bounding rectangle informaton, and machine learning guesses
def ContourAreaDetails(contours, hierarchy):
    area_range=[0,0]
    areas=[]
    heights=[]
    widths=[]
    rects=[]
    guesses=[]

    for count in xrange(0,len(contours)):
        if hierarchy[0,count,3]==-1:
            cnt=contours[count]
            [x,y,w,h] = cv2.boundingRect(cnt)
            if h<maxh and w<maxw:
                heights.append(h)
                widths.append(w)
                areas.append(cv2.contourArea(cnt))
                rects.append([[y,x],[y+h, x+w]])
                guesses.append([])

    heights=np.array(heights)
    widths=np.array(widths)
    areas=np.array(areas)
    rects=np.array(rects)
##    PlotHist(areas)
    #print areas.mean()

##    return areas, heights, widths, rects
    return areas, heights, widths, rects, guesses

############################################################################################################################################
## Capture the mouse location on the displayed image, every time a left-click event happen
def CaptureMouse(event, x, y, flags, param):
##    print param
    global click, mx, my
    if event==cv2.EVENT_LBUTTONDOWN:
##        print x, y
        mx=x
        my=y
        click=True


############################################################################################################################################
## Perform OCR learning of a given image "roi", given a machine learning model
def Learning(roi, model):
    grayroi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
##    grayroi=roi
    preproc_roi=PP.PreProc(grayroi, 0.04)

    ds_roi=PP.GetBoxImg(preproc_roi)

    if ds_roi.size>0:
        indicators=np.float32(PP.PbyP_FeatureExtraction(ds_roi,20))
        indicators=indicators.reshape((1,indicators.shape[0]))
        retval=model.predict_all(indicators)
        retval=retval.ravel()
    else:
        print "empty space detected"
        retval=32
    return retval



############################################################################################################################################
## Identify the bounding rectangle which a left-clicked location fall in
def FindBoundingCnt(contours, rects):
    global mx,my
    bindingcnts=[]
    for count in xrange(0,len(rects)):
        if my>=rects[count][0][0] and my<=rects[count][1][0] and\
            mx>=rects[count][0][1] and mx<=rects[count][1][1]:
##                print "in"
            bindingcnts.append(count)

##    print "check finish"
    return bindingcnts




############################################################################################################################################
## Dither/move the ROI a little bit in the overall image in location and size
## A machine learning guess is then perform for each dithering, as well as the
## ROI itself
def DitheringROI(img, current_rect, model, plot=False):
    #These all need changing to increase index by 1 when subseting images
    x1=current_rect[0][1]
    y1=current_rect[0][0]
    x2=current_rect[1][1]
    y2=current_rect[1][0]

    rows=img.shape[0]
    cols=img.shape[1]

    retvals=[]
    dither_rects=[]
    dither_rois=[]

    org_roi=img[y1:y2+1,x1:x2+1]
    result=Learning(org_roi,model)
    retvals.append(chr(int(result)))

    h=y2-y1
    w=x2-x1
    factor=0.3
    w_inc=int(round(float(w)*factor/2))
    h_inc=int(round(float(h)*factor/2))

##    roi=org_img[y1:y2, x1:x2]
    zoomcount=5


    for zoomloop in xrange(1,3):

        dx1=x1-zoomloop*w_inc
        dy1=y1-zoomloop*h_inc
        dx2=x2+1+zoomloop*w_inc
        dy2=y2+1+zoomloop*h_inc

        if dx1<0: dx1=0
        if dy1<0: dy1=0
        if dx2>=cols: dx2=cols-1
        if dy2>=rows: dy2=rows-1
##        print (dy1,dx1), (dy2, dx2)

        dither_rects.append([[dy1,dx1],[dy2, dx2]])
        dither_rois.append(img[dy1:dy2, dx1:dx2])
    ##        print [dy1,dx1],[dy2, dx2]
    ##        result=Learning(dither_rois[zoomloop-1],model)
    ##        print "zoomloop is %d" %zoomloop
    ##        if zoomloop==2:
    ##            print "debug"
        if dx1<dx2 and dy1<dy2:
##            result=Learning(dither_rois[zoomloop+2],model)
            result=Learning(dither_rois[zoomloop-1],model)
            retvals.append(chr(int(result)))


##    print "List of guesses are %s" %retvals

    if plot:
        plt.subplot(181), plt.imshow(org_roi, 'gray')
        plt.subplot(182), plt.imshow(dither_rois[0],'gray')
        plt.subplot(183), plt.imshow(dither_rois[1],'gray')
##        plt.subplot(184), plt.imshow(dither_rois[2],'gray')
##        plt.subplot(185), plt.imshow(dither_rois[3],'gray')
##        plt.subplot(186), plt.imshow(dither_rois[4],'gray')
##        plt.subplot(187), plt.imshow(dither_rois[5],'gray')
##        plt.subplot(188), plt.imshow(dither_rois[6],'gray')

        plt.show()
##    print retvals

##    retvals=np.array(retvals)
    return retvals




############################################################################################################################################
## the following routine pick out the contour bounding rectangles that rested within the
## perceived "long windows" containing the strings being searched for i.e. "JC5A"
def RectsInLongWindow(longrect, rects):
    xl1=longrect[0][1]
    yl1=longrect[0][0]
    xl2=longrect[1][1]
    yl2=longrect[1][0]
    rects_lw=[]
    rects_lw_ids=[]
    cly=(yl2+yl1)/2
    leftpos=[]
    sort_rects=[]

##   failed attempt to vectorized the for loop :(
##    x1s=np.array([rect[0][1] for rect in rects])
##    y1s=np.array([rect[0][0] for rect in rects])
##    x2s=np.array([rect[1][1] for rect in rects])
##    y2s=np.array([rect[1][0] for rect in rects])
####    inrectsid=np.argwhere(rects[np.logical_and(y1s<cly,y2s>cly)])
##
##    condor1=np.logical_and(x1s>=xl1, x2s<=xl2)
##    condor2=np.logical_and(x1s<xl1, x2s>xl1, x2s<xl2)
##    condor3=np.logical_and(x1s>xl1, x1s<xl2, x2s>xl2)
##
##    condand1=np.logical_and(y1s<cly,y2s>cly)
##    condand2=np.logical_or(condor1, condor2, condor3)
##
##
##    inrectsid=np.argwhere(np.logical_and(condand1, condand2))
##    sort_rects=zip(x1s[inrectsid], rects[inrectsid], inrectsid)


    for count in xrange(0, len(rects)):
        x1=rects[count][0][1]
        y1=rects[count][0][0]
        x2=rects[count][1][1]
        y2=rects[count][1][0]

        #here need to filter the rects size
        if y1<cly and y2>cly:
            if (x1>=xl1 and x2<=xl2) or \
                 (x1<xl1 and x2>xl1 and x2<xl2) or \
                 (x1>xl1 and x1<xl2 and x2>xl2):

##                if leftmost==0 or y1<leftmost:
                sort_rects.append((x1, rects[count], count))


##                rects_lw.append(rects[count])
##                rects_lw_ids.append(count)

    sort_rects=sorted(sort_rects, key=itemgetter(0))
##    sort_rects=np.array(sort_rects)
    rects_lw=[row[1] for row in sort_rects]
    rects_lw_ids=[row[2] for row in sort_rects]

    return rects_lw, rects_lw_ids





############################################################################################################################################
## Evaluate the likelihood of a character in the keystr is among the
## machine learning guesses of the given rectangle
def CheckRef(keystr, rects_lw_ids, guesses):
    # check ref need to be rewritten to allow more flexibility in recognising key string
    potential_detected=0.0
    ref_detected=False
    for count in xrange(0, len(keystr)):
        if count==len(rects_lw_ids): break
        letter=keystr[count]


        if guesses[rects_lw_ids[count]]!=[]:
            currentguess=np.array(guesses[rects_lw_ids[count]])
            letter_ratio=float(len(currentguess[currentguess==letter]))/float(len(currentguess))
            if letter_ratio>0.5:
                potential_detected+=1
            if letter_ratio<0.35:
                potential_detected-=1

    lendif=len(rects_lw_ids)-len(keystr)
    if lendif>=2 or lendif<0 :
        potential_detected-=2


##  this is another point to detect the ref key string
    if potential_detected>=0:
        ref_detected=True
    return ref_detected





############################################################################################################################################
## Create a long window which include the current rectangle the routine is processing
## if there is a keystr letter among the machine learning guesses for this rectangle
##
## The size and placement of the window is depend on the estimated location of
## the letter's order in the keystr

def GetLongWindow(currentrectid, rects, heights, guesses):
    keystr=np.array(list(keyref))
    keychar_found=False
    x1=rects[currentrectid][0][1]
    y1=rects[currentrectid][0][0]
    x2=rects[currentrectid][1][1]
    y2=rects[currentrectid][1][0]
    currentrect_h=y2-y1
    currentrect_w=x2-x1
    currentguess=np.array(guesses[currentrectid])
    longrect=[]
    rects_lw=[]
    rects_lw_ids=[]
    ref_detected=False
    p1s=[]
    p2s=[]

##    heightLB, heightUB=np.percentile(heights, (20,100))
    offset=0
    if (currentrect_h>=minh and currentrect_h<=maxh):
        for count in xrange(0, len(keystr)):
            letter=keystr[count]
##  This part need to be rewritten to allow more flexible recognition
            letter_ratio=float(len(currentguess[currentguess==letter]))/float(len(currentguess))
            if letter_ratio>=0.4:
                offset=count
                keychar_found=True
                break

        if keychar_found:
##            print "offset is %d" %offset
            longrect_w=int(6*currentrect_w)
            xl1=int(x1-offset*1.4*currentrect_w)
            yl1=y1
            xl2=xl1+longrect_w
            if xl2<x2: xl2=x2
            yl2=y2
            longrect=[[yl1,xl1],[yl2,xl2]]
            rects_lw, rects_lw_ids=RectsInLongWindow(longrect, rects)

##          the following code correct the dimension of the long windows based on the rects identified to be in the long window
            for rectloop in xrange(0, len(rects_lw)):
                p1s.append(rects_lw[rectloop][0].tolist())
                p2s.append(rects_lw[rectloop][1].tolist())

            leftps=sorted(p1s, key=itemgetter(1))
            topps=sorted(p1s, key=itemgetter(0))
            rightps=sorted(p2s, key=itemgetter(1), reverse=True)
            bottomps=sorted(p2s, key=itemgetter(0), reverse=True)

            xl=leftps[0][1]
            yl=topps[0][0]
            xr=rightps[0][1]
            yr=bottomps[0][0]

##
##            if currentrectid==98:
##                print "debug"
##                 pdb.set_trace()

            longrect=[(yl, xl),(yr, xr)]
            lw_w=float(xr-xl)
            lw_h=float(yr-yl)
            lw_wh_ratio=lw_w/lw_h
            if lw_h>maxh or lw_wh_ratio>7 or lw_wh_ratio<2.5 or len(rects_lw)<2 :
                ref_detected=False
                longrect=[]
##            elif
##                ref_detected=False
##                longrect=[]
            else:
                if xl>xl1: xl=xl1
                if xr<xl2: xr=xl2
                ref_detected=CheckRef(keystr,rects_lw_ids, guesses)
##                ref_detected=True

    return longrect, rects_lw, rects_lw_ids, ref_detected





############################################################################################################################################
## Perform global learning - learning on each single rectangle
## For each rectangle on the image, produce a list of machine learning guesses
def GlobalLearning(img, contours, heights, widths, rects, guesses, model):
    learningcount=0
##    ind=np.argwhere(np.logical_and(heights>minh, heights<maxh))
    for count in xrange(0,len(rects)):

        h=heights[count]
        w=widths[count]

        if (h>=minh and h<=maxh):

            learningcount+=1
##            print "learning countour %d" %learningcount
            print "%s: learning countour %d" %(imgname, learningcount)

##            if count==3:
##                print "debug"
            current_rects=rects[count]
            guesses[count]=DitheringROI(img, current_rects, model)
    return guesses




############################################################################################################################################
## Try to recognise the key reference string by applying machine learning model
## if the key refrence string is identified, put displayed information on the image

def PatternRecognition(img, rects, heights, guesses, model):
    pattern_found=False
    pattern_list=[]
    detectcount=0
    ref_longwins=[]
    # need to create a routine call display rect!
    # need to capture all the rectangle that suspect to have the reference
    # potentially create a routien that can save the suspect images in a folder.


##    heightLB, heightUB=np.percentile(heights, (20,100))
    recg_count=0
    for count in xrange(0,len(rects)):
        h=heights[count]


        if (h>=minh and h<=maxh) and guesses[count]!=[]:
##            print "hello world"
            recg_count+=1
##            print "recognizing countour %d" %recg_count
            print "%s: recognizing countour %d" %(imgname, recg_count)

            if count==12:
                print "debug"


            longrect, rects_lw,rects_lw_ids,ref_detected=GetLongWindow(count, rects, heights,guesses)
            if longrect!=[]:
##                print "long win exisits"
                x1=longrect[0][1]
                y1=longrect[0][0]
                x2=longrect[1][1]
                y2=longrect[1][0]
##                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

                if ref_detected and (rects_lw_ids not in ref_longwins):
                    ref_longwins.append(rects_lw_ids)
                    pattern_found=True

                for lwcount in xrange(0, len(rects_lw)):
                    x1=rects_lw[lwcount][0][1]
                    y1=rects_lw[lwcount][0][0]
                    x2=rects_lw[lwcount][1][1]
                    y2=rects_lw[lwcount][1][0]
                    cy=int((y1+y2)/2)
                    cx=int((x1+x2)/2)
                    string=str(rects_lw_ids[lwcount])
                    if ref_detected:
                        detectcount+=1
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(img,string,(x1, y1+10),0,0.5,(255,0,0),1)
##                        pattern_found=True

##                    else:
##                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
    return img, ref_longwins




############################################################################################################################################
## Display a GUI for the image that is being studied.
## This only reliably works on small-medium sized image

def ImgGUI(img, orgimg, contours, rects, heights, guesses):
    global click, mx, my
    based_img=img.copy()
    learned=False
    while (1):
        cv2.imshow('in',img)
        cv2.setMouseCallback("in", CaptureMouse,img)
        k=cv2.waitKey(33)
        if k==27:    # Esc key to stop
            cv2.destroyAllWindows()
##            top.destroy()
            break
        elif click:
##            print mx, my
##            cv2.destroyWindow('roi')
            bindingcnts=FindBoundingCnt(contours, rects)

            if len(bindingcnts)>0:
                img=based_img.copy()
                for count in xrange(0, len(bindingcnts)):
                    cntid=bindingcnts[count]
                    x1=rects[cntid][0][1]
                    y1=rects[cntid][0][0]
                    x2=rects[cntid][1][1]
                    y2=rects[cntid][1][0]

                    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)

                    print "the clicked area rect id is %d" %cntid
                    print "List of guesses are %s" %guesses[cntid]
                    if guesses[cntid]!=[]:
                        learned=True
                    else:
                        learned=False

                current_rects=rects[bindingcnts[0]]
            click=False

        elif k==108:

            ##call GetLongWindow(img, currentrectid, rects, heights):
            print "show long scanning windows"
##            print learned
            if learned:
##                print cntid
                img=based_img.copy()
##                guesses=np.array(guesses)
                longrect, rects_lw, rects_lw_ids, ref_detected=GetLongWindow(cntid, rects, heights,guesses)
                if longrect!=[]:
                    x1=longrect[0][1]
                    y1=longrect[0][0]
                    x2=longrect[1][1]
                    y2=longrect[1][0]

                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    for count in xrange(0, len(rects_lw)):
                        x1=rects_lw[count][0][1]
                        y1=rects_lw[count][0][0]
                        x2=rects_lw[count][1][1]
                        y2=rects_lw[count][1][0]
                        if ref_detected:
                            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                        else:
                            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

        elif k==112:
            print "display dithering windows"
            if learned:
                dither_img=based_img.copy()
                print current_rects
                result=DitheringROI(dither_img, current_rects, model, True)

        elif k==115:
            if longrect!=[]:
                print "saving image for testing purpose"
                testpath="./"
                testimgpath=testpath+"slidingwindow_testimg.png"
                x1=longrect[0][1]
                y1=longrect[0][0]
                x2=longrect[1][1]
                y2=longrect[1][0]
                testimg=orgimg[y1:y2+1, x1:x2+1]
                cv2.imwrite(testimgpath, testimg)

        elif k==107:
            SaveModel(model,modelpath)


        elif k==-1:  # normally -1 returned,so don't print it
            continue


        else:
            print k # else print its value



############################################################################################################################################
## mapping individual character srcchr into dstchr in matchlist
def Mapchar(srcchr):
    dstchr=srcchr
    for matchloop in xrange(0,len(matchlist)):
        if srcchr in matchlist[matchloop]:
            dstchr=matchlist[matchloop][0]
    return dstchr



############################################################################################################################################
## replace the characters in the guesses array with characters in matchlist
def CharacterMatching(guesses, keyref):
##    if len(matchlist)>0:
    for guessloop in xrange(0,len(guesses)):
        for itemloop in xrange(0, len(guesses[guessloop])):
            srcchr=guesses[guessloop][itemloop]
            guesses[guessloop][itemloop]=Mapchar(srcchr)
    keyreflist=list(keyref)
    for refloop in xrange(0,len(keyreflist)):
        keyreflist[refloop]=Mapchar(keyreflist[refloop])

    keyref="".join(keyreflist)
    return guesses, keyref





############################################################################################################################################
## Draw the long window that contain the current left-clicked rectable in the image
def DrawLongWin(rects, ref_longwins, img):
    ref_rects=[]
    drawcount=0
    ref_longwins_ps=[]
    for lwloop in xrange(0, len(ref_longwins)):
        p1s=[]
        p2s=[]

        ref_rects=rects[ref_longwins[lwloop]]
        for rrloop in xrange(0, len(ref_rects)):
            p1s.append(ref_rects[rrloop][0].tolist())
            p2s.append(ref_rects[rrloop][1].tolist())

        leftps=sorted(p1s, key=itemgetter(1))
        topps=sorted(p1s, key=itemgetter(0))
        rightps=sorted(p2s, key=itemgetter(1), reverse=True)
        bottomps=sorted(p2s, key=itemgetter(0), reverse=True)


        xl=leftps[0][1]
        yl=topps[0][0]
        xr=rightps[0][1]
        yr=bottomps[0][0]

        ref_longwins_ps.append(((yl, xl), (yr,xr)))

        print"printing coordinate..."
        print (yl, xl)
        print (yr, xr)

        cv2.rectangle(img,(xl,yl),(xr,yr),(0,0,255),2)
        drawcount+=1
##    print drawcount
    return img, ref_longwins_ps




############################################################################################################################################
## Output potential OCRed result with key refrence as individual images
def OutputResults(img, orgimg, ref_longwins_ps):
##    imgoutfd=outputpath+
    imgfile, ext=os.path.splitext(imgname)
    img_output_folder=outputfolder+imgfile
    output_count=0
    if len(ref_longwins_ps)>0:
        print "Saving each suspected area into the output folder"
        if not os.path.exists(img_output_folder):
            os.makedirs(img_output_folder)
        else:
            ## clean up the output folder if it already exists
            filelist=os.listdir(img_output_folder)
            for f in filelist:
                fpath=img_output_folder+"/"+f
                os.remove(fpath)

        fullimg_output=img_output_folder+"/OCR_"+imgname
        cv2.imwrite(fullimg_output, img)
        for lwloop in xrange(0,len(ref_longwins_ps)):
            outputimg=[]
            imgoutputname=""
            output_count+=1
            img_output_file=img_output_folder+"/"+imgfile+"_"+str(output_count)+ext

            x1=ref_longwins_ps[lwloop][0][1]
            y1=ref_longwins_ps[lwloop][0][0]
            x2=ref_longwins_ps[lwloop][1][1]
            y2=ref_longwins_ps[lwloop][1][0]

            outputimg=orgimg[y1:y2+1, x1:x2+1]
            cv2.imwrite(img_output_file, outputimg)
    else:
        print "there is no output detected"





############################################################################################################################################
## Perform full image OCR
def ImgOCR(orgimg, model, outputfolder):

    global keyref
    ##  threshold operation for the input image
    gray = cv2.cvtColor(orgimg,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    del gray
##  Get a list contours based on simple heuristic that potentially contains characters
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    del thresh
##  Get the overall information regarding each of the bounding rectangle for the contours
    areas, heights, widths, rects, guesses=ContourAreaDetails(contours,hierarchy)

    img=orgimg.copy()
##  Establish guesses for each rectangle by applying SVM algorithm
    guesses=GlobalLearning(img, contours, heights, widths, rects, guesses, model)

##  Map characters based on the character match list
    print "performing characters mapping......"
    guesses, keyref=CharacterMatching(guesses,keyref)


##  Draw a rectangle that bound each area with suspected detection of the key reference
    img, ref_longwins=PatternRecognition(img, rects, heights, guesses, model)
    print "printing bounding rectangle for suspected area with reference"
    img,ref_longwins_ps=DrawLongWin(rects, ref_longwins, img)
    OutputResults(img, orgimg, ref_longwins_ps)
    print "Finished processing!"
    del img
    del orgimg




############################################################################################################################################
## Perform full image OCR for each image in a given folder
def GlobalOCR(inputfolder, outputfolder, model):
##    print "hello world!"
    global imgname
    errlist=[]
    imglist=[]
    if os.path.exists(inputfolder):
        filelist=os.listdir(inputfolder)
        for f in filelist:
##            imgname,ext==os.path.splitext(imgname)
            imgname=f
            imgpath=inputfolder+imgname
            print "input image is %s" %imgname
##            del orgimg
##            collected = gc.collect()
##            print "Garbage collector: collected %d objects." %(collected)
##            pdb.set_trace()
            try:
                orgimg = cv2.imread(imgpath)
                gc.collect()
                if not(orgimg==None):
                    ImgOCR(orgimg,model, outputfolder)
                    print "dumping image %s" %imgname
                del orgimg

            except IOError:
                print "cannot open image %s" %imgname
                errlist.append()
            except:
                print "unspecified errro with image %s" %imgname
                errlist.append()

    else:
        print"input folder doesn't exist"

    errlist=np.array(errlist)
    return errlist




############################################################################################################################################
## Set up the SVM model, method 1 (For KNN model) here is no longer in use.
def SetupSVMModel():
    ##  loading data, set up the feature vectors and labels, and train the SVM model
    data=PP.LoadIndicators(datafile, fs, keeprest, case_sensitive_setting, chrmatch)
    responses=data[:,0]
    samples=data[:,1:]
    if loadmodel:
        model=LoadModel(modelpath)
    else:
        if method==1:
            model = cv2.KNearest()
            model.train(samples,responses)
            GlobalOCR(inputpath, outputpath, model)

        elif method==2:
            params= dict( kernel_type = cv2.SVM_POLY,
            svm_type = cv2.SVM_C_SVC,
            C = 100, gamma=5, degree=3, coef0=10)
            model = cv2.SVM()
            model.train(samples, responses, params=params)
    return model





############################################################################################################################################
if __name__ == '__main__':
    loadmodel=False
    chrmatch=False
    samplerun=False
    start_time=time.time()
##    pdb.set_trace()
    model=SetupSVMModel()

##  check if this is a test run with one image, or a non-test run which parse images in a folder
    test=True

    if not test:
        if samplerun:
            inputfolder="./SamplingDrawings/"
            outputfolder="./OCR_SampleOutput/"
            errorlist=GlobalOCR(inputfolder, outputfolder, model)
        else:
            inputfolder="./Drawing_PNG/"
            outputfolder="./OCR_Output/"
            errorlist=GlobalOCR(inputfolder, outputfolder, model)

    else:

        testimgpath=folderpath+imgname
        testorgimg = cv2.imread(testimgpath)

    ##  threshold operation for the input image
        gray = cv2.cvtColor(testorgimg,cv2.COLOR_BGR2GRAY)


        thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
##        del gray

    ##  Get a list contours based on simple heuristic that potentially contains characters
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    ##  Release the memory occupy by gray and thresh, this is very import for big images
##        del thresh

    ##  Get the overall information regarding each of the bounding rectangle for the contours
        areas, heights, widths, rects, guesses=ContourAreaDetails(contours,hierarchy)


        testimg=testorgimg.copy()
    ##  Establish guesses for each rectangle by applying SVM algorithm
        guesses=GlobalLearning(testimg, contours, heights, widths, rects, guesses, model)

    ##  Map characters based on the character match list
        print "performing characters mapping......"
        guesses, keyref=CharacterMatching(guesses,keyref)

    ##  Draw a rectangle that bound each area with suspected detection of the key reference
        img, ref_longwins=PatternRecognition(testimg, rects, heights, guesses, model)
        print "printing bounding rectangle for suspected area with reference"
        img,ref_longwins_ps=DrawLongWin(rects, ref_longwins, testimg)


##        cv2.imwrite("GlobalOCR.png", img)
        OutputResults(img, testorgimg,ref_longwins_ps)
##        ImgGUI(testimg, testorgimg, contours, rects, heights, guesses)
        print "Finished processing!"

        a=gc.collect()
    end_time=time.time()
    duration=end_time-start_time


    print "it takes %.3f seconds to study the image"  %(duration)
    print "the study results are saved in the output folder"
##    ImgGUI(img,contours,rects,heights, guesses)






