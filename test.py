__author__ = 'yxie1'

__author__ = 'yxie1'

from bs4 import BeautifulSoup
import re
import os
import numpy as np
import pandas as pd
import csv

def OpenFile(filename):
    os.startfile(filename)


class HTMLfile(object):
    text=[]
    error=[]
    refs=[]

    def __init__(self, filepath):
        if os.path.isfile(filepath):
            self.name=os.path.basename(filepath)
            self.folder=os.path.basename(os.path.dirname(filepath))
            self.path=filepath
            self.get_text()
        else:
            self.error="Read Error"
            self.pdf_type="Unreadable"

    def get_text(self):
        # print "performing extraction for %s" %self.path
        try:
            self.text=self.convert(self.path)
            self.text=re.sub('[^A-Za-z0-9\-]+', ' ', self.text)

            if len(self.text)>500:
                self.pdf_type="Readable"
            else:
                self.pdf_type="ToCheck"

            self.refs= re.findall(r"01-[0-9]{8,}[xX0-9\-]+",self.text)
            self.refs=np.unique(self.refs)

        except BaseException as e:
            print str(e)
            self.error=str(e)
            self.pdf_type="Unreadable"

    def OpenFile(filename):
        os.startfile(filename)




############################################################################################################################################
if __name__ == '__main__':
    inputfolder="C:/Users/yxie1/Documents/WIP/Playground/Unstructured_Data_Extraction/unstructred_data_HTML/"
    file1=inputfolder+"ICS_CCA07_M_A330_9.html"
    file2=inputfolder+"XFA_A_XFA01_ITCM_MoM_Seats.html"
    file3=inputfolder+"USA YC MoM  ITCM.html"
    file4=inputfolder+"Catering list equipment_G2535FM0902936_Rev D_20100917.html"
    file5=inputfolder+"IBE-R_ITCM_MoM_ YC Seats.html"

    inputfile=file3

    outputfolder="C:/Users/yxie1/Documents/WIP/Playground/Unstructured_Data_Extraction/"
    outputfile=outputfolder+"retext.txt"
    f=open(outputfile,'w')

    soup=BeautifulSoup(open(inputfile).read(), 'html.parser')

    for elem in soup.findAll(['style','title']):
        elem.extract()

    # sub=soup.new_string(' new ')
    for elem in soup.findAll('p'):
        elem.insert_before(' ')
        elem.insert_after('\n ')
        # print elem.get_text()

    for elem in soup.findAll('br'):
        elem.insert_before(' ')
        elem.insert_after('\n ')

    text=soup.get_text()
    retext=re.sub('[^A-Za-z0-9\*\/\-\.\(\)\\n]+', ' ', text)
    print >>f, retext
    f.close()
    OpenFile(inputfile)