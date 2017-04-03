__author__ = 'YXIE1'

import os
import time
import cPickle
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import numpy as np
import re
import pandas as pd
import csv
import sys
import chardet

reload(sys)
sys.setdefaultencoding('utf-8')


def OpenFile(filename):
    os.startfile(filename)


########################################################################################################################
###### class object holding pdf file information
class PDFfile(object):
    text=[]
    error=[]
    pdf_type=''
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


    def convert(self, filename, pages=None):
        if not pages:
            pagenums = set()
        else:
            pagenums = set(pages)
        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)

        infile = file(filename, 'rb')
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        text = output.getvalue()
        output.close
        return text

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

########################################################################################################################
###### class object holding folder information
class folder(object):
    readable_pdfs=[]
    tobechecked_pdfs=[]
    unreadable_pdfs=[]
    ref_pdfs=[]
    noref_pdfs=[]
    # propoerty holding path for all n-1 level subfolders
    subfolders=[]

    # property holding path for all files in all level
    files=[]

    def __init__(self, inputfolder):
        if os.path.isdir(inputfolder):
            self.path=inputfolder

    def filecrawling(self):
        for root, dirs, files in os.walk(unicode(self.path)):
            for name in files:
                file_name=os.path.join(root, name)
                file_name=re.sub(r'\\', r'/',file_name)
                if (os.path.splitext(file_name)[1]).lower()=='.pdf':
                    aPDF=PDFfile(file_name)
                    self.files.append(aPDF)
                    dirname=os.path.basename(os.path.dirname(file_name))
                    try:
                        print "Folder - %s, file %s is %s" %(dirname, os.path.basename(file_name), aPDF.pdf_type)
                    except:
                        print "print error"

                    if aPDF.pdf_type[0]=='R':
                        self.readable_pdfs.append(aPDF)
                    if aPDF.pdf_type[0]=='T':
                        self.tobechecked_pdfs.append(aPDF)
                    if aPDF.pdf_type[0]=='U':
                        self.unreadable_pdfs.append(aPDF)

                    if len(aPDF.refs)>0:
                        self.ref_pdfs.append(aPDF)
                    else:
                        self.noref_pdfs.append(aPDF)


            for name in dirs:
                dir_name=os.path.join(self.path, name)
                self.subfolders.append(dir_name)


    def save_fileinfo(self):
        name=[]
        path=[]
        type=[]
        text=[]
        ref=[]
        err=[]
        for afile in self.files:
            try:
                # print afile.name
                name.append(afile.name)
                # print afile.path
                path.append(afile.path)
                # print afile.pdf_type
                type.append(afile.pdf_type)
                # print len(afile.text)
                text.append(afile.text)
                # print afile.refs
                ref.append(afile.refs)
                # print afile.error
                err.append(afile.error)
            except:
                print "printing error"

        output=pd.DataFrame({"name": name, "path":path, "type":type, "text":text, "ref":ref, "err":err},
                            columns=["name", "path", "type", "ref", "err", "text"])
        output.to_csv("file_info.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')

        picklepath="./pickledata/folder_info.data"
        pickle_data(picklepath, output)

###########################################################################################################
# allowing loading data from pickled file
def load_data(pickle_file):
    load_file=open(pickle_file,'rb')
    data=cPickle.load(load_file)
    return  data

###########################################################################################################
# allowing saving data into a pickle file
def pickle_data(path, data):
    file=path
    save_file=open(file, 'wb')
    cPickle.dump(data, save_file)
    save_file.close()


# def read_csv():



########################################################################################################################
if __name__ == '__main__':
    start_time=time.time()
    # inputfolder="C:/Users/yxie1/Documents/WIP/Playground/PDF_Mining/pdf_files/In_Service_Repair/"
    inputfolder="C:/Users/yxie1/Documents/WIP/Archives_MSDS/"
    testfolder=folder(inputfolder)
    testfolder.filecrawling()
    print "%d files have been study." %len(testfolder.files)
    print "%d readable files." %len(testfolder.readable_pdfs)
    print "%d tobechecked files."  %len(testfolder.tobechecked_pdfs)
    print "%d unreadable files." %len(testfolder.unreadable_pdfs)
    print "%d files are found to contain targeted ref patten." %len(testfolder.ref_pdfs)

    # save folder information into 'folder_info.data'
    testfolder.save_fileinfo()


    # picklepath="./pickledata/folderobject.data"
    # pickle_data(picklepath, testfolder)
    # test_info=pd.read_csv('./file_info.csv')

    end_time=time.time()
    duration=end_time-start_time
    print "it takes %.3f seconds to study the document"  %(duration)

    #started at line 170 in console