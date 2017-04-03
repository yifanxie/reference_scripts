__author__ = 'YXIE1'

import time

from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfdevice import PDFDevice
import sys
import re
import FileCrawler


def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = file(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text




############################################################################################################################################
if __name__ == '__main__':
    # Open a PDF file.
    # fp = open('Readable.pdf', 'rb')

    # Create a PDF parser object associated with the file object.
    # parser = PDFParser(fp)

    # Create a PDF document object that stores the document structure.
    # Supply the password for initialization.
    # document = PDFDocument(parser, "test")

    # Check if the document allows text extraction. If not, abort.
    # if not document.is_extractable:
    #     print "not extractable"
    #     # raise PDFTextExtractionNotAllowed
    # else:
    #     print "extractable"
    start_time=time.time()


    #C:\Users\yxie1\Documents\WIP\Playground\PDF_Mining\pdf_files\BMX_pdf
    inputfolder="C:/Users/yxie1/Documents/WIP/Playground/PDF_Mining/pdf_files/BMX_pdf/"
    file1="lehsfil1.pdf"
    file2="240000067891 - De - Hysol EA 9695.050.pdf"
    file2b="O240000067891 - De - Hysol EA 9695.050.pdf"
    file3="240000054321 - En - HTS 1773.pdf"
    file4="240000012345 - En - Metco 404.pdf"
    file5="240000008501 - Fr - finavestan.pdf"
    file6="240000000017 - Fr - 0121 MAKE UP.pdf" #01-2119457290-43-XXXX


    ## potential pattern to be matched
    # -	01-2119457290-43-XXXX in the 1st one
    # -	01-2119487078-27 in the 2nd
    # -	01-2119384822-32-0032 in the 4th


    teststring="this string has 01-2119458838-20-xxxx in it "
    inputfile=inputfolder+file6

    try:
        text=convert(inputfile)

        print "the input file is %s" %inputfile
        text=re.sub('[^A-Za-z0-9\-]+', ' ', text)

        ref= re.findall(r"01-[0-9]{8,}[xX0-9\-]+",text)
        print "the extracted ref is %s" %ref
    # print ref
    except BaseException as e:
        print str(e)



    end_time=time.time()
    duration=end_time-start_time
    print "it takes %.3f seconds to study the document"  %(duration)




#  python regex example:
# text3=re.sub('[^A-Za-z0-9]+', ' ', text) # remove all non alpha numeric
# re.findall(r"01-\w+-\w+-\w+", text2)


# This is what I want!!!!!
# text4 = re.sub('[^A-Za-z0-9\-]+', ' ', text)
# re.findall(r"01-\w+-\w+-\w+", text2)
# re.findall(r"01-[0-9]{8,}[X0-9\-]+",text4)