__author__ = 'yxie1'

__author__ = 'YXIE1'


import pandas as pd
import time
import numpy as np
import cPickle
import re

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

###########################################################################################################
# given a body of text, and a re object, return the unique list of terms identifed, as specified by the re object
def term_identification(texts, patr, lookforward_keyword='', lookbackbord_keyword=''):
    terms=[]
    for i in xrange(0, len(texts)):
        text=texts[i]
        identified_terms=patr.findall(text)
        if len(identified_terms)>0:
            identified_terms=np.unique(np.char.strip(identified_terms)).tolist()
            terms+=identified_terms

    return set(terms)


###########################################################################################################
# given a body of text, and a re object, return the unique list of terms identifed in the text body
def term_items_identification(texts, patr, lookforward_keyword='', lookbackbord_keyword=''):
    terms=[]
    for i in xrange(0, len(texts)):
        text=texts[i]
        identified_terms=patr.findall(text)
        if len(identified_terms)>0:
            identified_terms=np.unique(np.char.strip(identified_terms))
            identified_terms=' , '.join(identified_terms)
            if len(lookforward_keyword)>0:
                identified_terms=identified_terms.replace(lookforward_keyword,"")
        else:
            identified_terms=' , '.join(identified_terms)
        terms.append(identified_terms)
    return terms

############################################################################################################################################
if __name__ == '__main__':
    start_time=time.time()
    file_name='./LAST_SEATS_LR.xlsx'
    xl_file = pd.ExcelFile(file_name)
    df=xl_file.parse("Seat Model")
    # seat_classes=xl_file.parse("Seat Classes")
    suppliers=pd.Series(df['R27_SUPPLIER'].unique())
    seat_models=pd.Series(df['R27_MODEL'].unique())



    folderpickle="C:/Users/yxie1/Documents/WIP/Playground/Unstructured_Data_Extraction/pickledata/folder_info_br.data"
    folder_data=load_data(folderpickle)

    texts=folder_data['text'].as_matrix()


    # set up regular expression (re) object for term extraction
    patr_supplier=re.compile(" | ".join(suppliers))
    seat_combile=" | ".join(seat_models.apply(str))
    seat_combile=seat_combile.replace('(','\(').replace(')','\)')
    patr_seat_models=re.compile(seat_combile)
    patr_msn=re.compile(r"(?<=MSN)[ ]{0,2}[:]{0,1}[ ]{0,5}[0-9]+")
    patr_classes=re.compile(r"[B-Z]{1,2}/C")

    # perform re recognition
    suppliers_terms=term_identification(texts, patr_supplier)
    seat_model_terms=term_identification(texts, patr_seat_models)
    classes_terms=term_identification(texts, patr_classes)
    MSN=term_identification(texts, patr_msn, "MSN")


    folder_data['suppliers']=pd.Series(suppliers_terms, index=folder_data.index).astype(str)
    folder_data['seat models']=pd.Series(seat_model_terms, index=folder_data.index).astype(str)
    folder_data['MSN']=pd.Series(MSN, index=folder_data.index).astype(str)

    #
    # write_data=folder_data.drop(['type','ref','err','text'], axis=1)

    write_data_cols=['name','suppliers','seat models', 'MSN', 'path']
    write_data=folder_data[write_data_cols]



    try:
        writer = pd.ExcelWriter('file_matrix.xlsx')
        write_data.to_excel(writer, sheet_name='sheet')
        writer.save()
    except:
        print "document is opened, saving is aborted"

    picklepath="./pickledata/folder_info_termextraction.data"
    pickle_data(picklepath, write_data)

    end_time=time.time()
    duration=end_time-start_time
    print "it takes %.3f seconds to study the document"  %(duration)