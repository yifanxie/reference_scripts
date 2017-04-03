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
            identified_terms=np.unique(np.char.strip(identified_terms))
            identified_terms=' , '.join(identified_terms)
            # if len(lookforward_keyword)>0:
            #     identified_terms=identified_terms.replace(lookforward_keyword,"")

        else:
            identified_terms=' , '.join(identified_terms)
        terms.append(identified_terms)
    return terms


###########################################################################################################
# given a body of text, and a re object, return the unique list of terms identifed, as specified by the re object
def term_items_identification(texts, patr, lookforward_keyword='', lookbackbord_keyword=''):
    terms=[]
    for i in xrange(0, len(texts)):
        text=texts[i]
        identified_terms=patr.findall(text)
        if len(identified_terms)>0:
            identified_terms=np.unique(np.char.strip(identified_terms)).tolist()
            terms+=identified_terms
    return set(terms)


###############################################################################################################
# Given two list of terms - "terrm_list" and "term_values" extract phrase that start and end with entries in the two lists
def term_value_identification(texts, term_list, term_values):
    id_term_value_combo=[]
    # id_term_value_combo=[]
    for i in xrange(0, len(texts)):
        text=texts[i]
        term_list_crit=" | ".join(term_list.apply(str))
        patr_term_list=re.compile(term_list_crit)
        potential_terms=patr_term_list.findall(text)
        if len(potential_terms)>0:
            potential_terms=np.unique(np.char.strip(potential_terms))

        term_value_combo=[]
        for j in xrange(0, len(potential_terms)):
            term=potential_terms[j]
            term_value_crit=" | ".join(term_values.apply(str))
            term_value_crit=term_value_crit.replace('(','\(').replace(')','\)')

            compile_str=r"(?<= "+term+")[ ]{0,2}[:]{0,1}[ ]{0,5}("+term_value_crit+")"
            patr_term_value=re.compile(compile_str)
            potential_values=patr_term_value.findall(text)

            if len(potential_values)>0:
                identified_values=np.unique(np.char.strip(potential_values))
                identified_values=' , '.join(identified_values)
                # term_value_combo.append([i, term, identified_values])
                term_value_combo.append(term + " " + identified_values)

        if len(term_value_combo)>0:
            id_term_value_combo.append("".join(term_value_combo))
        else:
            id_term_value_combo.append("")
    return id_term_value_combo



###########################################################################################################
def term_value_identification_gen(texts, term_list, term_values):
    id_term_value_combo=[]

    for i in xrange(0, len(texts)):
        text=texts[i]
        term_list_crit="|".join(term_list.apply(str))
        term_list_crit=term_list_crit.replace('(','\(').replace(')','\)').replace('.','\.')
        term_list_crit=r'\b('+term_list_crit+r')\b'


        term_value_crit="|".join(term_values.apply(str))
        term_value_crit=term_value_crit.replace('(','\(').replace(')','\)').replace('.','\.')
        term_value_crit=r'\b('+term_value_crit+r')\b'


        # crit1=r'(\b('+class_combile+r')\b[\w\:\s]+\b(5810|1284)\b)(?:\s|$)'

        compile_str=r"(("+term_list_crit+")[\w\:\, ]+("+term_value_crit+")(?:\s|$|\n))"
        # compile_str=r"("+term_list_crit+")[ ]{0,2}[:]{0,1}[ ]{0,5}("+term_value_crit+")"
        patr_term_value=re.compile(compile_str, re.IGNORECASE)
        identified_values=patr_term_value.findall(text)
        if len(identified_values)>0:
            identified_values=np.unique(np.array(identified_values)[:,0])
            identified_values=' , '.join(identified_values.tolist())
            # print (i, identified_values)
            id_term_value_combo.append(identified_values)
        else:
            id_term_value_combo.append("")
        # print identified_values
    return id_term_value_combo



############################################################################################################################################
if __name__ == '__main__':
    start_time=time.time()

    file_name='./Domain_Semantics.xlsx'
    xl_file = pd.ExcelFile(file_name)
    df=xl_file.parse("Terminology")
    suppliers=pd.Series(df['Suppliers'].unique()).dropna().order(ascending=False)
    seat_models=pd.Series(df['Seat Models'].unique()).dropna().order(ascending=False)
    seat_classes=pd.Series(df['Seat Classes'].unique()).dropna().order(ascending=False)
    color=pd.Series(df['Color'].unique()).dropna().order(ascending=False)
    dado_panel=pd.Series(df['dado panels'].unique()).dropna().order(ascending=False)
    seat_parts=pd.Series(df['Seat Parts'].unique()).dropna().order(ascending=False)



    folderpickle="C:/Users/yxie1/Documents/WIP/Playground/Unstructured_Data_Extraction/pickledata/folder_info_woImg_ADB11.data"
    folder_data=load_data(folderpickle)

    texts=folder_data['text'].as_matrix()


    # set up regular expression (re) object for term extraction
    # supplier_crit=r'(\b('+"|".join(suppliers)+')\b)'
    supplier_crit=r'\b('+"|".join(suppliers)+r')\b'
    patr_supplier=re.compile(supplier_crit, re.IGNORECASE)

    seat_crit="|".join(seat_models.apply(str)).replace('(','\(').replace(')','\)')
    seat_crit=r'\b('+seat_crit+r')\b'
    patr_seat_models=re.compile(seat_crit, re.IGNORECASE)

    color_crit="|".join(color.apply(str)).replace('(','\(').replace(')','\)')
    color_crit=r'\b('+color_crit+r')\b'
    patr_color=re.compile(color_crit, re.IGNORECASE)

    dado_crit="|".join(dado_panel.apply(str)).replace('(','\(').replace(')','\)')
    dado_crit=r'\b('+dado_crit+r')\b'
    patr_dado=re.compile(dado_crit, re.IGNORECASE)

    seat_parts_crit="|".join(seat_parts.apply(str)).replace('(','\(').replace(')','\)')
    seat_parts_crit=r'\b('+seat_parts_crit+r')\b'
    patr_seat_parts=re.compile(seat_parts_crit, re.IGNORECASE)


    patr_msn=re.compile(r"(?<=MSN)[ ]{0,2}[:]{0,1}[ ]{0,5}[0-9]+")
    patr_classes=re.compile(r"[B-Z]{1,2}/C", re.IGNORECASE)


    # perform re recognition
    suppliers_terms=term_identification(texts, patr_supplier)
    seat_model_terms=term_identification(texts, patr_seat_models)
    MSN=term_identification(texts, patr_msn, "MSN")
    # classes_terms=term_items_identification(texts, patr_classes)


    folder_data['suppliers']=pd.Series(suppliers_terms, index=folder_data.index).astype(str)
    folder_data['seat models']=pd.Series(seat_model_terms, index=folder_data.index).astype(str)
    folder_data['MSN']=pd.Series(MSN, index=folder_data.index).astype(str)

    Seat_classes_model=term_value_identification_gen(texts,seat_classes, seat_models)
    folder_data['Class_Model']=pd.Series(Seat_classes_model, index=folder_data.index).astype(str)

    dado_color=term_value_identification_gen(texts,dado_panel, color)
    folder_data['Dado Panel_Color']=pd.Series(dado_color, index=folder_data.index).astype(str)

    seatpart_color=term_value_identification_gen(texts,seat_parts, color)
    folder_data['Seat Parts_Color']=pd.Series(seatpart_color, index=folder_data.index).astype(str)


    # write_data=folder_data.drop(['type','ref','err','text'], axis=1)

    write_data_cols=['name','suppliers','seat models',
                     'Class_Model',
                     'Dado Panel_Color',
                     'Seat Parts_Color',
                     'MSN', 'path']
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


    # compile_str=r"(?<= "+term+")[ ]{0,2}[:]{0,1}[ ]{0,5}("+term_value_crit+")"

    # outputfile="C:/Users/yxie1/Documents/WIP/Playground/Unstructured_Data_Extraction/text.txt"
    # f=open(outputfile,'w')
    # text=folder_data.iloc[171,5]
    # print >>f, text

    crit1=r'(\b(Attendant seat|Stowage|walls)\b(\s[0-9a-zA-Z/]+){0,3}\s\b(shrouds|belts|toilets|toilet|handrails|Decor)\b)'