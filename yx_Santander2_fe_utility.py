# **** Disclaimer ^_^
#//When I wrote this, only God and I understood what I was doing
# //Now, God only knows

import pickle
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import feather
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import gc
# pickle load utility
def pickle_load(file_path):
    start_time = time.time()
    data_file = open(file_path, 'rb')
    data = pickle.load(data_file)
    print("finished loading data, took %.2f minutes" % ((time.time() - start_time) / 60))
    return data

# pickle_save utility
def pickle_save(data, file_path):
    start_time=time.time()
    output=open(file_path, 'wb')
    pickle.dump(data, output)
    output.close()
    print("finished saving data, took %.2f minutes" %((time.time()-start_time)/60))


# label_encoding with nan impute to -1
def label_encoding(train, test, colname, train_and_test=True):
    start_time = time.time()
    le = LabelEncoder()
    # create non-null mask
    train_mask = ~train[colname].isnull()
    if train_and_test:
        test_mask = ~test[colname].isnull()

    # fit and transform with label encoder
    train.loc[train_mask, colname] = le.fit_transform(train[colname][train_mask])
    train.loc[-train_mask, colname] = -1

    if train_and_test:
        test.loc[test_mask, colname] = le.transform(test[colname][test_mask])
        test.loc[-test_mask, colname] = -1

    print("finished label encoding, took %.2f minutes" %((time.time()-start_time)/60))
    if train_and_test:
        return train[colname], test[colname]
    else:
        return train[colname]


# transform train data to generate the purchase record for each month.
# use month1 as previous month record
# use month2 as the current that purchase records to be generated
# test_head: only used as experimental testing purpose
def train_data_transformation(train, month1, month2, test_head=-1):
    feat_list = ['ncodpers']
    for col_id in range(24, 48):
        feat = train.columns[col_id]
        feat_list.append(feat)

    m1 = train.loc[train.fecha_dato == month1, feat_list]
    m2 = train.loc[train.fecha_dato == month2, feat_list]

    m1np = m1.as_matrix()
    m2np = m2.as_matrix()

    ncodpers_m1 = m1.ncodpers.values
    ncodpers_m2 = m2.ncodpers.values

    ncodpers_m21_diff = list((set(ncodpers_m2) - set(ncodpers_m1)))
    ncodpers_m12_interset = list(set(ncodpers_m2).intersection(set(ncodpers_m1)))

    if test_head>-1:
        ncodpers_m12_interset=ncodpers_m12_interset[0:test_head]
        ncodpers_m21_diff = ncodpers_m21_diff[0:100]
    start_time = time.time()
    purchase_collections = []
    count = 0
    for code in ncodpers_m12_interset:
        count += 1
        if count > 0 and (count % 10000) == 0:
            print(count)
        code_arg_m1 = np.argwhere(ncodpers_m1 == code)
        val_m1 = m1np[code_arg_m1[0][0], 1:]
        code_arg_m2 = np.argwhere(ncodpers_m2 == code)
        val_m2 = m2np[code_arg_m2[0][0], 1:]
        purchase_products = np.nonzero((val_m2 == 1) & (val_m1 < val_m2))[0]

        for ind in range(0, len(purchase_products)):
            if len(purchase_collections) == 0:
                purchase_collections = np.array([code, month2, purchase_products[ind]])
            else:
                purchase_collections = np.vstack((purchase_collections, [code, month2, purchase_products[ind]]))

    for code in ncodpers_m21_diff:
        code_arg_m2 = np.argwhere(ncodpers_m2 == code)
        val_m2 = m2np[code_arg_m2[0][0], 1:]
        purchase_products = np.nonzero(val_m2 == 1)[0]
        for ind in range(0, len(purchase_products)):
            purchase_collections = np.vstack((purchase_collections, [code, month2, purchase_products[ind]]))

    index = range(0, len(purchase_collections))
    columns = ['ncodpers', 'fecha_dato', 'product_class']
    purchase_collections_df = pd.DataFrame(purchase_collections, index=index, columns=columns)
    purchase_collections_df.ncodpers = purchase_collections_df.ncodpers.astype(np.int32)
    purchase_collections_df.fecha_dato = pd.to_datetime(purchase_collections_df.fecha_dato)
    purchase_collections_df.product_class = purchase_collections_df.product_class.astype(np.int8)
    print("finished data transformation, took {:.2f} minutes".format((time.time() - start_time) / 60))
    return purchase_collections_df


# create a range of customer data from 'month_from' to 'month_to'
def create_monthly_customer_data(train, month_from, month_to=None):
    cust_feat = ['fecha_dato', 'ncodpers']
    for col_id in range(2, 24):
        feat = train.columns[col_id]
        cust_feat.append(feat)
    if month_to is None:
        validation = train.loc[train.fecha_dato == month_from, cust_feat]
    elif month_from < month_to:
        validation = train.loc[(train.fecha_dato >= month_from) & (train.fecha_dato <= month_to), cust_feat]
    else:
        print("please make sure the 'month_from' value is smaller than the 'month_to' value")
        validation = []
    return validation



# average precision function tailored for Santander Product Recommendation
def apk(actual, predicted, k=7):
    if type(np.isnan(actual)) is np.bool_ or type(np.isnan(predicted)) is np.bool_:
        return 0.0
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    return score / min(len(actual), k)

# mean average precision function tailored for Santander Product Recommendation
def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# function used to generated output dataframe in submission format.
def create_output_df(data, preds, train):
    target_cols = []
    for col_id in range(24, 48):
        feat = train.columns[col_id]
        target_cols.append(feat)
    target_cols = np.array(target_cols)
    ids=np.array(data.ncodpers)
    final_preds = [" ".join(list(target_cols[i])) for i in preds]
    out_df = pd.DataFrame({'ncodpers': ids, 'added_products': final_preds})[['ncodpers', 'added_products']]
    return out_df


def create_probability_matrix_bak(pred, train, ignored_list=[]):
    pdf=set_products_df(train, ignored_list)
    index = np.arange(len(pred))
    col_list = pdf.ES.values
    prob_matrix = pd.DataFrame(index=index, columns=col_list)
    for i in range(0, 24):
        adjusted_class = pdf.adjusted_class[i]
        if adjusted_class < 0:
            prob_matrix[col_list[i]] = 0
        else:
            prob_matrix[col_list[i]] = pred[:, adjusted_class]
    return prob_matrix

def create_probability_matrix(pred, train, test_ncodpers, ignored_list=[]):
    pdf=set_products_df(train, ignored_list)
    index = np.arange(len(pred))
    col_list = ['ncodpers']+pdf.ES.values.tolist()
    prob_matrix = pd.DataFrame(index=index, columns=col_list)
    prob_matrix['ncodpers']=test_ncodpers
    for i in range(0, 24):
        adjusted_class = pdf.adjusted_class[i]
        if adjusted_class < 0:
            prob_matrix[col_list[i+1]] = 0
        else:
            prob_matrix[col_list[i+1]] = pred[:, adjusted_class]
    return prob_matrix



# create a list of datatime object representing each months in the data
def set_months():
    months=(pd.datetime(2015, 1, 28),
            pd.datetime(2015, 2, 28),
            pd.datetime(2015, 3, 28),
            pd.datetime(2015, 4, 28),
            pd.datetime(2015, 5, 28),
            pd.datetime(2015, 6, 28),
            pd.datetime(2015, 7, 28),
            pd.datetime(2015, 8, 28),
            pd.datetime(2015, 9, 28),
            pd.datetime(2015, 10, 28),
            pd.datetime(2015, 11, 28),
            pd.datetime(2015, 12, 28),
            pd.datetime(2016, 1, 28),
            pd.datetime(2016, 2, 28),
            pd.datetime(2016, 3, 28),
            pd.datetime(2016, 4, 28),
            pd.datetime(2016, 5, 28),
            pd.datetime(2016, 6, 28)
            )
    return months

# create a list of name directly referencing the purchase record files generated in feather format
def set_purchase_files():
    purchase_files = ['./data/purchase_201501.feather',
                      './data/purchase_201502.feather',
                      './data/purchase_201503.feather',
                      './data/purchase_201504.feather',
                      './data/purchase_201505.feather',
                      './data/purchase_201506.feather',
                      './data/purchase_201507.feather',
                      './data/purchase_201508.feather',
                      './data/purchase_201509.feather',
                      './data/purchase_201510.feather',
                      './data/purchase_201511.feather',
                      './data/purchase_201512.feather',
                      './data/purchase_201601.feather',
                      './data/purchase_201602.feather',
                      './data/purchase_201603.feather',
                      './data/purchase_201604.feather',
                      './data/purchase_201605.feather'
                      ]
    return purchase_files

# create a list of product with product id (0-23), Spanish name and English name
# ignore_list: a list of products in spanish name to be ignored
def set_products(train, ignore_list=[]):
    products = []
    products_en=['Saving Account', 'Guarantees', 'Current Accounts', 'Derivada Accounts',
                 'Payroll Account', 'Junior Account', 'Mas particular Account', 'particular Account',
                 'particular Plus Account', 'Short-term deposits', 'Medium-term deposits', 'Long-term deposits',
                 'e-account', 'Funds', 'Mortgage', 'Pensions',
                 'Loans', 'Taxes', 'Credit Card', 'Securities',
                 'Home Account', 'Payroll', 'Pensions', ' Direct Debit']
    pid=0
    for col_id in range(24, 48):
        feat = train.columns[col_id]
        if feat in ignore_list:
            products.append([-1, col_id - 24, feat, products_en[col_id - 24]])
        else:
            products.append([pid, col_id-24, feat, products_en[col_id-24]])
            pid+=1
    return products

# create a dataframe of product with product id (0-23), Spanish name and English name
# ignore_list: a list of products in spanish name to be ignored
def set_products_df(train, ignore_list=[]):
    products_en=['Saving Account', 'Guarantees', 'Current Accounts', 'Derivada Accounts',
                 'Payroll Account', 'Junior Account', 'Mas particular Account', 'particular Account',
                 'particular Plus Account', 'Short-term deposits', 'Medium-term deposits', 'Long-term deposits',
                 'e-account', 'Funds', 'Mortgage', 'Pensions',
                 'Loans', 'Taxes', 'Credit Card', 'Securities',
                 'Home Account', 'Payroll', 'Pensions', ' Direct Debit']
    columns=['product_class','adjusted_class', 'ES', 'EN']
    index=np.arange(24)
    products_df=pd.DataFrame(index=index, columns=columns)

    pids=[]
    apids=[]
    es=[]
    en=[]
    apid_count=0
    negid=-1
    for col_id in range(24, 48):
        feat = train.columns[col_id]
        pids.append(col_id - 24)
        if feat in ignore_list:
            apids.append(negid)
            negid-=1
        else:
            apids.append(apid_count)
            apid_count+=1

        es.append(feat)
        en.append(products_en[col_id-24])
    products_df['product_class']=pids
    products_df['adjusted_class']=apids
    products_df['ES']=es
    products_df['EN']=en
    return products_df



# create labels for local validation in format acceptable for mapk
def create_val_labels(validation, valid_purchase, step=2000):
    start_time = time.time()
    validation['label'] = np.nan
    validation['label'] = validation['label'].astype(object)
    labels = validation['label'].values
    valid_ncodpers = validation['ncodpers'].values
    purchase_ncodpers = valid_purchase.ncodpers.unique()
    count = 0
    for code in purchase_ncodpers:
        count += 1
        if count > 0 and (count % step) == 0:
            print(count)
        code_arg = np.argwhere(valid_ncodpers == code)[0][0]
        labels[code_arg] = valid_purchase.loc[valid_purchase.ncodpers == code, 'product_class'].values.tolist()

    validation['label'] = labels
    print("finished creating validation labels, took {:.2f} minutes".format((time.time() - start_time) / 60))
    return validation[['fecha_dato', 'ncodpers', 'label']]


# adjust classes in purchase information for ignored_list
def adjust_purchase(purchase, train, ignored_list=[]):
    #perform class adjustment if ignored_list is not empty
    if len(ignored_list)>0:
        pdf=set_products_df(train,ignored_list)
        purchase = pd.merge(purchase, pdf[['product_class', 'adjusted_class']], how='left', on=['product_class'])
        purchase['product_class'] = purchase['adjusted_class']
        purchase.drop(['adjusted_class'], axis=1, inplace=True)
        purchase = purchase[purchase.product_class >=0]
    return purchase

# restored the adjusted classes label back to original product label after prediction.
# this is required for accurately calculate mapk scores
def restore_product_class(preds, train, ignored_list):
    pdf = set_products_df(train, ignored_list)
    for pid in range(0, 23):
        product_class = 23 - pid
        adjusted_class = pdf.loc[pdf.product_class == product_class, 'adjusted_class'].values[0]
        preds[preds == adjusted_class] = product_class
    return preds

# create loccal training and validatoin dataset with data corresponding to the given month_id
# also capture customer record of product profile in the previous month (month_id-1)
def create_local_validation_data(train, data, month_id):
    months = set_months()
    cust_feat = ['fecha_dato', 'ncodpers']
    for col_id in range(2, 24):
        feat = train.columns[col_id]
        cust_feat.append(feat)
    # product_feat = ['ncodpers']
    # for col_id in range(24, 48):
    #     feat = train.columns[col_id]+'-1'
    #     product_feat.append(feat)
    month_data = train.loc[train.fecha_dato == months[month_id], cust_feat]
    # pre_month_product_data = train.loc[train.fecha_dato == months[month_id - 1], product_feat]
    # product_feat.remove('ncodpers')
    output = pd.merge(data, month_data, how='left', on=['fecha_dato', 'ncodpers'])
    # output = pd.merge(output, pre_month_product_data, how='left', on=['ncodpers'])
    # output.loc[:, product_feat] = output.loc[:, product_feat].fillna(0).astype(np.int8)
    return output

def fe_minus_n_month_feature(train, data, month_id, n=2):
    months=set_months()
    cust_feat = ['fecha_dato', 'ncodpers']
    for col_id in range(2, 24):
        feat = train.columns[col_id]
        cust_feat.append(feat)
    product_feat = ['ncodpers']
    for col_id in range(24, 48):
        feat = train.columns[col_id]
        product_feat.append(feat)

    minus_n_month_data=train.loc[train.fecha_dato == months[month_id - n], product_feat]
    # print(minus_n_month_data.shape)
    minus_n_month_feats=[]
    for feat in product_feat:
        if feat!='ncodpers':
            new_feat_name=feat+'-'+str(n)
            minus_n_month_data.rename(columns={feat:new_feat_name}, inplace=True)
            minus_n_month_feats.append(new_feat_name)

    # print(minus_n_month_data.columns)
    output=pd.merge(data, minus_n_month_data, how='left', on=['ncodpers'])
    output.loc[:, minus_n_month_feats] = output.loc[:, minus_n_month_feats].fillna(0).astype(np.int8)
    # output['adjusted_product_sum-'+str(n)]=output[minus_n_month_feats].sum(axis=1)
    return output


def fe_multiple_months_lag(train, data, month_list, n=2):
    months = set_months()
    count=0
    for i in month_list:
        month_data=data.loc[data.fecha_dato==months[i]]
        # print(month_data.columns)
        month_data_with_lag=fe_minus_n_month_feature(train, month_data, i, n)
        # print(month_data_with_lag.shape)
        if count==0:
            output=month_data_with_lag
            count+=1
        else:
            output=pd.concat([output, month_data_with_lag], axis=0)

        # print(output.columns)
    return output




# create test data with customer record of product profiles for May 2016
def create_test_data(train, data):
    months = set_months()
    cust_feat = ['fecha_dato', 'ncodpers']
    for col_id in range(2, 24):
        feat = train.columns[col_id]
        cust_feat.append(feat)
    product_feat = ['ncodpers']
    for col_id in range(24, 48):
        feat = train.columns[col_id]
        product_feat.append(feat)
    pre_month_product_data = train.loc[train.fecha_dato == months[16], product_feat]
    product_feat.remove('ncodpers')
    # print(months[16])
    # print(pre_month_product_data.shape)
    # print(pre_month_product_data.info())
    # print(pre_month_product_data.head(5))
    test_data = pd.merge(data[cust_feat], pre_month_product_data, how='left', on=['ncodpers'])
    # print(test_data.info())
    test_data.loc[:, product_feat] = test_data.loc[:, product_feat].fillna(0).astype(np.int8)
    return test_data


# create purchase data with customer features for each months
def create_monthly_purchase(train):
    months = set_months()
    purchase_files = set_purchase_files()
    purchases = []
    for i in range(0, 17):
        if i == 0:
            purchases.append([])
        else:
            purchase = feather.read_dataframe(purchase_files[i])
            purchase = create_local_validation_data(train, purchase, months[i])
            print(months[i], purchase.shape)
            purchases.append(purchase)
    return purchases


def create_purchase_train(train, month_list, ignored_list=[]):
    purchase_files = set_purchase_files()
    purchases = []
    for i in range(len(month_list)):
        month_id=month_list[i]
        purchase = feather.read_dataframe(purchase_files[month_id])
        purchase = adjust_purchase(purchase, train, ignored_list)
        purchase = create_local_validation_data(train, purchase, month_id)
        if i==0:
            purchases=purchase
        else:
            purchases=pd.concat([purchases, purchase], axis=0)
    return purchases



# create full history of purchase data with customer feature for all available months
def create_full_purchase_history(train):
    months = set_months()
    purchase_files = set_purchase_files()
    purchases = []
    for i in range(0, 17):
        if i == 0:
            continue
        else:
            purchase = feather.read_dataframe(purchase_files[i])
            purchase = create_local_validation_data(train, purchase, months[i])
            if i==1:
                purchases=purchase
            else:
                purchases=pd.concat([purchases, purchase], axis=0)
    return purchases


# perform some simple treatment of the train and test data for model building
def datatreatment(data, labelname=None, local_valid=True, consider_dato=False, ignored_list=[]):
    #convert the -1 (na imputation) back to na, some classifier (like xgb can perform imputation on the run)
    # code to be added
    if consider_dato:
        data['dato_year']=data['fecha_dato'].dt.year
        data['dato_month'] = data['fecha_dato'].dt.month

    data['alta_year']=data['fecha_alta'].dt.year
        # data['alta_month']=data['fecha_alta'].dt.month
    # remove feature with low variant
    # data = data.drop(['ncodpers', 'fecha_dato', 'nomprov', 'fecha_alta', 'ult_fec_cli_1t', 'tipodom', 'conyuemp',
    #                   'ind_ahor_fin_ult1','ind_aval_fin_ult1', 'ind_deco_fin_ult1','ind_deme_fin_ult1'], axis=1)

    # data = data.drop(['ncodpers', 'fecha_dato', 'nomprov', 'fecha_alta', 'ult_fec_cli_1t'], axis=1)

    drop_list=['nomprov', 'fecha_alta', 'ult_fec_cli_1t', 'tipodom', 'conyuemp']
    drop_list=drop_list+ignored_list
    data = data.drop(drop_list, axis=1)


    if local_valid:
        label = data[labelname]
        data = data.drop([labelname], axis=1)
        return data, label
    else:
        return data

# output a data frame that provide insight on how the model perform on prediction by comparing
# actual label to prediction results
def prediction_diagnosis(act, pred, valid_month, train_local, train, ignored_list=[]):
    purchase_files = set_purchase_files()
    pdf=set_products_df(train, ignored_list)
    valid_purchase=feather.read_dataframe(purchase_files[valid_month])
    index = np.arange(24)
    columns = ['pid', 'ES', 'EN', 'Train', 'Predict', 'Actual','Miss']
    diagnosis_df = pd.DataFrame(index=index, columns=columns)
    pids = []
    es = []
    en = []
    train_counts = []
    pred_counts=[]
    valid_counts = []
    for pid in range(0, 24):
        adjust_class=pdf.loc[pdf.product_class==pid, 'adjusted_class'].values[0]
        train_count = train_local[train_local.product_class== adjust_class].shape[0]
        pred_count=pred[pred==pid].shape[0]
        valid_count = valid_purchase.loc[valid_purchase.product_class == pid].shape[0]
        pids.append(pid)
        es.append(pdf.loc[pdf.product_class==pid, 'ES'].values[0])
        en.append(pdf.loc[pdf.product_class==pid, 'EN'].values[0])
        train_counts.append(train_count)
        pred_counts.append(pred_count)
        valid_counts.append(valid_count)
    idx = np.arange(len(pred))
    miss = np.zeros([24])
    for i in idx:
        if not (type(np.isnan(act[i])) is np.bool_):
            diff = np.setdiff1d(act[i], pred[i])
            miss[diff] += 1
    diagnosis_df['pid'] = pids
    diagnosis_df['ES'] = es
    diagnosis_df['EN'] = en
    diagnosis_df['Train'] = train_counts
    diagnosis_df['Predict'] = pred_counts
    diagnosis_df['Actual'] = valid_counts
    diagnosis_df['Miss'] = np.int32(miss)
    diagnosis_df['Miss_ratio'] = diagnosis_df['Miss'] / diagnosis_df['Actual']
    return diagnosis_df


#age feature scaling
def getAge(age):
    mean_age=40.
    min_age=20.
    max_age=90.
    range_age=max_age-min_age
    age=age.astype(np.float)
    age[age==-1]=mean_age
    age[age<min_age]=min_age
    age[age>max_age]=max_age
    age=round((age-min_age)/range_age, 4)
    return age


def getCustSeniority(antiguedad):
    min_value=0.
    max_value=256.
    range_value=max_value-min_value
    missing_value=0
    antiguedad=antiguedad.astype(np.float)
    antiguedad[antiguedad==-1]=missing_value
    antiguedad[antiguedad<=min_value]=min_value
    antiguedad[antiguedad>=max_value]=max_value
    antiguedad=round((antiguedad-min_value)/range_value, 4)
    return antiguedad

def getRenta(renta):
    min_value=0.
    max_value=1500000.
    range_value=max_value-min_value
    missing_value=101850.
    renta=renta.astype(np.float)
    renta[renta==-1]=missing_value
    renta[renta<=min_value]=min_value
    renta[renta>=max_value]=max_value
    renta=round((renta-min_value)/range_value, 6)
    return renta

def LB_split_simulation(va_y, preds, sim_round=5):
    public_sim_scores = []
    private_sim_scores = []
    abs_diffs=[]
    val_size = len(va_y)
    sample_size = int(val_size * 0.3)
    for i in range(0, sim_round):
        ind = np.random.randint(val_size, size=sample_size)
        mask = np.ones(val_size, np.bool)
        mask[ind] = 0
        public_va_y = va_y[ind]
        public_preds = preds[ind]
        private_va_y = va_y[mask]
        private_preds = preds[mask]
        sim_public = mapk(public_va_y, public_preds)
        sim_private = mapk(private_va_y, private_preds)
        abs_diff=np.abs(sim_public-sim_private)
        print('simulated public score #{:d}: {:.7f}, simulated private score: {:.7f}'.format(i, sim_public, sim_private))
        public_sim_scores.append(sim_public)
        private_sim_scores.append(sim_private)
        abs_diffs.append(abs_diff)
    public_sim_scores = np.array(public_sim_scores)
    private_sim_scores = np.array(private_sim_scores)
    abs_diffs=np.array(abs_diffs)
    print('mean simulated public score is {:.7f} with std {:.8f}'.format(public_sim_scores.mean(),
                                                                         public_sim_scores.std()))
    print('mean simulated private score is {:.7f} with std {:.8f}'.format(private_sim_scores.mean(),
                                                                          private_sim_scores.std()))
    print('mean absolute difference is {:.8f}'.format(abs_diffs.mean()))
    return public_sim_scores, private_sim_scores, abs_diffs


# perform binary encoding for categorical variable
def fe_binary_encoding(train_x, valid_x, feat):
    train_feat_max = train_x[feat].max()
    valid_feat_max = valid_x[feat].max()

    if train_feat_max > valid_feat_max:
        feat_max = train_feat_max
    else:
        feat_max = valid_feat_max
    train_x.loc[train_x[feat] == -1, feat] = feat_max + 1
    valid_x.loc[valid_x[feat] == -1, feat] = feat_max + 1
    union_val = np.union1d(train_x[feat].unique(), valid_x[feat].unique())
    max_dec = union_val.max()
    max_bin_len = len("{0:b}".format(max_dec))
    index = np.arange(len(union_val))
    columns = list([feat])
    bin_df = pd.DataFrame(index=index, columns=columns)
    bin_df[feat] = union_val
    feat_bin = bin_df[feat].apply(lambda x: "{0:b}".format(x).zfill(max_bin_len))
    splitted = feat_bin.apply(lambda x: pd.Series(list(x)).astype(np.uint8))
    splitted.columns = [feat + '_bin_' + str(x) for x in splitted.columns]
    bin_df = bin_df.join(splitted)
    train_x = pd.merge(train_x, bin_df, how='left', on=[feat])
    # train_x = train_x.drop([feat], axis=1)
    valid_x = pd.merge(valid_x, bin_df, how='left', on=[feat])
    # valid_x = valid_x.drop([feat], axis=1)
    return train_x, valid_x




def fe_product_sum(train_x, valid_x, train):
    products = np.array(set_products(train))
    feats = []
    for col in range(0, len(train_x.columns)):
        feat = train_x.columns[col]
        if feat in products[:, 2]:
            feats.append(train_x.columns[col])
    train_x['prod_sum']=train_x[feats].sum(axis=1)
    valid_x['prod_sum']=valid_x[feats].sum(axis=1)
    return train_x, valid_x

def fe_frequency_encoding(train_x, valid_x, train, feat):
    colstr=[feat]
    cust=train.drop_duplicates('ncodpers', keep='last')
    freq_count=cust.groupby(colstr)['ncodpers'].agg({'count'})
    freq_count.reset_index(inplace=True)
    train_x=pd.merge(train_x, freq_count, how='left', on=colstr)
    valid_x=pd.merge(valid_x, freq_count, how='left', on=colstr)
    count_name=feat+'_'+'count'
    train_x.rename(columns={'count':count_name}, inplace=True)
    valid_x.rename(columns={'count': count_name}, inplace=True)
    # train_x.drop(colstr, axis=1)
    # valid_x.drop(colstr, axis=1)
    return train_x, valid_x


def fe_Categorical_Feature_Encoding(train_x, train_y, valid_x, train, feats, cat_op=1):
    for feat in feats:
        if cat_op==1:
            print('binary encoding feature: {0}'.format(feat))
            train_x, valid_x = fe_binary_encoding(train_x, valid_x, feat)
        elif cat_op==2:
            print('frequency encoding feature: {0}'.format(feat))
            train_x, valid_x=fe_frequency_encoding(train_x, valid_x, train, feat)
        elif cat_op == 3:
            print('target rate encoding feature: {0}'.format(feat))
            train_x, valid_x = fe_target_rate_encoding(train_x, train_y, valid_x, feat)
    return train_x, valid_x

def fe_Feature_Scale(train_x, valid_x):
    train_x['age_scale']=getAge(train_x['age'])
    valid_x['age_scale']=getAge(valid_x['age'])
    train_x['renta_scale'] = getRenta(train_x['renta'])
    valid_x['renta_scale'] = getRenta(valid_x['renta'])
    train_x['antiguedad'] = getRenta(train_x['antiguedad'])
    valid_x['antiguedad'] = getRenta(valid_x['antiguedad'])
    return train_x, valid_x

def fe_label_binary_encoding(labels, feat):
    label_max = labels.max()
    bin_df = pd.DataFrame()
    bin_df['product_class'] = np.arange(label_max + 1)
    max_bin_len = len("{0:b}".format(label_max))
    binary = bin_df['product_class'].apply(lambda x: "{0:b}".format(x).zfill(max_bin_len))
    splitted = binary.apply(lambda x: pd.Series(list(x)).astype(np.uint8))
    splitted.columns = [feat + '_trbin_' + str(x) for x in splitted.columns]
    bin_df = bin_df.join(splitted)
    labels_df = pd.merge(pd.DataFrame(labels), bin_df, how='left', on=['product_class'])
    labels_df.drop(['product_class'], axis=1, inplace=True)
    labels_df.index = labels.index
    return labels_df, bin_df, splitted.columns.tolist()

def fe_likelihood(cv_train_x, cv_train_y, cv_valid_x, feat, classes_encode='binary'):
    if classes_encode == 'binary':
        cv_train_y_df, bin_df, col_list = fe_label_binary_encoding(cv_train_y, feat)
    cv_train_xy = cv_train_x.to_frame().join(cv_train_y_df)
    for colid in range(0, len(col_list)):
        col = col_list[colid]
        agg = cv_train_xy.groupby(feat)[col].agg({'count', 'sum'}).reset_index()
        agg[col] = agg['sum'] / agg['count']
        cv_valid_x = pd.merge(cv_valid_x, agg[[feat, col]], on=[feat], how='left').fillna(0)
    return cv_valid_x, col_list


def fe_target_rate_encoding(train_x, train_y, valid_x, feat):
    TRE_values_train = pd.DataFrame()
    TRE_values_valid = pd.DataFrame()

    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    count = 0
    for train_kf, valid_kf in kf.split(train_x):
        cv_train_x = train_x[feat].ix[train_kf]
        cv_train_y = train_y.ix[train_kf]
        cv_valid_x = train_x[feat].ix[valid_kf].to_frame().reset_index()
        cv_valid_y, col_list = fe_likelihood(cv_train_x, cv_train_y, cv_valid_x, feat)
        TRE_values_train = pd.concat([TRE_values_train, cv_valid_y], axis=0)
        count += 1
    TRE_values_train = TRE_values_train.sort_values(by='index')
    TRE_values_train.set_index(keys='index', drop=True, inplace=True)
    TRE_values_train.index.name = None

    local_valid_x = valid_x[feat].to_frame().reset_index()
    TRE_values_valid, col_list = fe_likelihood(train_x[feat], train_y, local_valid_x, feat)
    TRE_values_valid = TRE_values_valid.sort_values(by='index')
    TRE_values_valid.set_index(keys='index', drop=True, inplace=True)
    TRE_values_valid.index.name = None

    output_train_x = pd.concat([train_x, TRE_values_train[col_list]], axis=1)
    output_valid_x = pd.concat([valid_x, TRE_values_valid[col_list]], axis=1)
    return output_train_x, output_valid_x

def post_proc_swap_21_22(data):
    #todo: create routine to swap product 21 and product 22 in case predicted prob for 21 is greater than 22
    return data


def fe_binary_encoding_export(train, test, feats):
    feat_col = ['fecha_dato', 'ncodpers']
    for feat in feats:
        print('binary encoding feature: {0}'.format(feat))
        train, test = fe_binary_encoding(train, test, feat)
        train.drop(feat, axis=1, inplace=True)
        test.drop(feat, axis=1, inplace=True)
        for i in train.columns:
            if feat in i:
                feat_col.append(i)
        gc.collect()
    return train[feat_col], test[feat_col]


def fe_get_customer_change(data, months_list, id_feats, cust_feats, month_len):
    months=set_months()
    change_df=[]
    for month_id in months_list:
        min_month=month_id-month_len+1
        print(months[month_id].year, months[month_id].month, 'to', months[min_month].month)

        # if month_id==months_list[0]:
        #     data_m=data.loc[data.fecha_dato<months[month_id], ['ncodpers']+cust_feats]
        # else:
        data_m=data.loc[(data.fecha_dato<=months[month_id]) & (data.fecha_dato>=months[min_month]), ['ncodpers']+cust_feats]
        
        change_m=data_m.groupby('ncodpers')[cust_feats].agg({'std'}).fillna(0)
        change_m.columns=cust_feats
        change_m[change_m>0]=1
        change_m.reset_index(inplace=True)
        change_m['fecha_dato']=months[month_id]
        if month_id==months_list[0]:
            change_df=change_m
        else:
            change_df=pd.concat([change_df, change_m], axis=0)

    change_df=change_df[id_feats+cust_feats]

    return change_df


def feature_importance(data, model, imp_type='weight'):
    columns=['feature','feature_name']
    feats_df=pd.DataFrame(columns=columns)
    features=[]
    feature_names=[]
    for fid in range(0,len(data.columns)):
        features.append('f'+str(fid))
        feature_names.append(data.columns[fid])

    feats_df['feature']=features
    feats_df['feature_name']=feature_names 

    model_score=pd.DataFrame.from_dict(model.get_score(importance_type=imp_type), orient='index')
    model_score.reset_index(inplace=True)
    model_score.columns=['feature','score']

    model_score=pd.merge(model_score, feats_df, how='left', on=['feature'])
    model_score=model_score[['feature','feature_name','score']]
    model_score.sort_values(by='score', ascending=False, inplace=True)
    return model_score


def model_sharpe_ratio(benchmark, expected_score, std):
    sharpe_ratio=(expected_score-benchmark)/std
    return sharpe_ratio



def cos_sim(cv_train, cv_valid):
    # print(cv_train.shape, cv_valid.shape)
    sp_train=scipy.sparse.csr_matrix(cv_train)
    sp_valid=scipy.sparse.csr_matrix(cv_valid)
    cos_sim=cosine_similarity(sp_valid,sp_train,dense_output=False)
    # cs_max=cos_sim.max(axis=1)
    cs_mean=cos_sim.mean(axis=1)
    cs_min=cos_sim.min(axis=1)
    return cs_mean, cs_min

def fe_Kfold_cosine_sim(pvec, pid, n_split=10):    
    id_cols=['ncodpers', 'fecha_dato']
    output_df=pvec.loc[:, id_cols+['output']]
    output_df['cosim_mean_p'+str(pid)]=np.nan
    output_df['cosim_min_p'+str(pid)]=np.nan
    
    y=(pvec.output==pid).values
    X=pvec.drop(id_cols+['user_date', 'output'], axis=1).values
    skf=StratifiedKFold(n_splits=n_split, random_state=1234)
#     print(X.shape, y.shape)
    count=0
    for train_skf, valid_skf in skf.split(X, y):
        count+=1
        print('kfound round {: d}'.format(count))
        X_train, X_valid = X[train_skf], X[valid_skf]
        y_train, y_valid = y[train_skf], y[valid_skf]
        X_train=X_train[np.argwhere(y_train==True).flatten()]
        cs_max, cs_mean, cs_min=cos_sim(X_train, X_valid)
#         print(cs_max.shape, cs_mean.shape, cs_min.shape)
        output_df.loc[valid_skf, 'cosim_mean_p'+str(pid)]=np.asarray(cs_mean).reshape(-1)        
        output_df.loc[valid_skf, 'cosim_min_p'+str(pid)]=cs_min.toarray().flatten()
        gc.collect()
    return output_df

def fe_cosin_sim_train(data, feature_path, pid, n_split=10):
    output_df=fe_Kfold_cosine_sim(data, pid, n_split=n_split)
    feature_file=feature_path+'cos_sim_train_p'+str(pid)+'.feather'
    feather.write_dataframe(output_df.drop(['output'], axis=1), feature_file)
    return output_df




def fe_cosin_sim_test(tr_vec_df, te_vec_df, pid, feature_path, chunk_size=50000):
    id_cols=['ncodpers', 'fecha_dato']
    mean_feat='cosim_mean_p'+str(pid)
    min_feat='cosim_min_p'+str(pid)
    
    output_df=te_vec_df.loc[:,id_cols]
    output_df[mean_feat]=np.nan
    output_df[min_feat]=np.nan
    
    tr_y=(tr_vec_df.output==pid).values
    tr_vec=tr_vec_df.drop(id_cols+['output','user_date'], axis=1).values    
    tr_x=tr_vec[np.argwhere(tr_y==True).flatten()]
    te_vec=te_vec_df.drop(id_cols, axis=1).values
    
    chunk_number=int(len(te_vec)/chunk_size)
    split_indexes=(np.arange(chunk_number)+1)*chunk_size

    start_end_indexes=[]
    for count in range(0, len(split_indexes)):
        start_end_indexes.append([split_indexes[count]-chunk_size, split_indexes[count]-1])
        if count==len(split_indexes)-1:
            start_end_indexes.append([split_indexes[count], len(te_vec)-len(split_indexes)*chunk_size+split_indexes[count]-1])        
        
    te_vec_collections=np.split(te_vec, split_indexes)
    for count in range(0, len(te_vec_collections)):
        cs_mean, cs_min=feu.cos_sim(tr_x, te_vec_collections[count])
        print(start_end_indexes[count][0],start_end_indexes[count][1])
        output_df.loc[start_end_indexes[count][0]:start_end_indexes[count][1],mean_feat]=np.asarray(cs_mean).reshape(-1) 
        output_df.loc[start_end_indexes[count][0]:start_end_indexes[count][1],min_feat]=cs_min.toarray().flatten()    
    feature_file=feature_path+'cos_sim_test_p'+str(pid)+'.feather'
    feather.write_dataframe(output_df, feature_file)
    return output_df




def fe_cos_sim_feats_combine(pid_range, feature_path):
    id_cols=['ncodpers', 'fecha_dato']
    cos_sim_train=pd.DataFrame()
    cos_sim_test=pd.DataFrame()
    start_pid=pid_range[0]
    end_pid=pid_range[1]
    for pid in range(start_pid, end_pid+1):
        print('pid is {: d}'.format(pid))
        sub_train_file=feature_path+'cos_sim_train_p'+str(pid)+'.feather'
        sub_test_file=feature_path+'cos_sim_test_p'+str(pid)+'.feather'
        cos_sim_sub_train=feather.read_dataframe(sub_train_file)
        cos_sim_sub_test=feather.read_dataframe(sub_test_file)
        if pid==start_pid:
            cos_sim_train=cos_sim_sub_train
            cos_sim_test=cos_sim_sub_test
        else:
            cos_sim_train=pd.merge(cos_sim_train, cos_sim_sub_train, how='left', on=id_cols)
            cos_sim_test=pd.merge(cos_sim_test, cos_sim_sub_test, how='left', on=id_cols)
    return cos_sim_train, cos_sim_test