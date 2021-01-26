import csv
import numpy as np
import os
import pandas as pd
import sys
import sklearn.model_selection

#path = sys.argv[1]
def read_patients(path='C:/Users/nealm/Dropbox (Brown)/MIMIC DATA/mimic-iii-clinical-database-1.4/mimic-iii-clinical-database-1.4'):
    patients = pd.read_csv(os.path.join(path, 'PATIENTS.csv.gz'), compression="gzip", header=0, index_col=0)
    patients = patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    patients.DOB = pd.to_datetime(patients.DOB)
    patients.DOD = pd.to_datetime(patients.DOD)
    return patients


def read_admissions(path='C:/Users/nealm/Dropbox (Brown)/MIMIC DATA/mimic-iii-clinical-database-1.4/mimic-iii-clinical-database-1.4'):
    admits = pd.read_csv(os.path.join(path, 'ADMISSIONS.csv.gz'), compression="gzip", header=0, index_col=0)
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits

def read_events(subj_ids, path='C:/Users/nealm/Dropbox (Brown)/MIMIC DATA/mimic-iii-clinical-database-1.4/mimic-iii-clinical-database-1.4'):
    file = os.path.join(path, 'CHARTEVENTS.csv.gz')
    df = pd.DataFrame()
    counter=0
    for chunker in pd.read_csv(file, chunksize=100000, compression="gzip", header=0, index_col=0):
        if counter==0:
            temp = chunker
            counter += 1
        elif counter == 210:
            break
        else:
            temp = temp.append(chunker)
            counter += 1
        print(counter)
        df = temp
        
    df = df[df['SUBJECT_ID'].isin(subj_ids)]
    print(df)
    df.to_csv('chartevents.csv')
    return 0


def read_icustays(path='C:/Users/nealm/Dropbox (Brown)/MIMIC DATA/mimic-iii-clinical-database-1.4/mimic-iii-clinical-database-1.4'):
    stays = pd.read_csv(os.path.join(path, 'ICUSTAYS.csv.gz'), compression="gzip", header=0, index_col=0)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays

def read_notes(path='C:/Users/nealm/Dropbox (Brown)/MIMIC DATA/mimic-iii-clinical-database-1.4/mimic-iii-clinical-database-1.4'):
    notes = pd.read_csv(os.path.join(path, 'NOTEEVENTS.csv.gz'), compression="gzip", header=0, index_col=0)
    notes['CHARTDATE'] = pd.to_datetime(notes['CHARTDATE'])
    notes['CHARTTIME'] = pd.to_datetime(notes['CHARTTIME'])
    notes['STORETIME'] = pd.to_datetime(notes['STORETIME'])
    notes = notes[notes['CHARTTIME'].notnull()]
    notes = notes[notes['HADM_ID'].notnull()]
    notes = notes[notes['SUBJECT_ID'].notnull()]
    notes = notes[notes['TEXT'].notnull()]
    notes = notes[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]
    return notes

def filter_stays(icu_stays):
    icu_stays = icu_stays[(icu_stays.FIRST_WARDID == icu_stays.LAST_WARDID) & (icu_stays.FIRST_CAREUNIT == icu_stays.LAST_CAREUNIT)]
    return  icu_stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]

def merge_tables(stays, admissions, patients):
    table = stays.merge(admissions, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
    table = table.merge(patients, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])
    return table


def add_age_mortality(icu_stays):

    icu_stays['INTIME'] = pd.to_datetime(icu_stays['INTIME']).dt.date

    icu_stays['DOB'] = pd.to_datetime(icu_stays['DOB']).dt.date
    icu_stays['AGE'] = icu_stays.apply(lambda e: np.timedelta64(e['INTIME'] - e['DOB']) / np.timedelta64(1, 'h'), axis=1)
    mortality = icu_stays['DOD'].notnull() & (icu_stays['ADMITTIME'] <= icu_stays['DOD']) & (icu_stays['DISCHTIME'] >= icu_stays['DOD']) | (icu_stays['DEATHTIME'].notnull() & (icu_stays['ADMITTIME'] <= icu_stays['DEATHTIME']) & (icu_stays['DISCHTIME'] >= icu_stays['DEATHTIME']))
    icu_stays['MORTALITY'] = mortality.astype(int)
    return icu_stays
    
def save_data(X_train, X_test, y_train, y_test, notes):
    X_train.to_csv('X_train.csv')
    X_test.to_csv('X_test.csv')
    y_train.to_csv('y_train.csv')
    y_test.to_csv('y_test.csv')
    notes.to_csv('clinical_notes.csv')

def unique_hadm(icu_stays, notes):
    unique_hadm = np.unique(icu_stays['HADM_ID'])
    notes = notes[notes['HADM_ID'] in unique_hadm]
    return notes

def main():
    
    
    patients = read_patients()
    subj_id = np.unique(patients['SUBJECT_ID'])
    events = read_events(subj_id)
    print(events)
    notes = read_notes()
    admissions = read_admissions()
    icu_stays = read_icustays()
    icu_stays = filter_stays(icu_stays)
    icu_stays = merge_tables(icu_stays, admissions, patients)
    icu_stays = add_age_mortality(icu_stays)
    icu_stays.to_csv('icu_stays')
    labels = icu_stays['MORTALITY']
    icu_stays = icu_stays.drop(['MORTALITY'], axis=1)
    #notes = unique_hadm(icu_stays, notes)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(icu_stays, labels, test_size=0.2, shuffle=True)
    save_data(X_train, X_test, y_train, y_test, notes)
    
if __name__ == '__main__':
    main()
