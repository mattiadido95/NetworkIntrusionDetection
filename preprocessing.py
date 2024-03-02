import os

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import classification_report


def load_data():
    df = pd.read_csv('data\ML-EdgeIIoT-dataset.csv', low_memory=False)
    #  df = pd.read_csv('output_2024-02-25.csv', low_memory=False)
    return df


def clean(df):
    # remove columns with not relevant information
    temporal_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                        "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
                        "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
                        "tcp.dstport", "udp.port", "mqtt.msg"]
    df.drop(columns=temporal_columns, errors='ignore', inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df.drop_duplicates(subset=None, keep="first", inplace=True)
    shuffle(df)
    df.isna().sum()
    return df


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


def preprocess_columns(df):
    encode_text_dummy(df, 'http.request.method')
    encode_text_dummy(df, 'http.referer')
    encode_text_dummy(df, "http.request.version")
    encode_text_dummy(df, "dns.qry.name.len")
    encode_text_dummy(df, "mqtt.conack.flags")
    encode_text_dummy(df, "mqtt.protoname")
    encode_text_dummy(df, "mqtt.topic")
    return df


def get_df(six_class=False):
    if six_class:
        df = pd.read_csv('data\ML-EdgeIIoT-dataset-6-class.csv', low_memory=False)
        return df
    df = pd.read_csv('data\ML-EdgeIIoT-dataset-preprocessed.csv', low_memory=False)
    return df


def create_6_class_dataframe(df):
    labels = {
        'Normal': 'Normal',
        'DDoS_UDP': "DDoS",
        'DDoS_ICMP': "DDoS",
        'DDoS_HTTP': "DDoS",
        'DDoS_TCP': "DDoS",
        'Vulnerability_scanner': "Scanning",
        'Password': "Malware",
        'SQL_injection': "Injection",
        'Uploading': "Injection",
        'Backdoor': "Malware",
        'Port_Scanning': "Scanning",
        'XSS': "Injection",
        'Ransomware': "Malware",
        'Fingerprinting': "Scanning",
        'MITM': "MITM",
    }
    df['6_Class'] = df['Attack_type'].map(labels)
    df.drop(columns=['Attack_type', 'Attack_label'], inplace=True)
    file_path = os.path.join('data', 'ML-EdgeIIoT-dataset-6-class.csv')
    df.to_csv(file_path, index=False)


def extract_one_record_per_attack_type(df):
    unique_attack_types = df['Attack_type'].unique()
    records = []
    indexes_to_drop = []
    for attack_type in unique_attack_types:
        attack_records = df[df['Attack_type'] == attack_type].head(1)
        indexes_to_drop.append(attack_records.index.values[0])
        records.append(attack_records)

    final_test_dataset = pd.concat(records, ignore_index=True)
    updated_df = df.drop(indexes_to_drop)

    return final_test_dataset, updated_df


def run():
    df = load_data()  # read dataset
    df = clean(df)  # remove columns with not relevant information
    df = preprocess_columns(df)  # preprocess columns
    final_test_df, df = extract_one_record_per_attack_type(df)  # extract one record per attack type
    df.to_csv('data\ML-EdgeIIoT-dataset-preprocessed.csv', index=False)  # save preprocessed dataset
    create_6_class_dataframe(df)  # create 6 class dataset
    final_test_df.to_csv('data\ML-EdgeIIoT-dataset-final-test.csv', index=False)  # save final test dataset
