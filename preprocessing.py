from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import classification_report


def load_data():
    df = pd.read_csv('data\ML-EdgeIIoT-dataset.csv', low_memory=False)
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


def preprocess(df):
    encode_text_dummy(df, 'http.request.method')
    encode_text_dummy(df, 'http.referer')
    encode_text_dummy(df, "http.request.version")
    encode_text_dummy(df, "dns.qry.name.len")
    encode_text_dummy(df, "mqtt.conack.flags")
    encode_text_dummy(df, "mqtt.protoname")
    encode_text_dummy(df, "mqtt.topic")
    return df


def get_df():
    df = load_data()
    df = clean(df)
    df = preprocess(df)
    return df
