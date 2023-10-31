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


def split_data(df):
    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)
    return train, test


def create_classifier(train):
    # create decision tree classifier
    clf = DecisionTreeClassifier()
    # train the classifier
    clf.fit(train.drop(columns=['Attack_type']), train['Attack_type'])
    return clf


def test_classifier(clf, test):
    test['Predicted'] = clf.predict(test.drop(columns=['Attack_type']))
    confusion_matrix = pd.crosstab(test['Attack_type'], test['Predicted'], rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.subplot(1, 2, 2)
    report = classification_report(test['Attack_type'], test['Predicted'])
    print(report)


def get_column_types(df):
    for column in df.columns:
        print(column, df[column].dtype)


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


if __name__ == '__main__':
    df = load_data()
    # print columns name
    # print(df.columns)

    # remove columns with data and time information
    df = clean(df)

    # prepare data for training
    df = preprocess(df)

    # create test and train data
    train, test = split_data(df)

    # train a classifier with the train data
    clf = create_classifier(train)

    # test the classifier with the test data and create a confusion matrix
    test_classifier(clf, test)
