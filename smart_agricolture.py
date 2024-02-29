import os
import json
import csv
from datetime import datetime
import re


def traffic_prediction():
    traffic_preprocessing()
    fill_empty_values_with_zero('output_2024-02-25.csv')
    return 0


def traffic_preprocessing():
    with open('results\out.json', 'r') as file:
        logs = json.load(file)
        preprocessed_logs = preprocess_logs(logs)

    print("Pre-elaborazione completata con successo.")
    return 0

def hex_to_decimal(value):
    # Se il valore è esadecimale, convertilo in decimale
    if re.match(r'^0x[0-9a-fA-F]+$', value):
        return int(value, 16)
    else:
        return value

def find_values_recursive(obj, values_to_find, row):
    for key, value in obj.items():
        if isinstance(value, dict):
            # Se il valore è un dizionario, esploriamolo ricorsivamente
            find_values_recursive(value, values_to_find, row)
        elif key in values_to_find:
            # Se la chiave corrisponde a uno dei valori da cercare, aggiungiamola alla riga
            row[key] = hex_to_decimal(value)

def fill_empty_values_with_zero(csv_file):
    # Apriamo il file CSV in modalità lettura e scrittura
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Controlliamo ciascuna riga e sostituiamo una stringa vuota con zero
    for row in rows:
        for key, value in row.items():
            if value == '':
                row[key] = 0

    # Scriviamo le righe modificate nel file CSV
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def preprocess_logs(logs):

    values_to_find = ['arp.opcode', 'arp.hw.size', 'icmp.checksum', 'icmp.seq_le',
                      'icmp.unused', 'http.content_length', 'http.request.method',
                      'http.referer', 'http.request.version', 'http.response',
                      'http.tls_port', 'tcp.ack', 'tcp.ack_raw', 'tcp.checksum',
                      'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn',
                      'tcp.connection.synack', 'tcp.flags', 'tcp.flags.ack', 'tcp.len',
                      'tcp.seq', 'udp.stream', 'udp.time_delta', 'dns.qry.name',
                      'dns.qry.name.len', 'dns.qry.qu', 'dns.qry.type', 'dns.retransmission',
                      'dns.retransmit_request', 'dns.retransmit_request_in',
                      'mqtt.conack.flags', 'mqtt.conflag.cleansess', 'mqtt.conflags',
                      'mqtt.hdrflags', 'mqtt.len', 'mqtt.msg_decoded_as', 'mqtt.msgtype',
                      'mqtt.proto_len', 'mqtt.protoname', 'mqtt.topic', 'mqtt.topic_len',
                      'mqtt.ver', 'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id',
                      'Attack_label', 'Attack_type']

    found_values = []

    output_file = 'output_' + datetime.now().strftime('%Y-%m-%d') + '.csv'

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=values_to_find)
        writer.writeheader()

        for obj in logs:
            row = {}
            find_values_recursive(obj, values_to_find, row)
            writer.writerow(row)

    return logs


if __name__ == '__main__':
    traffic_prediction()
