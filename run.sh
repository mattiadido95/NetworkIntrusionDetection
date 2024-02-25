#!/bin/bash

# Funzione per eseguire il comando tshark e salvare l'output in un file JSON
capture_traffic() {
    echo "Cattura del traffico di rete..."
    # run thsark commanda and save the output in a JSON file
    tshark -i any -a duration:300 -n -T json | tee out.json
}

# Funzione per eseguire il programma Python
execute_python_program() {
    echo "Esecuzione del programma Python..."
    python nome_del_programma_python.py capture.json
}

# Loop infinito per eseguire le azioni ogni 10 minuti
while true; do
    # Cattura il traffico di rete e salvalo in un file JSON
    capture_traffic

    # Esegui il programma Python
    execute_python_program

    # Attendi 10 minuti prima di eseguire nuovamente le azioni
    sleep 600
done
