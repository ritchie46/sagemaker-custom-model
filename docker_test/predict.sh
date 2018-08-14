#!/bin/bash

payload=x.csv
content=${2:-text/csv}

curl --data-binary @x.csv -H "Content-Type: ${content}" -v http://localhost:8080/invocations
