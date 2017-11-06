How to run the code:

- Prerequisites:

1. Copy all .py files inside a working directory
2. Make sure util.py is in the parent directory or copy it to the working directory
3. Create the following directory structure inside the working directory

   report/
       - img/
       - indicators/
       - notes/

- Indicators

PYTHONPATH=../:. python indicators.py

This will generate all graphs for the indicators inside report/img/

- Best Possible Strategy

PYTHONPATH=../:. python BestPossibleStrategy.py

This will generate a performance graph inside report/img/

- Manual Strategy

PYTHONPATH=../:. python ManualStrategy.py

This will generate a performance graph (in-sample) inside report/img/
It will also write a CSV file with the orders generated (in-sample) to report/notes/
The portfolio stats will be displayed in the console

- Comparative Analysis

PYTHONPATH=../:. python comparative_analysis.py

This will generate a performance graph (out-of-sample) inside report/img/
It will also write a CSV file with the orders generated (out-of-sample) to report/notes/
The portfolio stats will be displayed in the console