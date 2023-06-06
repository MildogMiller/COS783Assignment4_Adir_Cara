# COS783 Assignment 3

```
Usage: bsep.py [options]

Options:
  -h, --help            show this help message and exit
  -t CSV, --train=CSV   Train a new model for this run using given CSV (csv
                        format ['text','suspect'])
  -k CSV, --train-out=CSV
                        Set the output directory/name of the new trained model
                        (requires the -t command to be used)
  -m MODEL, --model=MODEL
                        Use a model for this run
  -o OUTPUT, --output=OUTPUT
                        Set the output directory of the report
  -i INPUT, --input-text=INPUT
                        Set the input file as a txt file, with each line being
                        a new email
  -c INPUT, --input-csv=INPUT
                        Set the input file as a csv file ['id','email']
  --save-full           Include both the email and id in the output
  ```
